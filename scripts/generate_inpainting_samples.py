from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
import torchaudio
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.online_masking import build_voicecraftx_inpainting_input, sample_eval_mask_intervals  # noqa: E402
from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset  # noqa: E402
from helper import load_tokenizer, load_voicecraftx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate fixed validation inpainting samples for listening checks.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", default="exp/inpainting_generation_check")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--eval-split-size", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--eval-mask-len", type=int, default=20)
    parser.add_argument("--max-new-frames", type=int, default=80)
    parser.add_argument("--n-generations", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--language", default="english")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    device = _resolve_device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    text_tokenizer, audio_tokenizer = load_tokenizer(config)
    audio_tokenizer = audio_tokenizer.to(device)

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_size = min(args.limit, len(dataset))
    if args.eval_split_size <= 0:
        raise ValueError("--eval-split-size must be positive for validation generation")
    if args.eval_split_size >= selected_size:
        raise ValueError("--eval-split-size must be smaller than selected dataset size")

    eval_start = selected_size - args.eval_split_size
    sample_indices = list(range(eval_start + args.sample_offset, selected_size))
    sample_indices = sample_indices[: args.num_samples]
    if not sample_indices:
        raise RuntimeError("no validation samples selected")

    sample_items = [dataset[index] for index in sample_indices]
    y_lens = [int(item["tokens"].shape[-1]) for item in sample_items]
    mask_intervals, _ = sample_eval_mask_intervals(y_lens, mask_len=args.eval_mask_len)

    for checkpoint in args.checkpoint:
        checkpoint_path = Path(checkpoint)
        checkpoint_name = checkpoint_path.stem
        run_dir = output_root / checkpoint_name
        run_dir.mkdir(parents=True, exist_ok=True)

        model = load_voicecraftx(config).to(device)
        _load_training_checkpoint(model, checkpoint_path)
        model.eval()

        for local_idx, (item, intervals) in enumerate(zip(sample_items, mask_intervals)):
            sample_dir = run_dir / f"{local_idx:02d}_{item['utterance_id']}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            tokens = item["tokens"].long()
            example = build_voicecraftx_inpainting_input(
                tokens,
                intervals,
                speech_mask_idx=int(config.model.speech_mask_idx),
            )
            text = encode_text(
                text_tokenizer=text_tokenizer,
                text=item["text"],
                language=args.language,
            ).to(device)
            prompt_tokens = example.input_ids.unsqueeze(0).to(device)
            speaker_embedding = item["speaker_embedding"]
            if not isinstance(speaker_embedding, torch.Tensor):
                raise RuntimeError(f"speaker embedding missing for {item['utterance_id']}")
            speaker_embedding = speaker_embedding.to(device)

            generation_config = {
                "max_length": args.max_new_frames,
                "min_p": args.min_p,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
            }
            with torch.inference_mode():
                generated_spans = model.generate(
                    n_samples=args.n_generations,
                    text=text,
                    prompt_speech_token=prompt_tokens,
                    speaker_emb=speaker_embedding,
                    generation_config=generation_config,
                )

            original_wav = _load_original_wav(item["wav_path"], int(config.SAMPLE_RATE))
            masked_wav = _make_masked_wav(
                original_wav,
                intervals[0],
                sample_rate=int(config.SAMPLE_RATE),
                codec_frame_rate=int(config.CODEC_FRAME_RATE),
            )
            _save_wav(sample_dir / "original.wav", original_wav, int(config.SAMPLE_RATE))
            _save_wav(sample_dir / "masked_input.wav", masked_wav, int(config.SAMPLE_RATE))

            gap_len = intervals[0][1] - intervals[0][0]
            generated_metadata = []
            for gen_idx, generated_span in enumerate(generated_spans):
                generated_span = generated_span.detach().cpu().long()
                matched_span = _match_span_length(
                    generated_span,
                    target_len=gap_len,
                    silence_tokens=tuple(int(x) for x in config.model.silence_tokens),
                )
                generated_full = torch.cat([example.prefix.cpu(), generated_span, example.suffix.cpu()], dim=-1)
                matched_full = torch.cat([example.prefix.cpu(), matched_span, example.suffix.cpu()], dim=-1)
                generated_wav = _decode_tokens(audio_tokenizer, generated_full.to(device))
                generated_gap_wav = _try_decode_tokens(audio_tokenizer, generated_span.to(device))
                matched_wav = _decode_tokens(audio_tokenizer, matched_full.to(device))
                matched_gap_wav = _try_decode_tokens(audio_tokenizer, matched_span.to(device))

                suffix = "" if args.n_generations == 1 else f"_{gen_idx:02d}"
                _save_wav(sample_dir / f"generated{suffix}.wav", generated_wav, int(config.SAMPLE_RATE))
                if generated_gap_wav is not None:
                    _save_wav(sample_dir / f"generated_gap{suffix}.wav", generated_gap_wav, int(config.SAMPLE_RATE))
                _save_wav(sample_dir / f"generated_matched{suffix}.wav", matched_wav, int(config.SAMPLE_RATE))
                if matched_gap_wav is not None:
                    _save_wav(sample_dir / f"generated_gap_matched{suffix}.wav", matched_gap_wav, int(config.SAMPLE_RATE))
                generated_metadata.append(
                    {
                        "generation_index": gen_idx,
                        "generated_frames": int(generated_span.shape[-1]),
                        "matched_frames": int(matched_span.shape[-1]),
                        "generated_full_frames": int(generated_full.shape[-1]),
                        "matched_full_frames": int(matched_full.shape[-1]),
                        "generated_gap_wav_saved": generated_gap_wav is not None,
                        "matched_gap_wav_saved": matched_gap_wav is not None,
                    }
                )

            metadata = {
                "checkpoint": str(checkpoint_path),
                "utterance_id": item["utterance_id"],
                "speaker_id": item.get("speaker_id"),
                "chapter_id": item.get("chapter_id"),
                "text": item["text"],
                "wav_path": item["wav_path"],
                "original_frames": int(tokens.shape[-1]),
                "mask_interval": list(intervals[0]),
                "mask_start_sec": intervals[0][0] / float(config.CODEC_FRAME_RATE),
                "mask_end_sec": intervals[0][1] / float(config.CODEC_FRAME_RATE),
                "generated": generated_metadata,
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
            print(f"wrote {sample_dir}", flush=True)


def encode_text(text_tokenizer, text: str, language: str) -> torch.Tensor:
    transcript = f"<|fim_prefix|><|fim_suffix|><|fim_middle|>{text}"
    transcript = text_tokenizer.text_normalize(
        transcript,
        split=False,
        text_frontend=True,
        lang=language,
    )
    token_ids, _ = text_tokenizer(transcript)
    return token_ids.reshape(1, -1).long()


def _load_training_checkpoint(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"warning: unexpected checkpoint keys: {len(unexpected)}", flush=True)
    if missing:
        print(f"warning: missing checkpoint keys: {len(missing)}", flush=True)


def _decode_tokens(audio_tokenizer, tokens: torch.Tensor) -> torch.Tensor:
    wav = audio_tokenizer.decode(tokens.unsqueeze(0))
    return wav.detach().cpu()


def _try_decode_tokens(audio_tokenizer, tokens: torch.Tensor) -> torch.Tensor | None:
    try:
        return _decode_tokens(audio_tokenizer, tokens)
    except RuntimeError as exc:
        print(f"warning: skipped short gap decode: {exc}", flush=True)
        return None


def _match_span_length(
    span: torch.Tensor,
    target_len: int,
    silence_tokens: tuple[int, ...],
) -> torch.Tensor:
    if span.shape[-1] == target_len:
        return span
    if span.shape[-1] > target_len:
        return span[:, :target_len]
    pad_len = target_len - span.shape[-1]
    pad_values = torch.tensor(silence_tokens, dtype=span.dtype).view(-1, 1).expand(-1, pad_len)
    return torch.cat([span, pad_values], dim=-1)


def _load_original_wav(wav_path: str | None, sample_rate: int) -> torch.Tensor:
    if wav_path is None:
        raise RuntimeError("wav_path is missing from metadata")
    path = Path(wav_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.unsqueeze(0)


def _make_masked_wav(
    wav: torch.Tensor,
    interval: tuple[int, int],
    sample_rate: int,
    codec_frame_rate: int,
) -> torch.Tensor:
    masked = wav.clone()
    start, end = interval
    start_sample = round(start * sample_rate / codec_frame_rate)
    end_sample = round(end * sample_rate / codec_frame_rate)
    masked[..., start_sample:end_sample] = 0.0
    return masked


def _save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().cpu()
    while wav.dim() > 2:
        wav = wav.squeeze(0)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav, sample_rate)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
