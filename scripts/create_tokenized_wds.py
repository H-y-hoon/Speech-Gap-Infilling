from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import torch
import torchaudio
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.libritts_r import LibriTTSRDataset  # noqa: E402
from helper import extract_speaker_embedding, load_speaker_model, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tokenized WebDataset shards for VoiceCraft-X LibriTTS-R training."
    )
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--root", default="data/LibriTTS_R")
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--output-dir", default="data/tokenized_wds/train-clean-100")
    parser.add_argument("--prefix", default="librittsr-train-clean-100")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-speaker-embedding", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_output_dir(output_dir, args.overwrite)

    dataset = LibriTTSRDataset(args.root, split=args.split)
    end_index = len(dataset) if args.limit is None else min(len(dataset), args.start_index + args.limit)
    if args.start_index < 0 or args.start_index >= len(dataset):
        raise ValueError(f"invalid start index {args.start_index} for dataset size {len(dataset)}")
    if end_index <= args.start_index:
        raise ValueError("no samples selected")

    _, audio_tokenizer = load_tokenizer(config)
    device = _resolve_device(args.device)
    audio_tokenizer = audio_tokenizer.to(device).eval()
    speaker_model = None if args.no_speaker_embedding else load_speaker_model(config)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(Path(args.root).resolve()),
        "split": args.split,
        "start_index": args.start_index,
        "end_index": end_index,
        "num_samples": end_index - args.start_index,
        "shard_size": args.shard_size,
        "codec": {
            "checkpoint": str(Path(config.pretrained_models, "encodec.th").resolve()),
            "sample_rate": int(config.SAMPLE_RATE),
            "codec_frame_rate": int(config.CODEC_FRAME_RATE),
            "num_codebooks": int(config.model.num_codebooks),
            "speech_token_size": int(config.model.speech_token_size),
        },
        "contains": ["tokens.pt", "txt", "json"]
        if args.no_speaker_embedding
        else ["tokens.pt", "spk.pt", "txt", "json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    shard = None
    shard_path = None
    written = 0
    try:
        for sample_index in range(args.start_index, end_index):
            if written % args.shard_size == 0:
                if shard is not None:
                    shard.close()
                shard_id = written // args.shard_size
                shard_path = output_dir / f"{args.prefix}-{shard_id:06d}.tar"
                shard = tarfile.open(shard_path, mode="w")

            item = dataset[sample_index]
            sample = _process_item(
                item=item,
                audio_tokenizer=audio_tokenizer,
                speaker_model=speaker_model,
                sample_rate=int(config.SAMPLE_RATE),
                device=device,
                sample_index=sample_index,
                split=args.split,
            )
            _write_sample(shard, sample)
            written += 1

            if written == 1 or written % 100 == 0:
                print(
                    f"wrote {written}/{end_index - args.start_index} samples "
                    f"(latest={item.utterance_id}, shard={shard_path.name})",
                    flush=True,
                )
    finally:
        if shard is not None:
            shard.close()

    print(f"done: wrote {written} samples to {output_dir}", flush=True)


def _check_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing_shards = sorted(output_dir.glob("*.tar"))
    if existing_shards and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains tar shards; pass --overwrite or choose another output dir"
        )
    if overwrite:
        for shard in existing_shards:
            shard.unlink()


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _process_item(
    item,
    audio_tokenizer,
    speaker_model,
    sample_rate: int,
    device: torch.device,
    sample_index: int,
    split: str,
) -> dict[str, Any]:
    wav, sr = torchaudio.load(item.wav_path)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)

    with torch.no_grad():
        tokens = audio_tokenizer(wav.unsqueeze(0).to(device))[0].detach().cpu().long()
    if tokens.dim() != 2:
        raise RuntimeError(f"expected tokens [K, T], got {tuple(tokens.shape)} for {item.wav_path}")

    sample = {
        "__key__": item.utterance_id,
        "tokens.pt": tokens,
        "txt": item.text,
        "json": {
            "utterance_id": item.utterance_id,
            "speaker_id": item.speaker_id,
            "chapter_id": item.chapter_id,
            "wav_path": item.wav_path,
            "sample_index": sample_index,
            "split": split,
            "sample_rate": sample_rate,
            "num_codebooks": int(tokens.shape[0]),
            "num_frames": int(tokens.shape[1]),
            "duration_sec": float(wav.shape[-1] / sample_rate),
        },
    }
    if speaker_model is not None:
        sample["spk.pt"] = extract_speaker_embedding(speaker_model, wav).detach().cpu().float()
    return sample


def _write_sample(tar: tarfile.TarFile, sample: dict[str, Any]) -> None:
    key = sample["__key__"]
    _add_bytes(tar, f"{key}.tokens.pt", _tensor_bytes(sample["tokens.pt"]))
    if "spk.pt" in sample:
        _add_bytes(tar, f"{key}.spk.pt", _tensor_bytes(sample["spk.pt"]))
    _add_bytes(tar, f"{key}.txt", sample["txt"].encode("utf-8"))
    _add_bytes(tar, f"{key}.json", json.dumps(sample["json"], ensure_ascii=False).encode("utf-8"))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    main()
