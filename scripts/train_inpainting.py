from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Sequence

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset, VoiceCraftXTokenizedWDSCollator  # noqa: E402
from helper import load_tokenizer, load_voicecraftx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VoiceCraft-X on online speech inpainting batches.")
    parser.add_argument("--config", default="src/config/inference/tts.yaml")
    parser.add_argument("--wds-root", default="data/tokenized_wds/train-clean-100")
    parser.add_argument("--output-dir", default="exp/inpainting_debug")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-split-size", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--mask-len-min", type=int, default=10)
    parser.add_argument("--mask-len-max", type=int, default=150)
    parser.add_argument("--max-n-spans", type=int, default=1)
    parser.add_argument("--mask-sample-dist", default="poisson1")
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--deterministic-mask", action="store_true")
    parser.add_argument("--eval-mask-len", type=int, default=20)
    parser.add_argument("--language", default="english")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_tokenizer, _ = load_tokenizer(config)
    model = load_voicecraftx(config).to(device)
    model.train()
    if args.freeze_llm:
        for param in model.llm.parameters():
            param.requires_grad = False
        model.llm.eval()

    dataset = TokenizedLibriTTSRWDSDataset(args.wds_root)
    selected_size = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
    if args.eval_split_size < 0:
        raise ValueError("--eval-split-size must be non-negative")
    if args.eval_split_size >= selected_size:
        raise ValueError("--eval-split-size must be smaller than the selected dataset size")

    train_indices = list(range(selected_size - args.eval_split_size))
    eval_indices = list(range(selected_size - args.eval_split_size, selected_size))
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices) if eval_indices else None

    collator = VoiceCraftXTokenizedWDSCollator(
        speech_mask_idx=int(config.model.speech_mask_idx),
        pad_token=int(config.model.speech_empty_idx),
        mask_len_min=args.mask_len_min,
        mask_len_max=args.mask_len_max,
        max_n_spans=args.max_n_spans,
        mask_sample_dist=args.mask_sample_dist,
        min_gap=args.min_gap,
        eval_mask_len=args.eval_mask_len,
        deterministic_eval=args.deterministic_mask,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_collator = VoiceCraftXTokenizedWDSCollator(
            speech_mask_idx=int(config.model.speech_mask_idx),
            pad_token=int(config.model.speech_empty_idx),
            eval_mask_len=args.eval_mask_len,
            deterministic_eval=True,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_collator,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    step = 0
    examples_seen = 0
    loss_sum = 0.0
    train_start = time.perf_counter()
    dataset_size = len(train_dataset)
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in dataloader:
            text, text_attention_mask = encode_texts(
                text_tokenizer=text_tokenizer,
                texts=batch["texts"],
                language=args.language,
                pad_token_id=int(config.model.llm_padding_idx),
            )
            speaker_embeddings = batch["speaker_embeddings"]
            if not isinstance(speaker_embeddings, torch.Tensor):
                raise RuntimeError("speaker embeddings are required for VoiceCraft-X training")

            text = text.to(device)
            text_attention_mask = text_attention_mask.to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            speaker_embeddings = speaker_embeddings.to(device)

            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(
                    text=text,
                    text_attention_mask=text_attention_mask,
                    prompt_speech_token=input_ids,
                    speaker_emb=speaker_embeddings,
                    labels=labels,
                    return_logits=False,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            grad_norm = None
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            batch_size = int(input_ids.shape[0])
            examples_seen += batch_size
            loss_value = float(loss.detach().cpu().item() * args.gradient_accumulation_steps)
            loss_sum += loss_value
            if step % args.log_every == 0:
                target_tokens = int(batch["loss_mask"].sum().item())
                metrics = {
                    "loss": _format_metric(loss_value),
                    "grad_norm": _format_metric(_to_float(grad_norm)),
                    "learning_rate": _format_metric(optimizer.param_groups[0]["lr"]),
                    "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
                    "target_frames": str(target_tokens),
                    "step": str(step),
                }
                print(metrics, flush=True)

            if eval_dataloader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = evaluate(
                    model=model,
                    dataloader=eval_dataloader,
                    text_tokenizer=text_tokenizer,
                    config=config,
                    args=args,
                    device=device,
                )
                eval_metrics["epoch"] = _format_metric(examples_seen / max(dataset_size, 1))
                eval_metrics["step"] = str(step)
                print(eval_metrics, flush=True)

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(output_dir / f"step-{step:06d}.pt", model, optimizer, step, args)

            if step >= args.max_steps:
                break

    save_checkpoint(output_dir / "last.pt", model, optimizer, step, args)
    train_runtime = time.perf_counter() - train_start
    train_metrics = {
        "train_runtime": _format_metric(train_runtime),
        "train_samples_per_second": _format_metric(examples_seen / train_runtime if train_runtime > 0 else 0.0),
        "train_steps_per_second": _format_metric(step / train_runtime if train_runtime > 0 else 0.0),
        "train_loss": _format_metric(loss_sum / max(step, 1)),
        "epoch": _format_metric(examples_seen / max(dataset_size, 1)),
    }
    print(train_metrics, flush=True)



@torch.no_grad()
def evaluate(model, dataloader, text_tokenizer, config, args, device):
    was_training = model.training
    model.eval()
    eval_start = time.perf_counter()
    loss_sum = 0.0
    samples = 0
    steps = 0
    for batch in dataloader:
        if args.eval_max_batches is not None and steps >= args.eval_max_batches:
            break
        text, text_attention_mask = encode_texts(
            text_tokenizer=text_tokenizer,
            texts=batch["texts"],
            language=args.language,
            pad_token_id=int(config.model.llm_padding_idx),
        )
        speaker_embeddings = batch["speaker_embeddings"]
        if not isinstance(speaker_embeddings, torch.Tensor):
            raise RuntimeError("speaker embeddings are required for VoiceCraft-X eval")

        text = text.to(device)
        text_attention_mask = text_attention_mask.to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        speaker_embeddings = speaker_embeddings.to(device)

        outputs = model(
            text=text,
            text_attention_mask=text_attention_mask,
            prompt_speech_token=input_ids,
            speaker_emb=speaker_embeddings,
            labels=labels,
            return_logits=False,
        )
        batch_size = int(input_ids.shape[0])
        loss_sum += float(outputs["loss"].detach().cpu().item()) * batch_size
        samples += batch_size
        steps += 1

    runtime = time.perf_counter() - eval_start
    if was_training:
        model.train()
        if args.freeze_llm:
            model.llm.eval()
    return {
        "eval_loss": _format_metric(loss_sum / max(samples, 1)),
        "eval_runtime": _format_metric(runtime),
        "eval_samples_per_second": _format_metric(samples / runtime if runtime > 0 else 0.0),
        "eval_steps_per_second": _format_metric(steps / runtime if runtime > 0 else 0.0),
    }


def encode_texts(text_tokenizer, texts: Sequence[str], language: str, pad_token_id: int):
    encoded = []
    for text in texts:
        transcript = f"<|fim_prefix|><|fim_suffix|><|fim_middle|>{text}"
        transcript = text_tokenizer.text_normalize(
            transcript,
            split=False,
            text_frontend=True,
            lang=language,
        )
        token_ids, _ = text_tokenizer(transcript)
        encoded.append(token_ids.reshape(-1).long())

    max_len = max(ids.numel() for ids in encoded)
    batch = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    for idx, ids in enumerate(encoded):
        batch[idx, : ids.numel()] = ids
        attention_mask[idx, : ids.numel()] = True
    return batch, attention_mask


def save_checkpoint(path: Path, model, optimizer, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def _to_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _format_metric(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e4:
        return f"{value:.4g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
