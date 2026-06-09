from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

import torch


Interval = Tuple[int, int]


@dataclass(frozen=True)
class InpaintingExample:
    input_ids: torch.Tensor
    target_tokens: torch.Tensor
    mask_interval: Interval
    loss_positions: torch.Tensor
    prefix: torch.Tensor
    middle: torch.Tensor
    suffix: torch.Tensor


def sample_mask_intervals(
    y_lens: Iterable[int] | torch.Tensor,
    mask_len_min: int,
    mask_len_max: int,
    max_n_spans: int,
    mask_sample_dist: str = "poisson1",
    min_gap: int = 5,
    generator: torch.Generator | None = None,
) -> tuple[list[list[Interval]], list[list[Interval]]]:
    """Sample non-overlapping random mask intervals for online augmentation.

    Intervals use frame indices in half-open form: ``[start, end)``.
    """
    lengths = _to_int_list(y_lens)
    if mask_len_min <= 0:
        raise ValueError("mask_len_min must be positive")
    if mask_len_max < mask_len_min:
        raise ValueError("mask_len_max must be >= mask_len_min")
    if max_n_spans <= 0:
        raise ValueError("max_n_spans must be positive")
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")

    mask_intervals: list[list[Interval]] = []
    non_mask_intervals: list[list[Interval]] = []
    for y_len in lengths:
        if y_len <= 0:
            raise ValueError(f"y_lens must be positive, got {y_len}")

        intervals = _sample_one_sequence(
            y_len=y_len,
            mask_len_min=min(mask_len_min, y_len),
            mask_len_max=min(mask_len_max, y_len),
            max_n_spans=max_n_spans,
            mask_sample_dist=mask_sample_dist,
            min_gap=min_gap,
            generator=generator,
        )
        mask_intervals.append(intervals)
        non_mask_intervals.append(_invert_intervals(intervals, y_len))

    return mask_intervals, non_mask_intervals


def sample_eval_mask_intervals(
    y_lens: Iterable[int] | torch.Tensor,
    mask_len: int,
    position: str = "center",
) -> tuple[list[list[Interval]], list[list[Interval]]]:
    """Create deterministic mask intervals for stable evaluation."""
    if mask_len <= 0:
        raise ValueError("mask_len must be positive")
    if position != "center":
        raise ValueError("only position='center' is supported")

    mask_intervals: list[list[Interval]] = []
    non_mask_intervals: list[list[Interval]] = []
    for y_len in _to_int_list(y_lens):
        if y_len <= 0:
            raise ValueError(f"y_lens must be positive, got {y_len}")
        length = min(mask_len, y_len)
        start = (y_len - length) // 2
        intervals = [(start, start + length)]
        mask_intervals.append(intervals)
        non_mask_intervals.append(_invert_intervals(intervals, y_len))
    return mask_intervals, non_mask_intervals


def build_voicecraftx_inpainting_input(
    audio_input_ids: torch.Tensor,
    mask_intervals: Sequence[Interval],
    speech_mask: torch.Tensor | None = None,
    speech_mask_idx: int | None = None,
) -> InpaintingExample:
    """Build VoiceCraft-X single-gap input: prefix + mask + suffix + mask + middle."""
    if audio_input_ids.dim() != 2:
        raise ValueError("audio_input_ids must have shape [num_codebooks, num_frames]")
    if len(mask_intervals) != 1:
        raise NotImplementedError("VoiceCraft-X initial augmentation supports one mask span")
    if speech_mask is None and speech_mask_idx is None:
        raise ValueError("provide speech_mask or speech_mask_idx")

    num_codebooks, seq_len = audio_input_ids.shape
    start, end = _validate_interval(mask_intervals[0], seq_len)

    prefix = audio_input_ids[:, :start]
    middle = audio_input_ids[:, start:end]
    suffix = audio_input_ids[:, end:]
    mask = _make_speech_mask(
        num_codebooks=num_codebooks,
        device=audio_input_ids.device,
        dtype=audio_input_ids.dtype,
        speech_mask=speech_mask,
        speech_mask_idx=speech_mask_idx,
    )

    input_ids = torch.cat([prefix, mask, suffix, mask, middle], dim=-1)
    middle_offset = prefix.shape[-1] + mask.shape[-1] + suffix.shape[-1] + mask.shape[-1]
    loss_positions = torch.arange(
        middle_offset,
        middle_offset + middle.shape[-1],
        device=audio_input_ids.device,
    )
    return InpaintingExample(
        input_ids=input_ids,
        target_tokens=middle,
        mask_interval=(start, end),
        loss_positions=loss_positions,
        prefix=prefix,
        middle=middle,
        suffix=suffix,
    )


def reconstruct_original(example: InpaintingExample) -> torch.Tensor:
    """Rebuild the original token sequence from an inpainting example."""
    return torch.cat([example.prefix, example.middle, example.suffix], dim=-1)


def pad_voicecraftx_inpainting_batch(
    examples: Sequence[InpaintingExample],
    pad_token: int,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Pad variable-length inpainting examples and mark raw middle positions."""
    if not examples:
        raise ValueError("examples must not be empty")

    num_codebooks = examples[0].input_ids.shape[0]
    max_len = max(example.input_ids.shape[-1] for example in examples)
    batch_size = len(examples)
    device = examples[0].input_ids.device

    input_ids = torch.full(
        (batch_size, num_codebooks, max_len),
        pad_token,
        dtype=examples[0].input_ids.dtype,
        device=device,
    )
    labels = torch.full(
        (batch_size, num_codebooks, max_len),
        ignore_index,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for batch_idx, example in enumerate(examples):
        if example.input_ids.shape[0] != num_codebooks:
            raise ValueError("all examples must have the same num_codebooks")
        seq_len = example.input_ids.shape[-1]
        input_ids[batch_idx, :, :seq_len] = example.input_ids
        attention_mask[batch_idx, :seq_len] = True
        labels[batch_idx, :, example.loss_positions] = example.target_tokens.long()
        loss_mask[batch_idx, example.loss_positions] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }


def _sample_one_sequence(
    y_len: int,
    mask_len_min: int,
    mask_len_max: int,
    max_n_spans: int,
    mask_sample_dist: str,
    min_gap: int,
    generator: torch.Generator | None,
) -> list[Interval]:
    n_spans = _sample_num_spans(max_n_spans, mask_sample_dist, generator)
    intervals: list[Interval] = []
    for _ in range(n_spans):
        accepted = False
        for _attempt in range(100):
            length = _randint(mask_len_min, mask_len_max + 1, generator)
            start = _randint(0, y_len - length + 1, generator)
            candidate = (start, start + length)
            if _has_min_gap(candidate, intervals, min_gap):
                intervals.append(candidate)
                intervals.sort()
                accepted = True
                break
        if not accepted:
            break

    if intervals:
        return intervals

    length = min(mask_len_max, max(mask_len_min, y_len))
    start = max(0, (y_len - length) // 2)
    return [(start, start + length)]


def _sample_num_spans(
    max_n_spans: int,
    mask_sample_dist: str,
    generator: torch.Generator | None,
) -> int:
    if mask_sample_dist == "uniform":
        return _randint(1, max_n_spans + 1, generator)
    if mask_sample_dist.startswith("poisson"):
        lam_text = mask_sample_dist[len("poisson"):] or "1"
        lam = float(lam_text)
        if lam <= 0 or not math.isfinite(lam):
            raise ValueError("poisson lambda must be positive")
        value = int(torch.poisson(torch.tensor(lam), generator=generator).item())
        return min(max(value, 1), max_n_spans)
    raise ValueError(f"unsupported mask_sample_dist: {mask_sample_dist}")


def _randint(low: int, high: int, generator: torch.Generator | None) -> int:
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _to_int_list(values: Iterable[int] | torch.Tensor) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().flatten().tolist()]
    return [int(value) for value in values]


def _validate_interval(interval: Interval, seq_len: int) -> Interval:
    start, end = int(interval[0]), int(interval[1])
    if not 0 <= start < end <= seq_len:
        raise ValueError(f"invalid interval {(start, end)} for length {seq_len}")
    return start, end


def _invert_intervals(intervals: Sequence[Interval], seq_len: int) -> list[Interval]:
    non_mask: list[Interval] = []
    cursor = 0
    for start, end in sorted(intervals):
        _validate_interval((start, end), seq_len)
        if cursor < start:
            non_mask.append((cursor, start))
        cursor = end
    if cursor < seq_len:
        non_mask.append((cursor, seq_len))
    return non_mask


def _has_min_gap(candidate: Interval, intervals: Sequence[Interval], min_gap: int) -> bool:
    c_start, c_end = candidate
    for start, end in intervals:
        if c_start < end + min_gap and start < c_end + min_gap:
            return False
    return True


def _make_speech_mask(
    num_codebooks: int,
    device: torch.device,
    dtype: torch.dtype,
    speech_mask: torch.Tensor | None,
    speech_mask_idx: int | None,
) -> torch.Tensor:
    if speech_mask is not None:
        mask = speech_mask.to(device=device, dtype=dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        if mask.shape != (num_codebooks, 1):
            raise ValueError(f"speech_mask must have shape [{num_codebooks}, 1]")
        return mask
    return torch.full((num_codebooks, 1), int(speech_mask_idx), dtype=dtype, device=device)
