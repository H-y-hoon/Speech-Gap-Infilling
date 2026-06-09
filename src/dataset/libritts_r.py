from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torchaudio

from .online_masking import (
    build_voicecraftx_inpainting_input,
    pad_voicecraftx_inpainting_batch,
    sample_eval_mask_intervals,
    sample_mask_intervals,
)


@dataclass(frozen=True)
class LibriTTSRItem:
    wav_path: str
    text: str
    utterance_id: str
    speaker_id: str
    chapter_id: str


class LibriTTSRDataset(torch.utils.data.Dataset):
    """LibriTTS-R file index for VoiceCraft-X experiments."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train-clean-100",
        text_suffix: str = ".normalized.txt",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.text_suffix = text_suffix
        split_root = self.root / split
        if not split_root.exists():
            raise FileNotFoundError(f"LibriTTS-R split not found: {split_root}")

        self.items = self._scan(split_root)
        if not self.items:
            raise RuntimeError(f"no wav/text pairs found under {split_root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> LibriTTSRItem:
        return self.items[index]

    def _scan(self, split_root: Path) -> list[LibriTTSRItem]:
        items: list[LibriTTSRItem] = []
        for wav_path in sorted(split_root.rglob("*.wav")):
            text_path = wav_path.with_suffix("")
            text_path = text_path.with_name(text_path.name + self.text_suffix)
            if not text_path.exists():
                continue
            utterance_id = wav_path.stem
            speaker_id = wav_path.parent.parent.name
            chapter_id = wav_path.parent.name
            text = text_path.read_text(encoding="utf-8").strip()
            items.append(
                LibriTTSRItem(
                    wav_path=str(wav_path),
                    text=text,
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    chapter_id=chapter_id,
                )
            )
        return items


class VoiceCraftXOnlineMaskingCollator:
    """Encode LibriTTS-R audio and build VoiceCraft-X online inpainting batches."""

    def __init__(
        self,
        audio_tokenizer: Callable[[torch.Tensor], torch.Tensor],
        sample_rate: int,
        speech_mask_idx: int,
        pad_token: int,
        mask_len_min: int = 10,
        mask_len_max: int = 150,
        max_n_spans: int = 1,
        mask_sample_dist: str = "poisson1",
        min_gap: int = 5,
        eval_mask_len: int = 50,
        deterministic_eval: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.audio_tokenizer = audio_tokenizer
        self.sample_rate = sample_rate
        self.speech_mask_idx = speech_mask_idx
        self.pad_token = pad_token
        self.mask_len_min = mask_len_min
        self.mask_len_max = mask_len_max
        self.max_n_spans = max_n_spans
        self.mask_sample_dist = mask_sample_dist
        self.min_gap = min_gap
        self.eval_mask_len = eval_mask_len
        self.deterministic_eval = deterministic_eval
        self.generator = generator

    def __call__(self, items: Sequence[LibriTTSRItem | dict]) -> dict[str, torch.Tensor | list]:
        speech_tokens = [self._encode_audio(self._get_value(item, "wav_path")) for item in items]
        y_lens = [tokens.shape[-1] for tokens in speech_tokens]

        if self.deterministic_eval:
            mask_intervals, _ = sample_eval_mask_intervals(y_lens, mask_len=self.eval_mask_len)
        else:
            mask_intervals, _ = sample_mask_intervals(
                y_lens,
                mask_len_min=self.mask_len_min,
                mask_len_max=self.mask_len_max,
                max_n_spans=self.max_n_spans,
                mask_sample_dist=self.mask_sample_dist,
                min_gap=self.min_gap,
                generator=self.generator,
            )

        examples = [
            build_voicecraftx_inpainting_input(
                tokens,
                intervals,
                speech_mask_idx=self.speech_mask_idx,
            )
            for tokens, intervals in zip(speech_tokens, mask_intervals)
        ]
        batch = pad_voicecraftx_inpainting_batch(examples, pad_token=self.pad_token)
        batch.update(
            {
                "texts": [self._get_value(item, "text") for item in items],
                "wav_paths": [self._get_value(item, "wav_path") for item in items],
                "utterance_ids": [self._get_value(item, "utterance_id") for item in items],
                "speaker_ids": [self._get_value(item, "speaker_id") for item in items],
                "chapter_ids": [self._get_value(item, "chapter_id") for item in items],
                "mask_intervals": [example.mask_interval for example in examples],
                "y_lens": torch.tensor(y_lens, dtype=torch.long),
            }
        )
        return batch

    def _encode_audio(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(wav)
        with torch.no_grad():
            tokens = self.audio_tokenizer(wav.unsqueeze(0))[0].detach().cpu().long()
        if tokens.dim() != 2:
            raise RuntimeError(f"audio tokenizer must return [K, T], got {tuple(tokens.shape)}")
        return tokens

    @staticmethod
    def _get_value(item: LibriTTSRItem | dict, key: str):
        if isinstance(item, dict):
            return item[key]
        return getattr(item, key)
