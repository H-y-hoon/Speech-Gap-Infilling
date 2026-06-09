from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
import tarfile
from typing import Sequence

import torch

from .online_masking import (
    build_voicecraftx_inpainting_input,
    pad_voicecraftx_inpainting_batch,
    sample_eval_mask_intervals,
    sample_mask_intervals,
)


@dataclass(frozen=True)
class TokenizedWDSRecord:
    shard_path: str
    key: str
    tokens_member: str
    text_member: str
    metadata_member: str
    speaker_member: str | None = None


class TokenizedLibriTTSRWDSDataset(torch.utils.data.Dataset):
    """Map-style reader for VoiceCraft-X tokenized LibriTTS-R tar shards."""

    def __init__(
        self,
        root: str | Path,
        shard_glob: str = "*.tar",
        load_speaker_embedding: bool = True,
    ) -> None:
        self.root = Path(root)
        self.load_speaker_embedding = load_speaker_embedding
        if not self.root.exists():
            raise FileNotFoundError(f"tokenized WDS directory not found: {self.root}")

        self.shards = sorted(self.root.glob(shard_glob))
        if not self.shards:
            raise RuntimeError(f"no shards matching {shard_glob!r} under {self.root}")
        self.records = self._index_shards(self.shards)
        if not self.records:
            raise RuntimeError(f"no complete tokenized samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with tarfile.open(record.shard_path, mode="r") as tar:
            tokens = _load_tensor(tar, record.tokens_member).long()
            text = _load_bytes(tar, record.text_member).decode("utf-8")
            metadata = json.loads(_load_bytes(tar, record.metadata_member).decode("utf-8"))
            speaker_embedding = None
            if self.load_speaker_embedding and record.speaker_member is not None:
                speaker_embedding = _load_tensor(tar, record.speaker_member).float()

        if tokens.dim() != 2:
            raise RuntimeError(f"expected tokens [K, T], got {tuple(tokens.shape)} for {record.key}")
        metadata.setdefault("utterance_id", record.key)
        return {
            "tokens": tokens,
            "text": text,
            "metadata": metadata,
            "speaker_embedding": speaker_embedding,
            "utterance_id": metadata.get("utterance_id", record.key),
            "speaker_id": metadata.get("speaker_id"),
            "chapter_id": metadata.get("chapter_id"),
            "wav_path": metadata.get("wav_path"),
            "num_frames": int(tokens.shape[-1]),
        }

    def _index_shards(self, shards: Sequence[Path]) -> list[TokenizedWDSRecord]:
        records: list[TokenizedWDSRecord] = []
        for shard in shards:
            with tarfile.open(shard, mode="r") as tar:
                grouped: dict[str, dict[str, str]] = {}
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    key, suffix = _split_member_name(member.name)
                    grouped.setdefault(key, {})[suffix] = member.name

            for key in sorted(grouped):
                parts = grouped[key]
                required = {"tokens.pt", "txt", "json"}
                if not required.issubset(parts):
                    continue
                records.append(
                    TokenizedWDSRecord(
                        shard_path=str(shard),
                        key=key,
                        tokens_member=parts["tokens.pt"],
                        text_member=parts["txt"],
                        metadata_member=parts["json"],
                        speaker_member=parts.get("spk.pt"),
                    )
                )
        return records


class VoiceCraftXTokenizedWDSCollator:
    """Build VoiceCraft-X online inpainting batches from pre-tokenized samples."""

    def __init__(
        self,
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

    def __call__(self, items: Sequence[dict]) -> dict[str, torch.Tensor | list]:
        speech_tokens = [item["tokens"].long() for item in items]
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
                "texts": [item["text"] for item in items],
                "metadata": [item["metadata"] for item in items],
                "utterance_ids": [item["utterance_id"] for item in items],
                "speaker_ids": [item["speaker_id"] for item in items],
                "chapter_ids": [item["chapter_id"] for item in items],
                "wav_paths": [item["wav_path"] for item in items],
                "speaker_embeddings": _stack_optional_tensors(
                    [item.get("speaker_embedding") for item in items]
                ),
                "mask_intervals": [example.mask_interval for example in examples],
                "y_lens": torch.tensor(y_lens, dtype=torch.long),
            }
        )
        return batch


def _split_member_name(name: str) -> tuple[str, str]:
    filename = Path(name).name
    for suffix in ("tokens.pt", "spk.pt", "txt", "json"):
        marker = f".{suffix}"
        if filename.endswith(marker):
            return filename[: -len(marker)], suffix
    return filename, ""


def _load_bytes(tar: tarfile.TarFile, member_name: str) -> bytes:
    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(member_name)
    return extracted.read()


def _load_tensor(tar: tarfile.TarFile, member_name: str) -> torch.Tensor:
    return torch.load(io.BytesIO(_load_bytes(tar, member_name)), map_location="cpu")


def _stack_optional_tensors(tensors: Sequence[torch.Tensor | None]) -> torch.Tensor | list:
    if not tensors or any(tensor is None for tensor in tensors):
        return list(tensors)
    shapes = {tuple(tensor.shape) for tensor in tensors if tensor is not None}
    if len(shapes) != 1:
        return list(tensors)
    return torch.stack([tensor for tensor in tensors if tensor is not None], dim=0)
