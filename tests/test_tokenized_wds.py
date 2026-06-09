import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from dataset.tokenized_wds import TokenizedLibriTTSRWDSDataset, VoiceCraftXTokenizedWDSCollator


class TokenizedWDSTest(unittest.TestCase):
    def test_dataset_reads_tar_samples_and_collator_masks_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "samples-000000.tar"
            with tarfile.open(shard_path, mode="w") as tar:
                self._write_sample(tar, "utt_a", torch.arange(4 * 12).view(4, 12), "hello", "spk1")
                self._write_sample(tar, "utt_b", torch.arange(4 * 10).view(4, 10), "world", "spk2")

            dataset = TokenizedLibriTTSRWDSDataset(tmpdir)
            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            self.assertEqual(first["tokens"].shape, (4, 12))
            self.assertEqual(first["text"], "hello")
            self.assertEqual(first["utterance_id"], "utt_a")
            self.assertEqual(first["speaker_embedding"].shape, (3,))

            collator = VoiceCraftXTokenizedWDSCollator(
                speech_mask_idx=2049,
                pad_token=2048,
                mask_len_min=2,
                mask_len_max=3,
                max_n_spans=1,
                generator=torch.Generator().manual_seed(0),
            )
            batch = collator([dataset[0], dataset[1]])

            self.assertEqual(batch["input_ids"].shape[0], 2)
            self.assertEqual(batch["input_ids"].shape[1], 4)
            self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
            self.assertTrue(batch["loss_mask"].any())
            self.assertEqual(batch["texts"], ["hello", "world"])
            self.assertEqual(batch["speaker_embeddings"].shape, (2, 3))
            self.assertEqual(batch["y_lens"].tolist(), [12, 10])

    def _write_sample(
        self,
        tar: tarfile.TarFile,
        key: str,
        tokens: torch.Tensor,
        text: str,
        speaker_id: str,
    ) -> None:
        self._add_bytes(tar, f"{key}.tokens.pt", self._tensor_bytes(tokens))
        self._add_bytes(tar, f"{key}.spk.pt", self._tensor_bytes(torch.ones(3)))
        self._add_bytes(tar, f"{key}.txt", text.encode("utf-8"))
        metadata = {
            "utterance_id": key,
            "speaker_id": speaker_id,
            "chapter_id": "chapter",
            "wav_path": f"/tmp/{key}.wav",
            "num_frames": int(tokens.shape[-1]),
        }
        self._add_bytes(tar, f"{key}.json", json.dumps(metadata).encode("utf-8"))

    def _tensor_bytes(self, tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def _add_bytes(self, tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    unittest.main()
