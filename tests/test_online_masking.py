import unittest

import torch

from dataset.libritts_r import VoiceCraftXOnlineMaskingCollator
from dataset.online_masking import (
    build_voicecraftx_inpainting_input,
    pad_voicecraftx_inpainting_batch,
    reconstruct_original,
    sample_eval_mask_intervals,
    sample_mask_intervals,
)


class OnlineMaskingTest(unittest.TestCase):
    def test_sample_mask_intervals_respects_bounds(self):
        generator = torch.Generator().manual_seed(0)
        masks, non_masks = sample_mask_intervals(
            [100, 73],
            mask_len_min=10,
            mask_len_max=20,
            max_n_spans=1,
            generator=generator,
        )

        for y_len, intervals, complements in zip([100, 73], masks, non_masks):
            self.assertEqual(len(intervals), 1)
            start, end = intervals[0]
            self.assertLess(start, end)
            self.assertLessEqual(end, y_len)
            self.assertGreaterEqual(end - start, 10)
            self.assertLessEqual(end - start, 20)
            covered = sum(end - start for start, end in intervals + complements)
            self.assertEqual(covered, y_len)

    def test_eval_mask_intervals_are_centered(self):
        masks, non_masks = sample_eval_mask_intervals([101], mask_len=21)

        self.assertEqual(masks, [[(40, 61)]])
        self.assertEqual(non_masks, [[(0, 40), (61, 101)]])

    def test_build_voicecraftx_inpainting_input(self):
        audio = torch.arange(4 * 10, dtype=torch.long).view(4, 10)
        example = build_voicecraftx_inpainting_input(
            audio,
            [(3, 7)],
            speech_mask_idx=2049,
        )

        expected = torch.cat(
            [
                audio[:, :3],
                torch.full((4, 1), 2049),
                audio[:, 7:],
                torch.full((4, 1), 2049),
                audio[:, 3:7],
            ],
            dim=-1,
        )
        self.assertTrue(torch.equal(example.input_ids, expected))
        self.assertTrue(torch.equal(example.target_tokens, audio[:, 3:7]))
        self.assertTrue(torch.equal(reconstruct_original(example), audio))
        self.assertEqual(example.loss_positions.tolist(), [8, 9, 10, 11])

    def test_pad_voicecraftx_inpainting_batch_marks_middle_only(self):
        first = build_voicecraftx_inpainting_input(
            torch.arange(4 * 10, dtype=torch.long).view(4, 10),
            [(3, 7)],
            speech_mask_idx=2049,
        )
        second = build_voicecraftx_inpainting_input(
            torch.arange(4 * 8, dtype=torch.long).view(4, 8),
            [(2, 5)],
            speech_mask_idx=2049,
        )

        batch = pad_voicecraftx_inpainting_batch([first, second], pad_token=2048)

        self.assertEqual(batch["input_ids"].shape, (2, 4, 12))
        self.assertEqual(batch["labels"].shape, (2, 4, 12))
        self.assertTrue(batch["loss_mask"][0, first.loss_positions].all())
        self.assertTrue(batch["loss_mask"][1, second.loss_positions].all())
        self.assertEqual(int((batch["labels"][0] != -100).sum().item()), 4 * 4)
        self.assertEqual(int((batch["labels"][1] != -100).sum().item()), 4 * 3)


if __name__ == "__main__":
    unittest.main()


class FakeAudioTokenizer:
    def __call__(self, wav):
        length = max(1, wav.shape[-1] // 320)
        return torch.arange(4 * length, dtype=torch.long).view(1, 4, length)


class OnlineMaskingCollatorTest(unittest.TestCase):
    def test_collator_builds_masked_batch_from_items(self):
        items = [
            {
                "wav_path": "data/samples/mfa_alignments/84_121123_000008_000000.wav",
                "text": "hello world",
                "utterance_id": "utt1",
                "speaker_id": "spk1",
                "chapter_id": "chap1",
            },
            {
                "wav_path": "data/samples/mfa_alignments/4446_2275_000003_000000.wav",
                "text": "another sample",
                "utterance_id": "utt2",
                "speaker_id": "spk2",
                "chapter_id": "chap2",
            },
        ]
        collator = VoiceCraftXOnlineMaskingCollator(
            audio_tokenizer=FakeAudioTokenizer(),
            sample_rate=16000,
            speech_mask_idx=2049,
            pad_token=2048,
            mask_len_min=2,
            mask_len_max=4,
            max_n_spans=1,
            generator=torch.Generator().manual_seed(1),
        )

        batch = collator(items)

        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["input_ids"].shape[1], 4)
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        self.assertTrue(batch["loss_mask"].any())
        self.assertEqual(batch["texts"], ["hello world", "another sample"])
        self.assertEqual(len(batch["mask_intervals"]), 2)
