import types
import unittest

import torch

from models.voicecraftx import VoiceCraftX


class VoiceCraftXTrainingTest(unittest.TestCase):
    def test_shift_batch_matches_codebook_delay_pattern(self):
        model = object.__new__(VoiceCraftX)
        model.num_codebooks = 4
        model.config = types.SimpleNamespace(speech_empty_idx=2048)

        seqs = torch.arange(2 * 4 * 5, dtype=torch.long).view(2, 4, 5)
        shifted = model.shift_batch(seqs)

        self.assertEqual(shifted.shape, (2, 4, 9))
        self.assertTrue(torch.equal(shifted[:, 0, 1:6], seqs[:, 0]))
        self.assertTrue(torch.equal(shifted[:, 1, 2:7], seqs[:, 1]))
        self.assertTrue(torch.equal(shifted[:, 2, 3:8], seqs[:, 2]))
        self.assertTrue(torch.equal(shifted[:, 3, 4:9], seqs[:, 3]))
        self.assertTrue(torch.equal(shifted[:, 0, 0], torch.full((2,), 2048)))

    def test_shift_batch_can_shift_ignore_labels(self):
        model = object.__new__(VoiceCraftX)
        model.num_codebooks = 4
        model.config = types.SimpleNamespace(speech_empty_idx=2048)

        labels = torch.full((1, 4, 6), -100, dtype=torch.long)
        labels[:, :, 4:] = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])

        shifted_labels = model.shift_batch(labels, fill_value=-100)[:, :, 1:]

        self.assertEqual(int(shifted_labels.ne(-100).sum().item()), 8)
        self.assertTrue(torch.equal(shifted_labels[0, 0, 4:6], torch.tensor([1, 2])))
        self.assertTrue(torch.equal(shifted_labels[0, 3, 7:9], torch.tensor([7, 8])))


if __name__ == "__main__":
    unittest.main()
