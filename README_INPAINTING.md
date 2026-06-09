# VoiceCraft-X Speech Gap Infilling

This branch contains experimental training code for speech inpainting / gap infilling with VoiceCraft-X.

## Current Scope

- Base model: official VoiceCraft-X.
- Dataset used in experiments: LibriTTS-R `train-clean-100`.
- Original audio rate: 24 kHz.
- Training input rate: 16 kHz, resampled before VoiceCraft-X tokenization.
- Codec/tokenization: VoiceCraft-X EnCodec tokenizer.
- Task format: speech inpainting with one masked span.
- Input construction: `prefix + speech_mask + suffix + speech_mask + middle`.
- Masking: online random masking during training; deterministic masking is available for debugging/eval.
- Current objective: masked speech token prediction.
- Current eval: `eval_loss`; PESQ/STOI, ASR WER, and speaker similarity are planned final metrics.

## Files Added for Inpainting

- `src/dataset/online_masking.py`
- `src/dataset/libritts_r.py`
- `src/dataset/tokenized_wds.py`
- `scripts/create_tokenized_wds.py`
- `scripts/inspect_tokenized_wds.py`
- `scripts/train_inpainting.py`
- `tests/test_online_masking.py`
- `tests/test_tokenized_wds.py`
- `tests/test_voicecraftx_training.py`

`src/models/voicecraftx.py` has an added training `forward()` path. The existing inference `generate()` path was not intentionally changed.

## Data and Model Files

Large files are intentionally not tracked.

Prepare these locally:

```text
VoiceCraft-X/
  pretrained_models/
    voicecraftx.ckpt
    voicecraftx-en.ckpt
    voicecraftx-zh.ckpt
    encodec.th
    speech_campplus.onnx
  data/
    LibriTTS_R/
```

## Environment

Use Python 3.10. The local experiment environment used:

- `torch==2.1.0`
- CUDA 12.1 runtime
- `torchaudio==2.1.0`
- `torchvision==0.16.0`
- `transformers==4.51.3`
- `flash-attn==2.5.6`
- `setuptools==69.5.1`

See `requirements.txt` and `requirements-inpainting.txt`.

## Create Tokenized WebDataset

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/create_tokenized_wds.py \
  --root data/LibriTTS_R \
  --split train-clean-100 \
  --output-dir data/tokenized_wds/train-clean-100 \
  --prefix librittsr-train-clean-100 \
  --shard-size 1000 \
  --device cuda \
  --overwrite
```

Inspect:

```bash
PYTHONPATH=src python scripts/inspect_tokenized_wds.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --limit 3
```

## Smoke Training

Small deterministic overfit:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_inpainting.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --output-dir exp/inpainting_deterministic_16 \
  --limit 16 \
  --batch-size 1 \
  --max-steps 300 \
  --save-every 100 \
  --log-every 10 \
  --deterministic-mask \
  --eval-mask-len 20 \
  --mask-len-min 20 \
  --mask-len-max 20 \
  --lr 1e-5
```

Online masking with deterministic eval:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_inpainting.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --output-dir exp/inpainting_online_1000_eval \
  --limit 1000 \
  --eval-split-size 100 \
  --batch-size 1 \
  --max-steps 5000 \
  --save-every 1000 \
  --log-every 100 \
  --eval-every 500 \
  --eval-mask-len 20 \
  --mask-len-min 10 \
  --mask-len-max 30 \
  --lr 5e-6
```

## Notes

This is an experimental training baseline, not a final recipe. The 16-sample deterministic overfit test converged, which suggests the basic forward/loss path is usable. Larger random-mask experiments were still weak, so the next work should focus on evaluation metrics, text-gap alignment, generation-based validation, and training recipe improvements.
