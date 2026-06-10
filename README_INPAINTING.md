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
- Current eval/debug metrics: `eval_loss`, `eval_masked_token_accuracy`, `eval_codebook_*_accuracy`, `loss_moving_avg`.
- Current generation diagnosis: fixed validation samples can be exported as `original.wav`, `masked_input.wav`, and `generated_matched.wav`.
- PESQ/STOI, ASR WER, and speaker similarity are planned final metrics.

## Files Added for Inpainting

- `src/dataset/online_masking.py`
- `src/dataset/libritts_r.py`
- `src/dataset/tokenized_wds.py`
- `scripts/create_tokenized_wds.py`
- `scripts/inspect_tokenized_wds.py`
- `scripts/train_inpainting.py`
- `scripts/generate_inpainting_samples.py`
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

Recommended current recipe for checkpoint generation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_inpainting.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --output-dir exp/inpainting_online_1000_mask10_30_lr1e5 \
  --limit 1000 \
  --eval-split-size 100 \
  --batch-size 1 \
  --max-steps 5000 \
  --save-every 5000 \
  --log-every 100 \
  --eval-every 500 \
  --eval-mask-len 20 \
  --mask-len-min 10 \
  --mask-len-max 30 \
  --lr 1e-5
```

For metric-only runs without checkpoint files, add:

```bash
--no-save
```

`--save-every 0` disables intermediate checkpoints but still writes `last.pt`. `--no-save` disables both intermediate checkpoints and `last.pt`.

## Generation Diagnosis

Export fixed validation samples for listening checks:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/generate_inpainting_samples.py \
  --wds-root data/tokenized_wds/train-clean-100 \
  --checkpoint exp/inpainting_online_1000_mask10_30_lr1e5/step-005000.pt \
  --output-dir exp/inpainting_generation_check \
  --limit 1000 \
  --eval-split-size 100 \
  --num-samples 4 \
  --eval-mask-len 20 \
  --max-new-frames 60
```

Each sample directory contains:

```text
original.wav              original validation audio
masked_input.wav          original audio with the target gap zeroed
generated.wav             raw generated span inserted into context
generated_matched.wav     generated span trimmed/padded to the target gap length
generated_gap*.wav        generated gap-only files when decodable
metadata.json             text, mask interval, generated frame counts
```

Use `generated_matched.wav` first for direct comparison with `original.wav` and `masked_input.wav`. Raw `generated.wav` may have a different duration because current VoiceCraft-X generation does not force the target gap length.

## Experiment Log

See `EXPERIMENTS_INPAINTING.md` for the accumulated experiment history, settings, results, and current interpretation.

## Notes

This is an experimental training baseline, not a final recipe. The 16-sample deterministic overfit test converged, which suggests the basic forward/loss path is usable. Larger random-mask experiments were still weak, so the next work should focus on evaluation metrics, text-gap alignment, generation-based validation, and training recipe improvements.
