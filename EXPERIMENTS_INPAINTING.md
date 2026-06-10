# Speech Gap Infilling Experiment Log

This document tracks VoiceCraft-X speech inpainting / gap infilling experiments. Append new experiments in the same format so results remain comparable.

## Current Setup

- Base model: VoiceCraft-X official checkpoint.
- Dataset source: LibriTTS-R `train-clean-100`.
- Tokenized WDS: `data/tokenized_wds/train-clean-100`.
- WDS size: 33,232 samples.
- Audio preprocessing: original 24 kHz LibriTTS-R audio is resampled to 16 kHz before VoiceCraft-X tokenization.
- Codec: VoiceCraft-X EnCodec tokenizer, 4 codebooks, 2048 speech tokens per codebook, 50 Hz frame rate.
- Current task format: `prefix + speech_mask + suffix + speech_mask + middle`.
- Current objective: masked speech token prediction.
- Main debug metrics: `loss`, `loss_moving_avg`, `eval_loss`, `eval_masked_token_accuracy`, `eval_codebook_*_accuracy`.
- Planned final metrics: PESQ/STOI, ASR WER, speaker similarity.

## Summary Table

| ID | Purpose | Data | Masking | Main Settings | Key Result | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | Training loop smoke with frozen LLM | 16 train | Online random | `max_steps=100`, LLM frozen | Loss did not clearly decrease | Frozen LLM path is too weak for useful adaptation. |
| E02 | Small online random overfit attempt | 16 train | Online random, `10-30` | `max_steps=100`, `lr=1e-5`, LLM unfrozen | Slight/noisy learning | Online random loss is hard to read at tiny scale. |
| E03 | Forward/loss sanity check | 16 train | Deterministic, len 20 | `max_steps=300`, `lr=1e-5` | Strong overfit | Training forward/loss alignment is not completely broken. |
| E04 | Small subset online learning | 100 train | Online random, `10-30` | `max_steps=2000`, `lr=5e-6` | Loss slowly decreased | Online random has a learning signal but is noisy. |
| E05 | Small subset deterministic overfit | 100 train | Deterministic, len 20 | `max_steps=2000`, `lr=5e-6` | Strong overfit | Deterministic masking is useful for debugging, not final infilling. |
| E06 | Medium subset online run | 1000 train | Online random, `10-30` | `max_steps=3000`, `lr=5e-6` | Weak/noisy convergence | Recipe is weak at 1000-sample scale. |
| E07 | Medium deterministic short run | 1000 train | Deterministic, len 20 | `max_steps=3000`, `lr=5e-6` | Did not overfit strongly | 3 epochs are insufficient at 1000 samples. |
| E08 | Medium deterministic long run | 1000 train | Deterministic, len 20 | `max_steps=10000`, `lr=5e-6` | Slow decrease, still weak | More steps alone do not solve the issue. |
| E09 | Medium deterministic higher LR | 1000 train | Deterministic, len 20 | `max_steps=3000`, `lr=1e-5` | Not clearly better than `5e-6` | LR increase alone is not enough. |
| E10 | Add valid split baseline | 900 train / 100 valid | Train online `10-30`, eval deterministic len 20 | `max_steps=5000`, `lr=5e-6` | `eval_loss 3.6287 -> 3.5176` | Valid loss decreases consistently but weakly. |
| E11 | Baseline with token accuracy metrics | 900 train / 100 valid | Train online `10-30`, eval deterministic len 20 | `max_steps=5000`, `lr=5e-6` | `eval_acc 0.2006 -> 0.2137` | Token accuracy improves but remains low. |
| E12 | Easier mask comparison | 900 train / 100 valid | Train online `5-15`, eval deterministic len 10 | `max_steps=5000`, `lr=5e-6` | `eval_acc 0.1842 -> 0.2052` | Shorter masks improve more, but final accuracy is not better. |
| E13 | Gradient accumulation | 900 train / 100 valid | Train online `10-30`, eval deterministic len 20 | `max_steps=5000`, `lr=5e-6`, `grad_acc=4` | `eval_acc 0.1985 -> 0.2096` | Calmer gradients, but worse than baseline under lower update budget. |
| E14 | Higher LR | 900 train / 100 valid | Train online `10-30`, eval deterministic len 20 | `max_steps=5000`, `lr=1e-5` | `eval_loss 3.5899 -> 3.5057`, `eval_acc 0.2040 -> 0.2115` | Better loss, accuracy not clearly better. |

## Detailed Experiments

### E01. 16 Samples / Online Random / LLM Frozen

**Purpose**

Check whether the basic training loop runs when only non-LLM parameters are updated.

**Setting**

```text
limit: 16
masking: online random
max_steps: 100
freeze_llm: true
```

**Result**

Loss did not show a clear downward trend.

**Interpretation**

Freezing the LLM made adaptation too weak for this task. This setting is not a promising training recipe.

---

### E02. 16 Samples / Online Random / LLM Unfrozen

**Purpose**

Check whether unfreezing the model gives a stronger signal on a tiny subset.

**Setting**

```text
limit: 16
masking: online random
mask_len_min: 10
mask_len_max: 30
max_steps: 100
lr: 1e-5
freeze_llm: false
```

**Result**

Loss moved but remained noisy. No strong overfit pattern appeared.

**Interpretation**

Online random masking changes the prediction problem every step, so tiny-subset step loss is not a reliable sanity check.

---

### E03. 16 Samples / Deterministic Mask

**Purpose**

Verify whether the model can overfit a fixed inpainting problem. This checks forward/loss alignment.

**Setting**

```text
limit: 16
masking: deterministic
eval_mask_len: 20
max_steps: 300
lr: 1e-5
freeze_llm: false
```

**Result**

Loss decreased strongly and the subset overfit.

**Interpretation**

The training forward path and masked-token loss are usable enough to learn a fixed small problem. Deterministic masking should be treated as a debugging tool, not a final training strategy.

---

### E04. 100 Samples / Online Random

**Purpose**

Check whether online random masking learns on a small but less trivial subset.

**Setting**

```text
limit: 100
masking: online random
mask_len_min: 10
mask_len_max: 30
max_steps: 2000
lr: 5e-6
```

**Result**

Average loss slowly decreased, but logs were noisy.

**Interpretation**

There is a learning signal under online random masking, but the recipe is unstable/noisy.

---

### E05. 100 Samples / Deterministic Mask

**Purpose**

Check whether deterministic overfit still works at 100 samples.

**Setting**

```text
limit: 100
masking: deterministic
eval_mask_len: 20
max_steps: 2000
lr: 5e-6
```

**Result**

The model overfit the fixed task strongly.

**Interpretation**

This further supports that the forward/loss code path works. The result does not prove real gap infilling ability because the mask is fixed.

---

### E06. 1000 Samples / Online Random

**Purpose**

Scale online random masking to a medium subset and check convergence.

**Setting**

```text
limit: 1000
masking: online random
mask_len_min: 10
mask_len_max: 30
max_steps: 3000
lr: 5e-6
eval: none
```

**Result**

Loss did not decrease stably. Some gradient norm spikes appeared.

**Interpretation**

The current recipe is weak at 1000-sample scale. A validation split and better metrics were needed.

---

### E07. 1000 Samples / Deterministic Short Run

**Purpose**

Check whether 1000 examples can be fit with a fixed deterministic mask.

**Setting**

```text
limit: 1000
masking: deterministic
eval_mask_len: 20
max_steps: 3000
lr: 5e-6
```

**Result**

The model did not overfit strongly within 3000 steps.

**Interpretation**

Three epochs are not enough at 1000 samples, even for the easier deterministic task.

---

### E08. 1000 Samples / Deterministic Long Run

**Purpose**

Check whether simply increasing training steps solves the weak convergence.

**Setting**

```text
limit: 1000
masking: deterministic
eval_mask_len: 20
max_steps: 10000
lr: 5e-6
```

**Result**

Loss slowly decreased over time, but remained weak after 10 epochs.

**Interpretation**

More steps help somewhat, but step count alone is not enough. The objective/conditioning/recipe likely needs improvement.

---

### E09. 1000 Samples / Deterministic / Higher LR

**Purpose**

Check whether increasing LR improves convergence.

**Setting**

```text
limit: 1000
masking: deterministic
eval_mask_len: 20
max_steps: 3000
lr: 1e-5
```

**Result**

No clear improvement over `lr=5e-6`.

**Interpretation**

Learning rate alone does not explain the weak convergence.

---

### E10. 900 Train / 100 Valid / Online Train + Deterministic Eval

**Purpose**

Introduce a stable validation protocol: random masks for training, fixed deterministic masks for evaluation.

**Setting**

```text
limit: 1000
eval_split_size: 100
train_size: 900
valid_size: 100
train_masking: online random
mask_len_min: 10
mask_len_max: 30
eval_masking: deterministic
eval_mask_len: 20
max_steps: 5000
lr: 5e-6
```

**Result**

```text
eval_loss: 3.6287 -> 3.5176
```

**Interpretation**

Validation loss decreased monotonically, so the model is learning something. The improvement is small, so the recipe is still weak.

---

### E11. Baseline with Token Accuracy Metrics

**Purpose**

Repeat the 900/100 baseline after adding token-level metrics.

**Setting**

```text
limit: 1000
eval_split_size: 100
train_size: 900
valid_size: 100
train_masking: online random
mask_len_min: 10
mask_len_max: 30
eval_masking: deterministic
eval_mask_len: 20
max_steps: 5000
lr: 5e-6
metrics: eval_loss, eval_masked_token_accuracy, eval_codebook_*_accuracy, loss_moving_avg
```

**Result**

```text
eval_loss: 3.6287 -> 3.5176
eval_masked_token_accuracy: 0.2006 -> 0.2137
final codebook accuracy:
  cb0: 0.2200
  cb1: 0.2105
  cb2: 0.2185
  cb3: 0.2060
```

**Interpretation**

Accuracy improves, but remains low at around 21%. Codebook 3 is weaker than the others. The model is learning, but not enough for strong infilling quality.

---

### E12. Easier Mask Comparison

**Purpose**

Test whether shorter gaps make the task easier and improve convergence.

**Setting**

```text
limit: 1000
eval_split_size: 100
train_size: 900
valid_size: 100
train_masking: online random
mask_len_min: 5
mask_len_max: 15
eval_masking: deterministic
eval_mask_len: 10
max_steps: 5000
lr: 5e-6
metrics: eval_loss, eval_masked_token_accuracy, eval_codebook_*_accuracy, loss_moving_avg
```

**Result**

```text
eval_loss: 3.7385 -> 3.5428
eval_masked_token_accuracy: 0.1842 -> 0.2052
final codebook accuracy:
  cb0: 0.2050
  cb1: 0.2130
  cb2: 0.2120
  cb3: 0.1910
```

**Interpretation**

The shorter-mask setting improved more from its starting point, but final accuracy was lower than the `10-30` baseline. Shorter masks reduce gap difficulty, but also reduce target tokens per step, which may weaken supervision.


---

### E13. Gradient Accumulation 4

**Purpose**

Check whether a larger effective batch stabilizes online random masking training.

**Setting**

```text
limit: 1000
eval_split_size: 100
train_size: 900
valid_size: 100
train_masking: online random
mask_len_min: 10
mask_len_max: 30
eval_masking: deterministic
eval_mask_len: 20
max_steps: 5000
lr: 5e-6
batch_size: 1
gradient_accumulation_steps: 4
checkpoint_save: disabled with --no-save
```

**Result**

```text
eval_loss: 3.6734 -> 3.5481
eval_masked_token_accuracy: 0.1985 -> 0.2096
final codebook accuracy:
  cb0: 0.2135
  cb1: 0.2120
  cb2: 0.2080
  cb3: 0.2050
train_loss: 3.8078
```

**Interpretation**

Gradient accumulation reduced gradient norm spikes and made updates numerically calmer, but it did not improve validation accuracy or loss compared with the baseline. Because `max_steps=5000` counts micro-steps, this run used only about 1250 optimizer updates, so it is not an equal-update-budget comparison.

---

### E14. Higher Learning Rate 1e-5

**Purpose**

Check whether the baseline `lr=5e-6` under-updates the model.

**Setting**

```text
limit: 1000
eval_split_size: 100
train_size: 900
valid_size: 100
train_masking: online random
mask_len_min: 10
mask_len_max: 30
eval_masking: deterministic
eval_mask_len: 20
max_steps: 5000
lr: 1e-5
batch_size: 1
checkpoint_save: disabled with --no-save
```

**Result**

```text
eval_loss: 3.5899 -> 3.5057
best eval_loss: 3.5039 at step 4500
eval_masked_token_accuracy: 0.2040 -> 0.2115
best eval_masked_token_accuracy: 0.2141 at step 3000
final codebook accuracy:
  cb0: 0.2165
  cb1: 0.2125
  cb2: 0.2140
  cb3: 0.2030
train_loss: 3.6868
```

**Interpretation**

`lr=1e-5` improved train loss and eval loss compared with the baseline, but final token accuracy did not clearly improve. The run peaked around 21.4% accuracy at step 3000 and then drifted slightly. Higher LR is somewhat better for loss, but it does not solve the core infilling accuracy bottleneck.

## Current Conclusions

1. The training code path works: deterministic small-subset overfit succeeds.
2. Online random masking produces a real but weak learning signal.
3. The current 1000-sample recipe reaches only about 20-21% masked token accuracy.
4. Reducing mask length alone does not solve the problem.
5. Gradient accumulation calms gradients but needs equal-update-budget testing before calling it worse.
6. `lr=1e-5` improves loss more than `lr=5e-6`, but does not clearly improve token accuracy.
7. Codebook 3 tends to be weaker than other codebooks.
8. Next experiments should focus on update budget, scheduler, generation-based eval, and objective/text alignment.

## Recommended Next Experiments

### N01. Effective Batch Size

```text
limit: 1000
eval_split_size: 100
train_masking: online random
mask_len_min: 10
mask_len_max: 30
eval_mask_len: 20
max_steps: 5000
lr: 5e-6
gradient_accumulation_steps: 4
```

Goal: check whether less noisy updates improve eval accuracy.

### N02. Higher LR Baseline

```text
limit: 1000
eval_split_size: 100
train_masking: online random
mask_len_min: 10
mask_len_max: 30
eval_mask_len: 20
max_steps: 5000
lr: 1e-5
```

Goal: check whether the current online recipe is under-updating.

### N03. Generation-Based Eval Smoke

Generate and decode a small number of validation examples at fixed masks.

Goal: determine whether token metrics correlate with audible gap quality.
