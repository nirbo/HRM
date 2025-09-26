# HRM Reasoning Curriculum Blueprint

This guide lays out the staged plan for teaching the hybrid HRM×Mamba2 model to reason, starting from the current language-only warmup run and progressing toward mixed reasoning workloads.

## Stage A — Language Stabilisation (current run)
- Dataset: `datasets/wiki_chunks/processed` (≈1.22B tokens).
- Flags: `--hrm_gate_warmup_steps 8000`, `--patience_grace_steps 10000`, `--eval_loss_patience 3`, `--lr_min_ratio 0.05`, `--optimizer adamw_8bit`, `--batch_size 22`, `--eval_batch_size 22`.
- Objective: let the decoder relearn English with the HRM gate closed until validation loss plateaus (target ≥30k steps, ideally full 308k).
- Checkpoints: monitor eval loss at steps 9k, 10.5k, 15k, etc.; best models saved automatically via `--save_best_model`.

```bash
python -m hrm_lm.training.train \
  --dataset datasets/wiki_chunks/processed/ \
  --batch_size 22 \
  --steps 308212 \
  --learning_rate 0.00025 \
  --warmup_steps 4000 \
  --lr_scheduler cosine \
  --lr_min_ratio 0.05 \
  --grad_clip 5.0 \
  --val_every 1500 \
  --run_name hrm-gated-pretrain \
  --checkpoint_limit 5 \
  --mixed_precision bf16 \
  --eval_batch_size 22 \
  --log_steps 1 \
  --dataset_workers 30 \
  --save_best_model \
  --optimizer adamw_8bit \
  --max_seq_len 512 \
  --max_val_samples 20000 \
  --eval_loss_patience 3 \
  --patience_grace_steps 10000 \
  --hrm_gate_warmup_steps 8000
```

## Stage B — Transition: Controlled HRM Exposure
1. **Enable auxiliary training hooks**
   - Set in config (or via CLI overrides):
     - `model.hrm.deep_supervision: true`
     - `model.hrm.ds_weight: 0.1`
     - `model.hrm.use_halting: true`
     - `model.hrm.halting_weight: 0.005` (tune 0.005–0.01)
   - Reduce/disable `--hrm_gate_warmup_steps` (e.g., 1000 or gradual ramp) so HRM gradients begin flowing earlier each run.

2. **Prepare reasoning primers**
   - Install once inside the project virtualenv (includes Pillow for ScienceQA images):

```bash
./venv/bin/pip install datasets pillow
```

   - Export raw JSONL files (stored under `datasets/reasoning/<name>/raw/`):

```bash
# GSM8K math chains (cap to 8k examples for a compact primer)
./venv/bin/python scripts/build_reasoning_dataset.py \
  --dataset gsm8k \
  --split train \
  --limit 8000 \
  --output datasets/reasoning/gsm8k/raw/gsm8k_train.jsonl

# MBPP programming-by-example tasks
./venv/bin/python scripts/build_reasoning_dataset.py \
  --dataset mbpp \
  --split train \
  --output datasets/reasoning/mbpp/raw/mbpp_train.jsonl

# ScienceQA rationales (trim to 5k for quick iterations)
./venv/bin/python scripts/build_reasoning_dataset.py \
  --dataset science_qa \
  --split train \
  --limit 5000 \
  --output datasets/reasoning/scienceqa/raw/scienceqa_train.jsonl
```

   - Tokenize/pack each JSONL with the standard language-prep script. Example for GSM8K:

```bash
./venv/bin/python scripts/prepare_language_dataset.py \
  --source datasets/reasoning/gsm8k/raw \
  --dest datasets/reasoning/gsm8k/processed \
  --tokenizer tokenizer.json \
  --max-seq-len 512 \
  --val-ratio 0.01 \
  --tokenizer-num-threads 30 \
  --tokenizer-batch-size 8192
```

   - Repeat the tokenization command for MBPP and ScienceQA (changing the `--source` and `--dest` paths accordingly).

3. **Merge with Wikipedia**
   - Start with a 90/5/5 mix (language/math/trace) by concatenating batches offline:

```bash
python scripts/merge_prepared_batches.py \
  --sources \
    datasets/wiki_chunks/processed \
    datasets/reasoning/gsm8k/processed \
    datasets/reasoning/mbpp/processed \
  --weights 0.90 0.05 0.05 \
  --output datasets/hybrid_mix_stageB \
  --train-samples 5000000 \
  --val-samples 20000
```

   - (Optional) Append `datasets/reasoning/scienceqa/processed` to `--sources` with an adjusted weight (e.g., `0.90 0.05 0.03 0.02`) to include rationale-heavy examples.

   - The script streams each source split and writes a combined `train.jsonl`, `val.jsonl`, and `meta.json`; adjust weights as needed.

4. **Resume training with auxiliaries enabled**

```bash
python -m hrm_lm.training.train \
  --dataset datasets/hybrid_mix_stageB \
  --batch_size 22 \
  --steps 180000 \
  --learning_rate 0.0002 \
  --warmup_steps 2000 \
  --lr_scheduler cosine \
  --lr_min_ratio 0.05 \
  --grad_clip 5.0 \
  --val_every 1500 \
  --run_name hrm-stageB \
  --checkpoint_limit 5 \
  --mixed_precision bf16 \
  --eval_batch_size 22 \
  --log_steps 1 \
  --dataset_workers 30 \
  --save_best_model \
  --optimizer adamw_8bit \
  --max_seq_len 512 \
  --max_val_samples 20000 \
  --eval_loss_patience 3 \
  --patience_grace_steps 5000 \
  --hrm_gate_warmup_steps 1000
```

5. **Monitoring**
   - Track overall validation loss plus small held-out reasoning dev sets (math, code) to ensure HRM signals improve targeted metrics.
   - Inspect halting probabilities to verify HRM cycles are being used consistently.

## Stage C — Full Reasoning Curriculum
1. **Dataset mix** (adjust per results):
   - 60–70% general language (Wikipedia or similar).
   - 15–20% multi-step reasoning datasets (math word problems, long-form scientific explanations, coding tasks with test cases).
   - 10–15% targeted HRM builders (longer chains, algorithmic tasks, synthetic proofs).

2. **Curriculum tactics**
   - Consider staged gate scaling (e.g., ramp gate_scale from 0→1 over several hundred steps at run start) instead of hard warmup.
   - Use deeper HRM cycles or increased `h_cycles/l_steps` once stability is proven.
   - Periodically refresh reasoning corpora with harder examples or longer chains to prevent overfitting.

3. **Evaluation suite**
   - Language perplexity on Wikipedia holdout.
   - Math accuracy (GSM8K-style dev set, synthetic arithmetic holdout).
   - Code-trace correctness (execute generated traces/tests).
   - Scientific QA rationale metrics (BLEU/ROUGE or custom scoring).

4. **Sample command skeleton**

```bash
python -m hrm_lm.training.train \
  --dataset datasets/hybrid_mix_stageC \
  --batch_size 20 \
  --steps 400000 \
  --learning_rate 0.00018 \
  --warmup_steps 3000 \
  --lr_scheduler cosine \
  --lr_min_ratio 0.05 \
  --grad_clip 5.0 \
  --val_every 2000 \
  --run_name hrm-stageC \
  --checkpoint_limit 8 \
  --mixed_precision bf16 \
  --eval_batch_size 20 \
  --log_steps 1 \
  --dataset_workers 40 \
  --save_best_model \
  --optimizer adamw_8bit \
  --max_seq_len 512 \
  --max_val_samples 25000 \
  --eval_loss_patience 4 \
  --patience_grace_steps 8000 \
  --deep_supervision 1 \
  --ds_weight 0.1 \
  --use_halting 1 \
  --halting_weight 0.01
```

5. **Iteration loop**
   - After each major phase, run qualitative evals (generation, reasoning probes) and adjust mixes/weights based on where the HRM helps or hurts.
   - Keep copies of best checkpoints and reasoning dev logs so regressions are easy to catch.

## Notes
- All dataset mixing commands assume preprocessed JSONL triples matching the trainer format.
- Adjust weights/steps/learning rates as empirical results dictate; values above are starting points.
- Remember to validate merged datasets with `json.loads` sweeps before launching large runs.
- **HRM signal diagnostics:**
  - `gate μ` and its `[min,max]` window report how much decoder memory is coming from the HRM bridge; rising means the gate head is routing harder batches toward HRM while low mins show easy samples skipping it.
  - `halt Σ` is the batch-mean sum of per-cycle halting probabilities; keep it near the configured target (1.0 by default) and watch the spread to spot wasted cycles or premature exits.
  - The halting penalty is scaled by both `halting_weight` and the average gate strength, so raising the weight only has bite once the gate is open. For mixed 75/25 language↔reasoning corpora start around 0.05–0.10, monitor the logs, then step toward 0.15+ only if `halt Σ` consistently overshoots.
