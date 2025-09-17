# HRM × LM (Mamba2/Transformer) — Integration Design Document

**Author:** Assistant  
**Audience:** Implementer-in-IDE  
**Goal:** Deliver a precise, end‑to‑end plan (with exact file paths, code diffs, and tests) to move from the current skeleton to a working prototype capable of training on real text + reasoning tasks, while preserving architectural rigor and parameter efficiency.

---

## 0) Executive Summary

**Objective.** Build a single, end‑to‑end differentiable model that couples a compact language module (LM: Mamba2 or Transformer) with an **HRM** (Hierarchical Reasoning) core. The LM handles natural language I/O; the HRM performs deep, multi‑step latent reasoning in one forward pass.

**Core design.**
- **Encoder (LM):** Mamba2 (linear-time SSM) if available; otherwise a compact Transformer encoder.
- **HRM core:** Two interdependent recurrent modules:
  - **L** (low-level) iterates `l_steps` times per high-level cycle, guided top‑down by H.
  - **H** (high-level) updates once per cycle, aggregating bottom‑up signals from L.
  - **One‑step gradient mode:** only the final HRM cycle participates in autograd; prior cycles run under `no_grad` (surrogate “DEQ-style” training to bound memory).
- **Bridge:** encoder‑to‑HRM (project pooled encoder state), and HRM‑to‑decoder via:
  - **Prefix mode (default):** project HRM latent `z` to `prefix_len` pseudo‑tokens as decoder memory.
  - **Cross‑attention mode:** single contextual vector attended by the decoder.
- **Decoder (LM):** compact Transformer decoder.

**Where we’re at now (your machine).**
- You successfully ran:  
  `python -m hrm_lm.training.train --dry_run 1`  
  and obtained: `dry_run loss: 10.50...`
- Packaging is in place (`pyproject.toml`), `uv pip install -e .` works.
- Encoder mask kwarg mismatch is fixed.
- A minor print warning remains (we’ll patch to `.item()`).

**Next steps.**  
1) Fix small polish items (loss print, decoder positional encodings).  
2) Add **deep supervision & optional halting** in HRM.  
3) Implement **cross‑attn bridge** path + gating.  
4) Plug in **tokenization + dataset** (synthetic first, then real).  
5) Provide a **generation script** and **evaluation metrics**.  
6) Optional: enable **Mamba2** on your Linux+CUDA box.

---

## 1) Current Repository State (Summary)

```
hrm_lm_skeleton/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ scripts/
│  ├─ setup.sh
│  └─ run_demo.sh
├─ src/hrm_lm/
│  ├─ __init__.py
│  ├─ configs/default.yaml
│  ├─ models/
│  │  ├─ transformer_layers.py   # PositionalEncoding, TransformerEncoder/Decoder
│  │  ├─ mamba2_layers.py        # Optional Mamba2 stack, falls back to Transformer
│  │  ├─ encoder.py              # LMEncoder: embeddings + pos enc + backend stack
│  │  ├─ hrm_core.py             # HRMCore: H/L loops, one-step-grad, readout
│  │  ├─ bridges.py              # PromptToHRM, HRMToPrefix, HRMToCrossAttn
│  │  └─ hybrid.py               # HRMLanguageModel: glueing encoder→HRM→decoder
│  └─ training/train.py          # toy loop + --dry_run
└─ tests/test_forward.py
```

**Key shapes (B=batch, S=enc len, T=dec len, D=d_model, P=prefix_len):**
- `input_ids`: (B, S), `decoder_input_ids`: (B, T)
- Encoder hidden: (B, S, D), pooled `[CLS]`: (B, D)
- HRM `z`: (B, D)
- Prefix bridge memory: (B, P, D); Cross‑attn memory: (B, 1, D)
- Decoder logits: (B, T, vocab_size)

**Known minor issues (we will fix):**
- Decoder currently uses a **zero** learned positional matrix rather than a proper positional encoding → hurts sequence modeling; we’ll switch to sinusoidal (matching encoder) or a learned embedding properly initialized.
- Loss print uses `float(loss)` (warns in PyTorch 2.2+). We’ll use `loss.item()`.

---

## 2) Mathematical Dataflow (Concise but Precise)

Let encoder produce hidden states `H_enc ∈ ℝ^{B×S×D}`, pool by `[CLS]` → `x ∈ ℝ^{B×D}`.  
Project encoder latents for HRM:
- `h_in = W_h x + b_h  ∈ ℝ^{B×D}`
- `l_in = W_l x + b_l  ∈ ℝ^{B×D}`

Initialize HRM internal sequences:
- `H_0 ∈ ℝ^{B×H_len×D}`, `L_0 ∈ ℝ^{B×L_len×D}` (learned parameters tiled across B)

For cycles `c = 1..N`:
- **Top‑down to L:** `td_c = W_td · mean(H_{c-1}, dim=seq) ∈ ℝ^{B×D}`  
  L receives additive input per step: `extra_L = td_c + l_in`.
- **Low‑level recurrence:** for steps `t = 1..T`:  
  `L_t = f_L(L_{t-1}; extra_L)` where `f_L` is a small encoder‑only transformer block with residuals + LayerNorm.
- **Bottom‑up to H:** `bu_c = W_bu · mean(L_T, dim=seq) ∈ ℝ^{B×D}`  
  H receives additive input: `extra_H = bu_c + h_in`.
- **High‑level update:**  
  `H_c = f_H(H_{c-1}; extra_H)` (same block type as f_L but on H sequence)

**One‑step grad (default):** For cycles `c < N`, the updates run under `torch.no_grad()`; only the **final cycle** builds autograd tape.

Readout: `z_c = mean(H_c, dim=seq) ∈ ℝ^{B×D}`, final `z = z_N`. Optionally keep all `{z_c}` for deep supervision.

**Bridging:**
- **Prefix:** `M = Proj(z) → ℝ^{B×P×D}` (reshape).
- **Cross‑attn:** `M = Proj(z) → ℝ^{B×1×D}`.

**Decoder:** Autoregressive Transformer decoder consumes `(decoder_input_ids, M)` to produce logits.

---

## 3) Implementation Plan — Tasks & Subtasks

> Style rule: keep code diffs minimal, 2-space indents, minimal inline comments. Add more narrative in this doc, not in code.

### Task 0 — Housekeeping (DONE/VERIFY)
- Ensure editable install works (`uv pip install -e .`), and module imports succeed.
- Ensure NumPy installed (you did).
- **Patch** the loss print to `.item()` (fix warning).

**File:** `src/hrm_lm/training/train.py`  
**Change:**
```diff
- print('dry_run loss:', float(out['loss']))
+ print('dry_run loss:', out['loss'].item())
```

### Task 1 — Decoder positional encoding (CRITICAL)
**Why:** Current decoder uses a zero `nn.Parameter` for positions; that removes positional cues. Replace with the sinusoidal `PositionalEncoding` already used in the encoder.

**File:** `src/hrm_lm/models/decoder.py`  
**Patch:**
```diff
-from .transformer_layers import TransformerDecoder
+from .transformer_layers import TransformerDecoder, PositionalEncoding

 class LMDecoder(nn.Module):
   def __init__(self, vocab_size, d_model, n_layers, max_seq_len):
     super().__init__()
     self.tok = nn.Embedding(vocab_size, d_model)
-    self.pos = nn.Parameter(torch.zeros(max_seq_len, d_model))
+    self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
     self.dec = TransformerDecoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)
     self.norm = nn.LayerNorm(d_model)
     self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

   def forward(self, input_ids, memory_tokens, attention_mask=None, memory_mask=None):
-    x = self.tok(input_ids) + self.pos[:input_ids.size(1)].unsqueeze(0)
+    x = self.tok(input_ids)
+    x = self.pos(x)
     mem = memory_tokens
     h = self.dec(x, mem,
       tgt_key_padding_mask=(attention_mask==0) if attention_mask is not None else None,
       memory_key_padding_mask=memory_mask)
     h = self.norm(h)
     logits = self.lm_head(h)
     return logits
```

**Test:** `python -m hrm_lm.training.train --dry_run 1` (expect same style output, stable loss).

### Task 2 — Deep Supervision & Optional Halting

**Goal:** Encourage HRM to produce progressively useful `z_c` latents and optionally learn an adaptive cycle halting signal.

**Config additions:** `src/hrm_lm/configs/default.yaml`
```yaml
model:
  hrm:
    deep_supervision: false
    ds_weight: 0.2
    use_halting: false
    halting_weight: 0.01
```

**A) Deep supervision wiring**
- In `HRMCore.forward(x, return_all=True)`, we already collect `z_per_cycle`.
- In `HRMLanguageModel.forward(...)`, if `deep_supervision=true`, run the decoder for each `z_c` and compute CE loss against `labels`; average with weights (e.g., uniform or linearly increasing). Keep `z_N` path as the primary loss; add DS loss multiplied by `ds_weight`.

**File:** `src/hrm_lm/models/hybrid.py`  
**Patch (concise):**
```diff
- z, _ = self.hrm(x, return_all=False)
+ z, aux = self.hrm(x, return_all=True)
  if self.bridge_type == 'prefix':
    mem = self.hrm2dec(z)
  else:
    mem = self.hrm2dec(z)
  logits = self.decoder(decoder_input_ids, mem,
                        attention_mask=dec_attn_mask, memory_mask=None)
  loss = None
  if labels is not None:
    ce = torch.nn.functional.cross_entropy(
      logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    loss = ce
+   if self.deep_supervision and aux is not None and 'z_per_cycle' in aux:
+     ds_losses = []
+     for zc in aux['z_per_cycle'][:-1]:
+       mem_c = self.hrm2dec(zc)
+       logits_c = self.decoder(decoder_input_ids, mem_c,
+                               attention_mask=dec_attn_mask, memory_mask=None)
+       ce_c = torch.nn.functional.cross_entropy(
+         logits_c.view(-1, logits_c.size(-1)), labels.view(-1), ignore_index=-100)
+       ds_losses.append(ce_c)
+     if len(ds_losses) > 0 and self.ds_weight > 0:
+       loss = loss + self.ds_weight * torch.stack(ds_losses).mean()
  return {'logits': logits, 'loss': loss}
```

**Where to store flags:** In `HRMLanguageModel.__init__`, read `hrm_cfg.get('deep_supervision', False)` and `ds_weight`; set `self.deep_supervision`, `self.ds_weight`.

**B) Halting (optional, later)**
- HRM already has a `halt_head`. If `use_halting=true`, compute per‑cycle halt probs `p_c = sigmoid(halt_head(H[:,0]))`.
- **Training loss:** add `halting_weight * (sum(p_c) - target_cycles)^2` (simple regularizer), or a monotonic halting objective if you prefer ACT‑style halting. For now, keep it simple; we won’t early‑exit at inference until we validate stability.

**Test:** Enable DS on dry run and confirm `loss` increases slightly (due to extra CE terms) but backprop runs.

### Task 3 — Cross‑Attention Bridge Path & Gating

**Goal:** Support two bridge modes you can A/B:
- `bridge.type: "prefix"` (default)
- `bridge.type: "cross_attn"`

We already support both classes. Add a simple **routing gate** (optional) to interpolate between raw decoder conditioning and HRM conditioning:
- Gate `g = sigmoid(wᵀ z + b)`.
- Memory passed to decoder becomes `(1-g)*mem_base + g*mem_hrm`. For simplicity, `mem_base` can be zeros (or a learned null memory).

**File:** `src/hrm_lm/models/bridges.py` (add optional gating module)
```python
class HRMGate(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.lin = nn.Linear(d_model, 1)
  def forward(self, z):
    return torch.sigmoid(self.lin(z))  # (B,1)
```

**File:** `src/hrm_lm/models/hybrid.py` (compute `g` and blend)
```python
self.hrm_gate = HRMGate(d_model)
...
z, aux = self.hrm(x, return_all=True)
g = self.hrm_gate(z)  # (B,1)
if self.bridge_type == 'prefix':
  mem_hrm = self.hrm2dec(z)         # (B,P,D)
  mem_base = torch.zeros_like(mem_hrm)
  mem = (1-g.view(-1,1,1))*mem_base + g.view(-1,1,1)*mem_hrm
else:
  mem_hrm = self.hrm2dec(z)         # (B,1,D)
  mem_base = torch.zeros_like(mem_hrm)
  mem = (1-g.view(-1,1,1))*mem_base + g.view(-1,1,1)*mem_hrm
```

**Acceptance:** Both modes run with identical inputs and produce logits; DS still works.

### Task 4 — Tokenization & Dataset

**Goal:** Move beyond random IDs. Start with a small, deterministic tokenizer and a synthetic reasoning dataset; then swap in a real tokenizer.

**A) Minimal tokenizer (for smoke tests)**  
**File:** `src/hrm_lm/data/simple_tokenizer.py`
```python
# 2-space indent, minimal comments
from collections import defaultdict

PAD, BOS, EOS = 0, 1, 2

class SimpleTokenizer:
  def __init__(self):
    self.stoi = {"<pad>":PAD, "<bos>":BOS, "<eos>":EOS}
    self.itos = {v:k for k,v in self.stoi.items()}
  def fit(self, texts):
    for t in texts:
      for w in t.strip().split():
        if w not in self.stoi:
          i = len(self.stoi); self.stoi[w] = i; self.itos[i] = w
  def encode(self, text, add_specials=True):
    ids = [self.stoi.get(w, None) or self._add(w) for w in text.strip().split()]
    if add_specials: ids = [BOS] + ids + [EOS]
    return ids
  def decode(self, ids):
    toks = [self.itos.get(i, "<unk>") for i in ids if i not in (PAD, BOS, EOS)]
    return " ".join(toks)
  def _add(self, w):
    i = len(self.stoi); self.stoi[w] = i; self.itos[i]=w; return i
```

**B) Synthetic dataset**  
**File:** `src/hrm_lm/data/synthetic.py`
```python
import random, torch
from .simple_tokenizer import SimpleTokenizer, PAD, BOS, EOS

def build_synthetic_dataset(n=1000, seed=1337):
  random.seed(seed)
  pairs = []
  for _ in range(n):
    a = random.randint(10,99); b = random.randint(10,99)
    q = f"What is {a} + {b}?"
    a_str = str(a+b)
    pairs.append((q, a_str))
  tok = SimpleTokenizer()
  tok.fit([q for q,_ in pairs] + [a for _,a in pairs])
  data = []
  for q, ans in pairs:
    x = tok.encode(q, add_specials=True)
    y = tok.encode(ans, add_specials=True)
    # decoder in: BOS + ans[:-1], labels: ans (with EOS)
    y_in = [BOS] + y[1:-1]
    labels = y
    data.append((x, y_in, labels))
  return tok, data

def pad_batch(batch, pad_id=PAD):
  xs, ys_in, ys = zip(*batch)
  S = max(len(x) for x in xs)
  T = max(len(y) for y in ys)
  x_pad = torch.full((len(xs), S), pad_id, dtype=torch.long)
  y_in_pad = torch.full((len(xs), T), pad_id, dtype=torch.long)
  y_pad = torch.full((len(xs), T), -100, dtype=torch.long)
  for i,(x, yi, y) in enumerate(batch):
    x_pad[i,:len(x)] = torch.tensor(x)
    y_in_pad[i,:len(yi)] = torch.tensor(yi)
    y_pad[i,:len(y)] = torch.tensor(y)
  attn_x = (x_pad != pad_id).long()
  attn_y = (y_in_pad != pad_id).long()
  return x_pad, y_in_pad, y_pad, attn_x, attn_y
```

**C) Wire into training**  
**File:** `src/hrm_lm/training/train.py` (add args and a simple dataloader loop when `--dry_run 0`)
```python
from hrm_lm.data.synthetic import build_synthetic_dataset, pad_batch
...
ap.add_argument('--dataset', default=None)
ap.add_argument('--steps', type=int, default=200)
...
if args.dry_run:
  ...
  return

if args.dataset == 'synthetic':
  tok, data = build_synthetic_dataset(n=2000, seed=cfg.train.seed)
  def iter_data():
    import random
    while True:
      batch = [random.choice(data) for _ in range(cfg.train.batch_size)]
      yield pad_batch(batch)
  it = iter_data()
else:
  raise ValueError("dataset required for non-dry run")

for step in range(args.steps):
  x, y_in, y, attn_x, attn_y = next(it)
  x, y_in, y, attn_x, attn_y = x.to(device), y_in.to(device), y.to(device), attn_x.to(device), attn_y.to(device)
  out = model(x, y_in, enc_attn_mask=attn_x, dec_attn_mask=attn_y, labels=y)
  loss = out['loss']
  opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
  if step % 10 == 0:
    print(f"step {step} loss {loss.item():.4f}")
```

**Run:**  
`python -m hrm_lm.training.train --dataset synthetic --steps 200 --dry_run 0`  
Expect loss to trend down.

### Task 5 — Generation Script

**Goal:** Greedy sampling to validate end‑to‑end reasoning + language generation.

**File:** `src/hrm_lm/inference/generate.py`
```python
import torch
from hrm_lm.models.hybrid import HRMLanguageModel

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=64, device='cpu'):
  model.eval(); device = torch.device(device)
  x = torch.tensor([tokenizer.encode(prompt, add_specials=True)], dtype=torch.long, device=device)
  y = torch.tensor([[tokenizer.BOS if hasattr(tokenizer,'BOS') else 1]], dtype=torch.long, device=device)
  attn_x = (x != 0).long()
  for _ in range(max_new_tokens):
    out = model(x, y, enc_attn_mask=attn_x, dec_attn_mask=(y!=0).long())
    logits = out['logits'][:,-1,:]
    nxt = torch.argmax(logits, dim=-1, keepdim=True)
    y = torch.cat([y, nxt], dim=1)
    if int(nxt.item()) == 2: break
  return tokenizer.decode(y[0].tolist())
```

**CLI:** `python -m hrm_lm.inference.generate --prompt "What is 12 + 7?"` (write a small argparse wrapper if desired).

### Task 6 — Training Loop Enhancements

**File:** `src/hrm_lm/training/train.py`
- Add args: `--val_every`, `--save_dir`, `--mixed_precision`, `--grad_clip`.
- Save checkpoints with `torch.save({'state_dict': model.state_dict(), 'cfg': cfg}, path)`.
- Optionally log DS loss separately (when enabled).

### Task 7 — Optional Mamba‑2 Encoder (Linux+CUDA)

- Install `mamba-ssm` & `causal-conv1d` on the RTX 5090 box.
- Set `model.encoder.backend: mamba2` in the YAML.
- Expect the encoder path to switch automatically (`MambaStack.use_mamba=True`).

### Task 8 — Unit & Property Tests

**A) Shapes & CE** (already present, keep):
```
tests/test_forward.py
```

**B) One‑step vs. BPTT memory** (CUDA only; skip on CPU):
**File:** `tests/test_memory.py`
```python
import torch, pytest
from omegaconf import OmegaConf
from hrm_lm.training.train import make_model

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_onestep_uses_less_memory():
  base = {'model': {'vocab_size': 1024, 'd_model': 128,
    'encoder': {'backend':'transformer','n_layers':2,'max_seq_len':256},
    'decoder': {'n_layers':2,'max_seq_len':128},
    'hrm': {'d_model':128,'h_len':4,'l_len':16,'h_layers':1,'l_layers':1,'h_cycles':3,'l_steps':3,'approx_grad':'one_step','out_dim':128,'use_halting':False}},
    'bridge': {'type':'prefix','prefix_len':4}}
  cfg = OmegaConf.create(base)
  m1 = make_model(cfg).cuda()
  x = torch.randint(0,1024,(2,32),device='cuda')
  y_in = torch.randint(0,1024,(2,16),device='cuda')
  y = torch.randint(0,1024,(2,16),device='cuda')
  torch.cuda.reset_peak_memory_stats()
  out = m1(x,y_in,labels=y); out['loss'].backward()
  mem_one = torch.cuda.max_memory_allocated()

  cfg.model.hrm.approx_grad='bptt'
  m2 = make_model(cfg).cuda()
  torch.cuda.reset_peak_memory_stats()
  out = m2(x,y_in,labels=y); out['loss'].backward()
  mem_bptt = torch.cuda.max_memory_allocated()

  assert mem_one < mem_bptt
```

**C) Bridge modes run**  
**File:** `tests/test_bridges.py`
```python
import torch
from omegaconf import OmegaConf
from hrm_lm.training.train import make_model

def run_once(bridge_type):
  cfg = OmegaConf.create({'model': {'vocab_size':512,'d_model':64,
    'encoder': {'backend':'transformer','n_layers':1,'max_seq_len':64},
    'decoder': {'n_layers':1,'max_seq_len':32},
    'hrm': {'d_model':64,'h_len':4,'l_len':8,'h_layers':1,'l_layers':1,'h_cycles':2,'l_steps':2,'approx_grad':'one_step','out_dim':64,'use_halting':False}},
    'bridge': {'type':bridge_type,'prefix_len':4}})
  m = make_model(cfg)
  x = torch.randint(0,512,(2,16))
  y_in = torch.randint(0,512,(2,8))
  y = torch.randint(0,512,(2,8))
  out = m(x,y_in,labels=y)
  assert out['logits'].shape[:2] == (2,8)

def test_prefix(): run_once('prefix')
def test_cross(): run_once('cross_attn')
```

**Run:** `pytest -q`.

### Task 9 — Metrics & Sanity Benchmarks

- **Synthetic arithmetic accuracy** (exact string match).
- **Length‑generalization**: train on 2‑digit sums, test on 3‑digit sums; log accuracy.
- **Ablations**:
  - HRM disabled (bypass) vs. HRM enabled.
  - Prefix vs. Cross‑attn bridge.
  - `h_cycles`/`l_steps` sweep.

**Acceptance:** HRM‑enabled configurations outperform HRM‑bypass on multi‑step tasks.

### Task 10 — (Later) Advanced HRM Training

- **Exact implicit gradient / DEQ‑style:** add a fixed‑point solver and implicit differentiation (Jacobian‑vector products). This is a **research step**; the current one‑step surrogate is sufficient for the prototype but not a perfect replica of implicit differentiation math.
- **Adaptive halting inference:** early‑exit when accumulated halting prob crosses a threshold; clamp max cycles.
- **Decoder as Mamba2:** after stability, try SSM decoder for full linear‑time pipeline.

---

## 4) Self‑Review: Design Choices & Limitations

**What’s strong now**
- A clean, minimal, **end‑to‑end differentiable** skeleton: LM encoder → HRM → LM decoder.
- **One‑step gradient** to bound autograd memory while preserving a meaningful training signal through the last HRM cycle.
- Pluggable **bridges** and **backends** (Transformer/Mamba2).

**What’s intentionally simplified**
- Decoder pos enc was initially zero (now instructed to fix).
- **Deep supervision** and **halting** are not yet wired into the loss (we add them here).
- The **“exact” DEQ gradient** is not implemented; we use a practical surrogate. This is acceptable for prototyping but not a perfect replica of implicit differentiation math.

**Why this still satisfies your goal**
- The skeleton **mathematically couples** LM and HRM and supports joint training.  
- HRM can, in principle, **absorb deep latent computation** while the LM focuses on language I/O—exactly the division of labor we want.
- The project remains within your **parameter budget** (hundreds of M), and is ready to scale later.

---

## 5) Concrete Code Diff Snippets (Copy/Paste Ready)

> Keep indentation at 2 spaces and minimal comments, per your style.

### 5.1 `src/hrm_lm/training/train.py` — print fix
```diff
- print('dry_run loss:', float(out['loss']))
+ print('dry_run loss:', out['loss'].item())
```

### 5.2 `src/hrm_lm/models/decoder.py` — sinusoidal positions
```diff
-from .transformer_layers import TransformerDecoder
+from .transformer_layers import TransformerDecoder, PositionalEncoding

 class LMDecoder(nn.Module):
   def __init__(self, vocab_size, d_model, n_layers, max_seq_len):
     super().__init__()
     self.tok = nn.Embedding(vocab_size, d_model)
-    self.pos = nn.Parameter(torch.zeros(max_seq_len, d_model))
+    self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
     self.dec = TransformerDecoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)
     self.norm = nn.LayerNorm(d_model)
     self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

   def forward(self, input_ids, memory_tokens, attention_mask=None, memory_mask=None):
-    x = self.tok(input_ids) + self.pos[:input_ids.size(1)].unsqueeze(0)
+    x = self.tok(input_ids)
+    x = self.pos(x)
     mem = memory_tokens
     h = self.dec(x, mem,
       tgt_key_padding_mask=(attention_mask==0) if attention_mask is not None else None,
       memory_key_padding_mask=memory_mask)
     h = self.norm(h)
     logits = self.lm_head(h)
     return logits
```

### 5.3 `src/hrm_lm/models/hybrid.py` — deep supervision plumbing
```diff
 class HRMLanguageModel(nn.Module):
   def __init__(self, vocab_size, d_model, enc_layers, dec_layers, max_enc_len, max_dec_len, hrm_cfg, bridge_cfg, enc_backend='transformer'):
     super().__init__()
     self.encoder = LMEncoder(vocab_size, d_model, enc_layers, max_enc_len, backend=enc_backend)
     self.prompt2hrm = PromptToHRMBridge(d_model)
     self.hrm = HRMCore(d_model=d_model, h_len=hrm_cfg['h_len'], l_len=hrm_cfg['l_len'], h_layers=hrm_cfg['h_layers'], l_layers=hrm_cfg['l_layers'], h_cycles=hrm_cfg['h_cycles'], l_steps=hrm_cfg['l_steps'], approx_grad=hrm_cfg['approx_grad'], out_dim=hrm_cfg['out_dim'], use_halting=hrm_cfg.get('use_halting', False))
+    self.deep_supervision = hrm_cfg.get('deep_supervision', False)
+    self.ds_weight = hrm_cfg.get('ds_weight', 0.0)
     self.bridge_type = bridge_cfg['type']
     if self.bridge_type == 'prefix':
       self.hrm2dec = HRMToPrefixBridge(d_model, bridge_cfg['prefix_len'])
     else:
       self.hrm2dec = HRMToCrossAttnBridge(d_model)
     self.decoder = LMDecoder(vocab_size, d_model, dec_layers, max_dec_len)

   def forward(self, input_ids, decoder_input_ids, enc_attn_mask=None, dec_attn_mask=None, labels=None):
     enc_h, cls = self.encoder(input_ids, enc_attn_mask)
     x = self.prompt2hrm(cls)
-    z, _ = self.hrm(x, return_all=False)
+    z, aux = self.hrm(x, return_all=True)
     if self.bridge_type == 'prefix':
       mem = self.hrm2dec(z)
     else:
       mem = self.hrm2dec(z)
     logits = self.decoder(decoder_input_ids, mem, attention_mask=dec_attn_mask, memory_mask=None)
     loss = None
     if labels is not None:
       loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
+      if self.deep_supervision and aux is not None and 'z_per_cycle' in aux:
+        ds = []
+        for zc in aux['z_per_cycle'][:-1]:
+          mem_c = self.hrm2dec(zc)
+          logits_c = self.decoder(decoder_input_ids, mem_c, attention_mask=dec_attn_mask, memory_mask=None)
+          ce_c = torch.nn.functional.cross_entropy(logits_c.view(-1, logits_c.size(-1)), labels.view(-1), ignore_index=-100)
+          ds.append(ce_c)
+        if len(ds) > 0 and self.ds_weight > 0:
+          loss = loss + self.ds_weight * torch.stack(ds).mean()
     return {'logits': logits, 'loss': loss}
```

### 5.4 Config change — enable DS (optional)
**File:** `src/hrm_lm/configs/default.yaml`
```diff
 model:
   hrm:
     d_model: 512
     h_len: 8
     l_len: 64
     h_layers: 2
     l_layers: 2
     h_cycles: 4
     l_steps: 8
     approx_grad: one_step
     use_halting: false
     out_dim: 512
+    deep_supervision: false
+    ds_weight: 0.2
```

### 5.5 Appendix (reference): Encoder mask kwarg patch (already applied)
**File:** `src/hrm_lm/models/encoder.py`
```diff
- h = self.enc(x, key_padding_mask=key_pad) if self.backend != 'mamba2' or not getattr(self.enc, 'use_mamba', False) else self.enc(x, key_padding_mask=key_pad)
+ h = self.enc(x, key_pad)
```

### 5.6 Appendix (reference): run_demo.sh cleanup
Ensure your demo script doesn’t contain stray text. It should be:
```bash
#!/usr/bin/env bash
set -e
source venv/bin/activate 2>/dev/null || source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m hrm_lm.training.train --dry_run 1
```

---

## 6) How to Test Each Increment

**After Task 1 (decoder sinusoidal):**
```bash
python -m hrm_lm.training.train --dry_run 1
# expect: one "dry_run loss: ..." line, no warnings
```

**After Task 2 (DS):**
- Temporarily set `deep_supervision: true` in config; run `--dry_run 1`.  
- Expect same shape/logits, loss still finite. To validate code path, print loss components (temporarily) or assert in a unit test that `loss` changes when DS toggled.

**Bridge A/B (prefix vs cross‑attn):**
- Toggle `bridge.type` in config and run `--dry_run 1`.  
- Expect same success; logits shape identical.

**Unit tests:**
```bash
pytest -q
```

**Synthetic training:**
- After implementing tokenizer + synthetic dataset + dataloader:
```bash
python -m hrm_lm.training.train --dataset synthetic --steps 200 --dry_run 0
```
- Expect training loss to trend down; spot check samples via the generation CLI.

---

## 7) Acceptance Criteria (Prototype Phase)

1) `--dry_run 1` passes with both `bridge.type` values and both `encoder.backend` values (`transformer` always; `mamba2` where installed).  
2) DS toggling doesn’t break training; loss remains finite; backward succeeds.  
3) Synthetic dataset training reduces loss; generation returns coherent answers for simple tasks.  
4) Code remains ≤ ~300M parameters with default config.  
5) Unit tests (shapes, DS toggling, bridge modes) pass locally.

---

## 8) Risk Register & Mitigations

- **Positional encodings on decoder (fixed).**  
  *Mitigation:* switched to sinusoidal.

- **One‑step gradient is a surrogate, not exact DEQ.**  
  *Mitigation:* acceptable for prototyping; plan for implicit differentiation as a later research task.

- **Model might “ignore” HRM and brute-force in decoder.**  
  *Mitigation:* DS loss encourages useful `z_c`; cap decoder capacity (e.g., fewer layers) in ablations; add tasks that require multi‑step reasoning.

- **Mamba2 install friction (macOS).**  
  *Mitigation:* optional; fallback to Transformer path; use Linux+CUDA box for Mamba2 benchmarks.

---

## 9) Roadmap Beyond Prototype

- **Exact implicit gradients:** add fixed‑point solvers + JVPs for DEQ‑style training.
- **Adaptive halting at inference:** budgeted cycles with confidence threshold.
- **Real datasets:** math word problems, code reasoning, scientific QA; add RAG + tools in later iterations.
- **Scaling:** 300M → 1B few‑GPU training, monitor throughput and memory; enable `torch.compile`.

---

## 10) Commands Cheat‑Sheet

```bash
# venv + install
source venv/bin/activate 2>/dev/null || source .venv/bin/activate
uv pip install -e .

# dry run
python -m hrm_lm.training.train --dry_run 1

# run demo script (sets PYTHONPATH if needed)
bash scripts/run_demo.sh

# tests
pytest -q

# toggle bridge
yq -Yi '.bridge.type="cross_attn"' src/hrm_lm/configs/default.yaml
yq -Yi '.bridge.type="prefix"' src/hrm_lm/configs/default.yaml

# enable deep supervision
yq -Yi '.model.hrm.deep_supervision=true' src/hrm_lm/configs/default.yaml
```

---

## 11) Final Self‑Check

- The architecture is **mathematically consistent**: continuous HRM dynamics are injected between language encoding and decoding via a well-defined bridge; training remains end‑to‑end.
- The **implementation plan** is incremental with immediate tests at each step.
- The **limitations are explicit** (exact DEQ gradients not yet implemented), with clear follow‑ups.

> Proceed with Task 1 and Task 2 first. If you hit any friction (especially around DS loss plumbing), capture the stack trace and we’ll iterate directly on that file.
