# Experiment Strategy — Inference-Time Ensembling in TabICLv2

Derived from reading `tabicl` v2.2.0 (`src/tabicl/_sklearn/{classifier,preprocessing}.py`).
Pin a commit SHA before starting; line numbers below refer to that reading.

---

## 0. Executive summary

Reading the code changed five things about the plan. In order of impact:

1. **The paper misstates the shipped aggregator.** TabICLv2 defaults to
   `average_logits=True` with `softmax_temperature=0.9` — it pools in *logit*
   space and applies a tempered softmax. The paper's aggregator table labels
   uniform *probability* averaging as "shipped". This must be corrected.
2. **The member cache is a ~30-line subclass, not a fork.** Per-member outputs
   already exist as a stacked array inside `predict_proba` immediately before
   aggregation.
3. **Three of the six axes have hard structural caps** that make the planned
   `M = 64` sweep impossible on them. Only one native axis scales.
4. **A6 is empty by construction** and should be repurposed as a harness
   determinism check.
5. **`n_estimators` is a request, not a guarantee** — the realised member count
   varies per dataset and must be logged or the scaling curves are corrupted.

---

## 1. Findings that change the design

### 1.1 The shipped aggregator is logit pooling with temperature

`classifier.py` defaults: `average_logits: bool = True`, `softmax_temperature: float = 0.9`.

The aggregation in `predict_proba` is:

```python
avg = np.zeros_like(outputs[0])
for i, shuffle in enumerate(class_shuffles):
    avg += outputs[i][..., shuffle]          # unshuffle, then sum
avg /= n_estimators
if self.average_logits:
    avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)
return avg / avg.sum(axis=1, keepdims=True)
```

So the shipped rule is **logarithmic pooling** (arithmetic mean of logits ≡
geometric mean of probabilities, up to normalisation), followed by a softmax at
`τ = 0.9`. Consequences:

- The aggregator table's "shipped" row is logarithmic pooling, not linear.
- `average_logits=False` gives the linear-pool comparison as a **one-flag
  ablation** — no implementation needed.
- The model *already* applies a global temperature to the aggregate. The
  "temperature scaling on aggregate" row is therefore partly shipped, and
  `τ = 0.9` is a tuned constant nobody has justified. **Sweeping τ is a free
  extra experiment** and a plausible calibration result on its own.
- The paper's framing sentence — "the reported prediction is almost always the
  uniform arithmetic mean" — is false for this model. Soften to: most systems
  pool uniformly; the space is pooled *in* differs and is undocumented.

### 1.2 Members are an arbitrary slice of the product space

`preprocessing.py :: EnsembleGenerator._generate_ensemble` (~L1090):

```python
shuffle_configs = list(itertools.product(X_shuffles, y_patterns))
self.rng_.shuffle(shuffle_configs)
shuffle_norm_configs = list(itertools.product(shuffle_configs, self.norm_methods_))
shuffle_norm_configs = shuffle_norm_configs[: self.n_estimators]
```

The full Cartesian product is built, **randomly permuted, then truncated** to
`n_estimators`. There is no balancing across axes and no round-robin guarantee
(contrast TabPFN-3, which round-robins feature subsets so no feature is
systematically excluded).

This is a substantive observation for §2.3 of the paper: the shipped 8 members
are an arbitrary random slice, so axis coverage at the default budget is
uncontrolled and varies run to run with `random_state`.

### 1.3 Structural caps per axis

| Axis | Controlled by | `n_elements` | Max distinct members |
|---|---|---|---|
| A1 feature order | `feat_shuffle_method` | `n_features_in_` | `shift`: `n_features`; `latin`: latin squares; **`random`: uncapped** (>5 features) |
| A2 class order | `class_shuffle_method` | `n_classes_` | `shift`: **exactly `n_classes`** |
| A3 preprocessing | `norm_methods` | — | **5** (`none`, `power`, `quantile`, `quantile_rtdl`, `robust`) |
| A6 seed | `random_state` | — | **1** (see §1.4) |

Three consequences:

- **A2 is near-dead on binary tasks** — exactly 2 distinct class permutations.
  TabArena classification is heavily binary, so a pooled `φ_A2 ≈ 0` would be a
  structural artifact, not a finding. **Stratify the attribution by `n_classes`**
  and report binary and multiclass separately, or the result is misleading.
- **A3 is capped at 5.** The budget sweep cannot exceed 5 members on A3 alone.
- **Only `feat_shuffle_method='random'` reaches M = 64.** The `M ∈ {1,…,64}`
  sweep is feasible on A1-random, and on A4/A5 which you implement yourself.
  For A2 and A3 the sweep tops out at `n_classes` and 5.

Revise the budget grid to `M' ∈ {1, 2, 4, 8, 16, 32, 64}` **capped per axis**,
and report the cap explicitly in the figure rather than letting curves flatten
for structural reasons that look like saturation.

### 1.4 A6 is structurally empty — repurpose it

`Shuffler.shuffle`: `if method == "none" or n_estimators == 1: return [indices]` —
a single identity pattern. With all shuffles `none` and one norm method, the
generator yields exactly **one** unique member, and the forward pass is
deterministic.

So `φ_A6 = 0` by construction, not by measurement. The one exception is
`quantile_rtdl`, which "adds noise to training data before fitting" and is
therefore the only genuinely stochastic preprocessing path.

**Use A6 as the harness determinism control.** Run `M = 8` with every axis set
to `none`, one norm method, varying only `random_state`. The measured spread
must be *exactly* zero. Anything nonzero means nondeterminism leaked in (GPU
reduction order, TF32, AMP, MPS kernels) — which would silently inflate every
diversity diagnostic in the study. This is a cheap, high-value pre-flight check
and it belongs in the appendix as evidence the pipeline is sound.

### 1.5 The realised member count is not the requested one

```python
n_estimators = len(class_shuffles)
# May be fewer than requested if dataset has quite limited features and classes
```

`M_requested ≠ M_realised`. **Log `M_realised` per (dataset, split, config)** and
key every cached tensor by it. If the scaling curves are plotted against the
requested budget, datasets that silently returned fewer members will distort
them.

### 1.6 Axes do not cost the same — attribute gain *per GPU-second*

This is the analytical move the efficiency framing turns on.

`EnsembleGenerator.transform` groups members by normalisation method, and
`predict_proba` iterates `for norm_method, (Xs, ys) in data.items()`. Members
within one norm method share a preprocessing fit and a normalised copy of the
data; a new norm method requires its own. Likewise `model_kv_cache_` is keyed by
norm method. So the marginal cost of member *m+1* is **not constant** — it
depends on which axis produced it.

Structurally: A1 and A2 members vary only the permutation applied to an
already-normalised array, while each additional A3 member drags in a fresh
preprocessing pipeline and its own cached tensors. A3 is therefore expected to
be the most expensive axis per member, and A5 more expensive still, since
changing the context invalidates any training-side reuse.

Consequence for the paper: ranking axes by φ alone answers the wrong question.
The efficiency claim needs

$$\tilde\phi_a = \phi_a \,/\, \text{(GPU-seconds per member on axis } a)$$

An axis with the largest φ and the worst φ̃ is a *bad* place to spend a fixed
budget, and that inversion — if it happens — is a more interesting result than
the raw attribution. **Measure per-member wall-clock and peak memory per axis in
the pilot** (`run_cell.py` records `seconds_per_member` and `peak_mem_bytes`);
do not infer cost from forward-pass counts.

---

## 2. Instrumentation

### 2.1 Where to intercept

`classifier.py :: predict_proba`. Immediately before the aggregation loop, two
objects hold everything needed:

- `outputs` — `(n_members, n_test, n_classes)`, logits when `average_logits=True`
- `class_shuffles` — the per-member permutation needed to align class indices

No fork of the inference loop is required.

### 2.2 The capture subclass

Duplicate `predict_proba` into a subclass method that returns the aligned member
tensor instead of collapsing it:

```python
class MemberCapturingTabICLClassifier(TabICLClassifier):
    def predict_members(self, X) -> np.ndarray:
        """Return (M_realised, n_test, n_classes) aligned member outputs.

        Identical to predict_proba up to aggregation. Returns LOGITS when
        self.average_logits, else probabilities.
        """
        # ... body copied verbatim from predict_proba up to and including
        #     the construction of `outputs` and `class_shuffles` ...
        members = np.stack(
            [outputs[i][..., shuffle] for i, shuffle in enumerate(class_shuffles)],
            axis=0,
        )
        return members
```

Duplication is the right call over a monkeypatch: it is explicit, it fails loudly
if upstream changes, and pinning the commit SHA makes it auditable. Add a test
asserting that your own aggregation of `predict_members` reproduces
`predict_proba` bit-for-bit — that single test is what licenses every downstream
claim.

### 2.3 Store logits, not probabilities

**Cache `outputs` in logit space with `average_logits=True`, and store `τ`
alongside.** Logits → probabilities is one-way; the reverse loses scale. Storing
logits keeps the whole aggregator library (log pooling, linear pooling, median,
trimmed, temperature sweeps) available post-hoc from one cache. Storing
probabilities forecloses roughly half of it and would force a re-run.

Cache layout:

```
cache/{dataset}/{split}/{coalition}/
    members.npy      # float16 (M, n_test, C) — logits
    meta.json        # M_requested, M_realised, axis config, τ,
                     # random_state, checkpoint SHA, tabicl commit SHA
```

float16 halves storage at negligible precision cost for logits; verify on one
dataset that float16 round-trip does not change argmax or move log-loss beyond
tolerance, then apply everywhere.

---

## 3. Axis isolation recipe

Exact constructor arguments to vary one axis at a time. Everything not listed
stays at default.

| Coalition | `feat_shuffle_method` | `class_shuffle_method` | `norm_methods` | Max M |
|---|---|---|---|---|
| ∅ (single member) | `none` | `none` | `["none"]` | 1 |
| A1 | `random` | `none` | `["none"]` | 64 |
| A2 | `none` | `shift` | `["none"]` | `n_classes` |
| A3 | `none` | `none` | all 5 | 5 |
| A1+A2 | `random` | `shift` | `["none"]` | 64 |
| A1+A3 | `random` | `none` | all 5 | 64 |
| A2+A3 | `none` | `shift` | all 5 | `5·n_classes` |
| A1+A2+A3 | `random` | `shift` | all 5 | 64 |
| **shipped** | `latin` | `shift` | `["none","power"]` | 8 |

Include the shipped configuration as its own coalition — it is the baseline the
paper is about, and it is *not* any of the clean coalitions.

**A4 and A5 you implement.** Both sit outside the generator:

- **A4 (feature subsampling):** wrap `fit`/`predict` over column subsets and
  aggregate yourself. Mirror TabPFN-3's round-robin so each feature appears in
  at least one member; that makes the cross-model comparison in the extension
  meaningful.
- **A5 (context construction):** subsample or select rows of the training
  context per member. Start with uniform random subsampling at a fixed fraction
  — it is the cleanest test of H1 and needs no retrieval index. Class-balanced
  and query-local retrieval are the follow-ups.

Note `y_patterns = [None]` for regression, so **A2 does not exist for the
regressor**. Scope A2 to classification and say so.

---

## 4. Execution plan — pilot first

The full campaign is not committed until the pilot says which paper is being
written. Scope is **efficiency**: what the wrapper costs, what it buys, where it
saturates, and the smallest budget that holds the accuracy. Worst-group accuracy
and any fairness framing are **out of scope** — drop WGA from the metric set and
keep it tight.

### Phase 0 — Gates (must pass before anything else)

| # | Gate | Script | Pass condition |
|---|---|---|---|
| 0.1 | Capture ≡ `predict_proba` | `slurm/02_a6_control.sh` | bitwise equal |
| 0.2 | A6 determinism | `slurm/02_a6_control.sh` | max\|Δ\| **exactly** 0 across seeds |
| 0.3 | `M_realised` = 1 with all axes off | same | 1 |
| 0.4 | float16 round-trip | local | argmax unchanged, Δlog-loss < 1e-4 |
| 0.5 | Reproduce shipped TabArena number | local, 3 datasets | within published CI |

Gate 0.2 is the one that matters most. If it fails, every diversity number in
the study is inflated by numerics rather than by perturbation, and no amount of
downstream analysis recovers it.

### Pilot — 8 datasets × 9 coalitions, M ≤ 32

`slurm/01_pilot_array.sh`. Datasets chosen to span the structural caps rather
than to be representative (`experiments/datasets.py::PILOT`): `d=5` and
`d=1777` for the A1 cap, `C=2` and `C=8` for the A2 cap, `n=748` to `n=32769`
for cost scaling. Estimated a few H100-hours.

The pilot must answer four questions, in priority order:

1. **Is the effect above the noise?** Measure per-dataset split-to-split
   variance and derive a minimum detectable effect. *If the MDE exceeds
   plausible axis effects, drop the Shapley machinery* — attributing a 0.05%
   total gain across five axes when split variance is 1% is measuring noise
   with elegant tooling. Fall back to paired one-at-a-time contrasts.
2. **Flat world or sharp world?** Does any native axis dominate, and what is
   total gain over a single member?
3. **What does a member cost per axis?** (§1.6) — `seconds_per_member` and
   `peak_mem_bytes` per coalition, which gives φ̃ and may invert the ranking.
4. **What is the real full-campaign estimate** on this hardware?

### Decision gate after the pilot

| Pilot outcome | Next |
|---|---|
| MDE > plausible effects | Drop Shapley. Paired contrasts + budget curves only. Paper becomes a saturation/compute study |
| Flat, total gain < ~0.3% | Skip A4. Go straight to budget curves + adaptive rule; the contribution is "the wrapper is nearly free of value, here is the budget you can reclaim" |
| One axis dominates | Full Shapley over {A1,A2,A3,A5}, then A4. The attribution is the headline |
| Cost ranking inverts gain ranking | φ̃ becomes the paper's central table |

### Phases 1–4 (conditional on the gate)

1. Native axes at full scale on official TabArena splits.
2. Add A5 (context), then A4 — 32 coalitions over {A1,A2,A3,A4,A5}.
   **A6 is skipped**: structurally zero, argued analytically and evidenced by
   the Phase 0 control. No compute spent.
3. Aggregators and the τ sweep — pure post-hoc on the cache, no GPU.
4. Adaptive budget rule; report forward passes **and GPU-seconds** saved against
   the shipped default of 8.

Dropping A6 takes the sweep from 64 coalitions to **32**.

---

## 5. Confound checklist

- [ ] `M_realised` logged and used as the x-axis, never `M_requested`
- [ ] Attribution stratified by `n_classes` (binary vs multiclass) — §1.3
- [ ] Per-axis caps drawn on scaling figures so structural ceilings are not read
      as saturation
- [ ] `random_state` fixed per (dataset, split, coalition) and recorded
- [ ] A6 determinism control passes at exactly zero before any other run
- [ ] `kv_cache` **left off** for the main campaign. The cached path
      (`_batch_forward_with_cache`) does not pass `feature_shuffles`, unlike the
      uncached path — so caching may change member semantics. If enabling it for
      speed, first assert cached and uncached members are identical.
- [ ] AMP / TF32 / MPS fallback settings fixed and recorded; mixed precision is a
      plausible source of the nondeterminism the A6 control is designed to catch
- [ ] Checkpoint pinned to `tabicl-classifier-v2-20260212.ckpt`
- [ ] `tabicl` commit SHA recorded in every `meta.json`

---

## 6. Compute Canada

H100, allocation-based. Scripts in `slurm/`.

```
slurm/00_setup_env.sh      login node, once   — venv + checkpoint + datasets
slurm/02_a6_control.sh     45 min, 1 GPU      — Phase 0 gates. Run FIRST.
slurm/01_pilot_array.sh    array 0-71%12      — the pilot
```

**Compute nodes have no internet.** The HF checkpoint, the OpenML files and
every pip package must be fetched on a login node first; `00_setup_env.sh` does
all three. `HF_HUB_OFFLINE=1` is exported in the job scripts so a stray fetch
fails immediately instead of hanging until the wall clock expires. This is the
most common way to lose a day on CC.

**Array layout.** One task = one `(dataset, coalition)` cell, looping splits
internally. `run_cell.py` skips any split whose `meta.json` exists, so the job
is idempotent: resubmitting the same `sbatch` after a preemption or timeout
resumes at the first missing split. The header carries a `sacct` one-liner to
requeue only non-`COMPLETED` tasks.

**Before submitting**, set `--account=def-CHANGEME`, and check the GPU request
line — some CC clusters want `--gpus-per-node=h100:1` rather than
`--gres=gpu:h100:1`. `--time=03:00:00` is deliberate: short jobs clear the queue
far sooner than long ones, and idempotent resume makes a timeout cheap.

**Determinism.** `CUBLAS_WORKSPACE_CONFIG`, `NVIDIA_TF32_OVERRIDE=0` and
`PYTHONHASHSEED` are pinned identically in every script. Gate 0.2 asserts
exactly-zero seed spread, which is only meaningful if the numerics are fixed —
so these must not drift between the control run and the campaign.

Fill the paper's `\TODO{measure}` per-member cost from pilot `meta.json`
(`seconds_per_member`), not from the README's figures.

---

## 7. Paper edits required

| Location | Change |
|---|---|
| §2.1, aggregation sentence | "almost always the uniform arithmetic mean" is false for TabICLv2 — pooling happens in logit space with `τ = 0.9` |
| §4.2 aggregators | Shipped row = logarithmic pooling + temperature, not linear pooling. Add a `τ` sweep |
| §2.3 | Add: members are a randomly-permuted truncation of the product space; no balanced axis coverage |
| §2.2 / Table 1 | Add structural caps per axis; note A2 ⊥ regression |
| §4.5 H2 corollary | `φ_A6 ≈ 0` is structural, not empirical — restate A6 as a determinism control |
| §5 protocol | `M = 64` is reachable only on A1-random, A4, A5; cap the grid per axis |
| §5 compute | 64 → 32 coalitions after dropping A6; re-derive from pilot timings |
| §5 metrics | **Drop worst-group accuracy.** Scope is efficiency; keep accuracy/ROC-AUC/log-loss, ECE, AURC, and add GPU-seconds + peak memory |
| §4 attribution (new) | Add φ̃ = φ / GPU-seconds-per-member (§1.6). Axes have different marginal costs, so ranking on φ alone answers the wrong question |
| §6.5 adaptive budget | Promote from closing contribution to the paper's method: report savings in GPU-seconds, not only forward passes |
| Appendix (new) | `M_realised` vs `M_requested` across TabArena |
| Appendix (new) | A6 determinism control as evidence the diversity numbers are not numerical artifacts |
