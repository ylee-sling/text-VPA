# text-VPA (PR-JEPA) — PyTorch reference implementation

Implementation of *PR-JEPA: Decoupling Latent Planning from Few-Step Discrete Realization for
Non-Autoregressive Generation of Code, Legal Documents, and Control* (Yeonseok Lee, SLING AI Inc.,
doi:10.5281/zenodo.23498811). **`docs/pr_jepa_preprint.pdf` is the ground truth** for every formula, tensor
dimension and data dependency. If the code and the PDF disagree, the PDF wins — except for the recorded
decisions below. Read the relevant section of the PDF before changing any math.

## Layout

| Path | Paper | Contents |
|---|---|---|
| `text_vpa/configs/` | — | Pydantic schema (`schema.py`) + Hydra YAML groups (`conf/`), `load_config`, `compose_raw` |
| `text_vpa/models/encoders.py` | Sec. 3.1 | `ContextEncoder` f_θ (pass 1), `TargetEncoder` f_θ̄ (frozen EMA copy, slot pooling) |
| `text_vpa/models/ema.py` | Eq. 4 | `EMATargetEncoder`, linear (paper) and cosine momentum schedules |
| `text_vpa/models/selector.py` | Sec. 3.1, Prop. 1 | `ModeSelector` π_φ^h (pass 2), `choose_mode` (sample / greedy / Gumbel) |
| `text_vpa/models/predictor.py` | Eqs. 3, 8 | `JEPAPredictor` g_φ (pass 3), `ensemble_mean`, `ensemble_uncertainty` (σ̄) |
| `text_vpa/models/bridge.py` | Sec. 3.2, Eq. 10 | `CrossAttentionBridge`: Z̃ = [Z; s_x; e(u)]/γ, spectral norm on W_K, W_V, cached K/V, plan dropout |
| `text_vpa/models/realizers/` | Secs. 3.3–3.4, Eqs. 11–21 | `ConditionalFSDFMRealizer`, `LogitKLRealizer`, AdaLN `RealizerBackbone`, `kernels.py` (Eq. 17), `paths.py` |
| `text_vpa/models/readout.py` | Sec. 3.3 | `PlanReadout`, the plan's slot decoder (kinds, slot texts, token inventory, length) |
| `text_vpa/models/plan.py` | Sec. 5.1 | `PredictedPlan` + `require_predicted_plan` (anti-leakage guard) |
| `text_vpa/models/text_vpa_model.py` | Fig. 1 | `TextVPAPlanner` (passes 1–3), `TextVPAModel` |
| `text_vpa/engine/losses.py` | Eqs. 5–7, Thm. 1, Props. 2–3 | `VICRegLoss`, `PlanLoss`, `ModeCrossEntropyLoss`, stable rank, `StableRankMonitor` |
| `text_vpa/engine/trainer_stage1.py` | Sec. 5.1 | Stage 1 (L_plan, EMA, collapse rejection) |
| `text_vpa/engine/trainer_stage2.py` | Sec. 5.1 | Stage 2 (frozen planner, anti-leakage invariant), Stage 3 (joint) |
| `text_vpa/engine/runtime.py` | — | DDP, bf16, AdamW groups, schedule, checkpoints, metric logging |
| `text_vpa/inference/sampler.py` | Algorithm 1 | `PRJEPASampler`, `DepthAudit` (K + 3) |
| `text_vpa/inference/energy_terms.py` | Sec. 3.3, Prop. 4 | energies, `PrefixLexer`, Prop. 4(d) check, network-free repair |
| `text_vpa/data/` | Table 1, Sec. 5.1 | tokenizers, tree-sitter `TreeSitterASTExtractor` + `ModeExtractor`, dataset (compact `Example`, cache, parallel build), collator, synthetic corpus |
| `text_vpa/data/corpus.py` | Sec. 5.1 "Data" | corpus builder: permissive licenses, self-contained documented functions, AST dedup, 13-word-gram decontamination |
| `text_vpa/evaluation/` | Sec. 5.2 | unbiased pass@k, resource-limited execution, HumanEval / MBPP tasks (`evaluate.py`) |
| `text_vpa/utils/verification.py` | Sec. 4.6, App. B | NumPy port of the paper's numerical checks (Table 2) |
| `text_vpa/utils/profiler.py` | Sec. 5.4, App. A.2 | roofline (Table 3), measured latency, simulated AR decoder |
| `text_vpa/utils/memory_probe.py` | — | peak GPU memory and step time of a config on worst-case batches |
| `train.py`, `generate.py`, `evaluate.py` | — | CLIs |
| `scripts/init_repo.sh`, `scripts/release.sh` | — | git setup; checks → commit → tag → push |
| `scripts/gcp/` | — | `create_vm.sh` (g2-standard-16), `setup_vm.sh`, `train_l4.sh` (resumable pipeline) |

## Commands

```bash
pip install -r requirements-dev.txt          # torch first for CUDA / ROCm / MPS
python -m pytest                             # fast suite (~1 min on CPU); must pass
python -m pytest -m slow                     # convergence test (~2 min): each context realizes its own program
ruff check .                                 # must pass
python -m text_vpa.utils.verification        # Appendix B checks; must reproduce Table 2
python -m text_vpa.utils.profiler --roofline # must reproduce Table 3
python evaluate.py --benchmark humaneval --data HumanEval.jsonl.gz --reference --out /tmp/he  # harness: 164/164
```

## Non-negotiable rules

1. **Mathematical strictness.** Every tensor operation implements an equation of the PDF; cite the equation
   in a comment. No approximations or alternative algorithms without a recorded decision below.
2. **Shapes and types.** Every public class and `forward` has type hints and a docstring with shapes (`[B, N, S]`).
3. **K + 3 sequential depth.** `PRJEPASampler.sample` must evaluate exactly `[f_θ, π_φ^h, g_φ] + [F_ψ] * K`;
   the bridge projections and the readout run once, between g_φ and the first F_ψ pass (fused into pass 4).
   `DepthAudit` checks this with forward hooks and raises; never remove or weaken it. Call networks as
   `module(...)`, never `module.forward(...)`. Guidance batches the conditional and null passes in one call.
4. **Anti-leakage (Stage 2).** The realizer conditions only on `PredictedPlan` from `TextVPAPlanner.plan`, never
   on `f_θ̄(y)`. Stage 2 has no target encoder; the planner is frozen, out of the optimizer, in eval mode,
   and fingerprinted (SHA-256). Do not weaken `require_predicted_plan` or `Stage2Trainer.assert_planner_frozen`.
5. **Prop. 4(d).** Plan-mediated hard energies depend on (j, Z) only; draft-dependent terms read only the
   prefix w_{<i} (so the left-to-right repair is exact). Keep `check_parallel_update` active in the sampler.
6. **No stubs** in core logic (losses, kernels, sampler, energies, trainers).

## Tests

- References are computed independently (explicit float64 loops / NumPy) from the PDF, never with the
  function under test. Fixed seeds; values, not only shapes. float32 vs float64: `rtol=1e-5, atol=1e-6`
  unless a stated numerical reason requires more. Monte Carlo checks: 5 standard errors.
- A test that exposes a library defect is kept as `pytest.mark.xfail(strict=True, reason=...)` until fixed,
  then the marker is removed (the test stays as a regression guard). Never weaken a correct reference.
- Self-tests run on CPU. MPS has no float64: never put float64 in parameters, buffers or forward paths.

## Decisions where the paper is silent (do not change silently)

- **Invariance normalisation.** `VICRegConfig.invariance_normalization = per_dim`: the prediction term is
  the per-element MSE (VICReg's form), so 25 : 25 : 1 keeps VICReg's balance at every d. Eq. 5 sums the
  squared error over d; with λ_invar = 25 that over-weights it by d and the plan's variance collapsed in
  Stage-1 runs. `frobenius` with `lambda_invar=1` is Eq. 5 verbatim.
- **Collapse guard.** srank(C) < d/2 aborts Stage 1 and writes `stage1_rejected.pt` only after
  `training.collapse_grace_steps` (random initialisations are low-rank; the paper checks per epoch).
- **Slots.** g_φ runs on all M slot queries at training and inference; the prediction term and VICReg rows
  use the reference's real slots; the bridge conditions on all M slots.
- **Plan readout.** The paper's "slot decoder" is `PlanReadout`, trained in Stages 2–3 on the predicted plan
  (supervision, never conditioning). Energies use its decoded plan at inference, or the oracle
  `PlanConstraints.from_reference` for evaluation.
- **Energies.** `PlanVocabularyEnergy` licenses identifier-bearing *tokens* (plan inventory + keywords +
  builtins), plan-mediated and hard. `GrammarEnergy` (bracket legality outside strings/comments, incl. PEP 701
  f-strings; [PAD] inside an open bracket or string) is hard and draft-dependent. `ArityEnergy` and
  `TypeSignatureEnergy` are soft (5.0) and draft-dependent. The final repair is network-free: left to right
  with the energies of the repaired prefix, it replaces hard violations, leftover [MASK]s and positions that
  never left w_0 (stalls: a stale draft-dependent term rejected the posterior's token, Prop. 4(b)), by
  argmax p·e^{-E/T} with p the last untilted posterior.
- **Source at inference.** w_0 is uniform over the plan-admissible tokens (`sampler.restrict_source`), so the
  premise of Prop. 4(d) holds from t = 0; training draws w_0 uniformly over V.
- **Stage 2 mode.** g_φ gets the reference mode (teacher forcing) by default; the plan is still predicted
  from the context. `training.stage2_mode_source=sampled` draws it from π_φ^h.
- **Time sampling.** Training states use t = k/K on the step-budget grid (Algorithm 1's evaluation points).
- **Canvas length.** Training canvases are target + [PAD] rounded up to `data.bucket_size`; the sampler takes
  N from the readout's length head by default. N is a cue the realizer can use, so tests that force a fixed
  length train with `data.bucket_size = data.max_output_tokens`.
- **Modes.** Rule-based AST features (`control` × `library`, 30 modes), not learned clusters.
- **AdaLN.** "adaLN-single" (shared projection + per-block tables), zero-initialised gates.
- **σ̄.** Plain RMS of Eq. 8 (VPA divided by √d γ*).
- **Stage 3 EMA.** The momentum schedule restarts at m0 over the Stage-3 steps.
- **Gradient accumulation.** `training.<stage>.grad_accum_steps`: micro-batch losses scaled by 1/k, DDP
  all-reduce on the last one. VICReg statistics are per micro-batch and the spectral-norm power iteration
  advances once per micro-batch (so accumulation is exact only for Eq. 12; tested with the iteration frozen).
- **Examples.** `Example` stores int32 arrays and only the text slots' ids; the list properties rebuild the
  dense layout. Never store per-token Python lists for a corpus (~100 KB per example at M = 256).
- **Decontamination.** Drop a training function sharing any 13-word n-gram (identifiers and numbers) with
  HumanEval / MBPP; punctuation tokens made shared signatures count as contamination.
- **Not implemented.** FS-DFM distillation from a 1,024-step teacher; HLS-C bit-width / directive-legality
  energies; contract energies; the continuous-control solver.
