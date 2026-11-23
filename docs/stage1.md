# Text-VPA Stage 1: a planner on the Stage 0 realizer (design)

Written on 21 Nov 2026, after the second Stage 0 run. It fixes the order of experiments, their gates and the open
choices before any of them exists, in the way `docs/stage0.md` did for Stage 0.

**Status (22 Nov 2026, 15:00).** The author confirmed the design's three open choices on 21 Nov: slots are
lines, modes are line skeletons, the bridge has no s_x rows. Step 1, the structure oracle, ran on 21 to 22 Nov
(`runs/stage1-oracle`, finished 10:38 KST): **clearly under 0.1** (rho = 0.000 [0.000, 0.000]; "Step 1: result"
below). It also answers steps 3 and 4 as designed (same section). Step 2, the latent oracle, **is running** since
22 Nov, about 14:55 KST (`runs/stage1-latent`, commit `c226d2f`, `TRAIN_ADAPTERS` = 1); its gate and its revised
reading rule were confirmed by the author before the run ("Step 2 as built"). Steps 3 and 4 are not built.

## The question

Does a plan let the Stage 0 realizer decide many tokens in one pass? The realizer is `runs/stage0-a` (Gemma 4
E2B, canvas-only adapters, 10,000 steps). On MBPP's validation split (90 problems, 5 samples, Pass@1):

| | baseline | K = 256, left to right | K = 128 | K = 32 | K = 8 |
|---|---|---|---|---|---|
| released model | 23.8% | | | | |
| realizer, left to right | | 17.1% | 17.1% | 1.6% | 0.0% |
| realizer, confidence order | | | 9.8% | 2.0% | 0.0% |
| realizer, exact kernel (the paper's sampler) | | | 0.7% | 1.1% | 0.4% |

H2 asks for half of the baseline at K = 8; the realizer alone is at 0.019 of it. Two facts shape the design:

- **Order matters more than the number of steps.** At K = 128, two tokens are decided per step in every order,
  and Pass@1 is 17.1%, 9.8% or 0.7% depending on where they are. A token far from known context does not know
  its line, its offset in the line, or what stands before it.
- **Adjacent tokens decided together collapse.** Left to right at K = 32 (4 to 6 neighbours per step): 1.6%.

A plan can address the first directly, by telling every canvas position where it is in the program. It can
address the second only by making the program nearly determined before the realizer starts.

## What the paper's own theory says to expect

Proposition 1 (Sec. 4.1, "Why the plan must carry a mode"):

- (a) A deterministic plan Z = g(c) leaves the error of one factorised parallel step, TC(W | c), unchanged. The
  realizer already sees the prompt c; a function of c gives it no information it lacks.
- (b) A *sampled* discrete mode u removes the between-mode part: the error becomes TC(W | c, U), which is zero
  only if each mode's target factorises, and what the mode can remove is bounded by N H(U | c) (Eq. 23).

Sec. 3.1 says the same in words: "The mode is not decoration. A predictor trained with squared error estimates
the conditional mean of s_y", which "need not decode to any valid program".

So in this architecture the predictor g_phi cannot, in theory, reduce the factorization error on its own. What
can is the mode u, and Z's job is to carry (c, u) to every canvas position in a form the realizer can use in
one pass. Two consequences for the design:

1. **The mode must say something about the program.** The repository's 30 rule-based modes (control flow x
   library) say almost nothing about where tokens go. This design makes the mode a *line skeleton* (below),
   which is what the paper's "clusters of AST shapes" (Sec. 5.1) becomes when taken literally.
2. **An oracle comes first.** Before a planner is built, the realizer is given the reference program's true
   structure. If the true structure does not make K = 8 work, no predicted or sampled structure will.

What to expect, stated before the runs: the structure oracle should lift the exact kernel clearly at K = 32
and K = 128, because it removes the unknown position that breaks random order. Whether it reaches half of the
baseline at K = 8 is open; identifiers and expressions inside a line stay undetermined, and K = 8 still decides
up to ten program tokens per pass. A predicted plan will do worse than the oracle by construction.

If structure turns out not to be enough, or skeletons cannot be enumerated (the codebook's coverage below), the
paper names its own alternative: "a stochastic latent predictor", flow matching inside the predictor, "could
replace our mode selector where modes are hard to enumerate" (Sec. 2). That would be a different stage, not a
variant of this one.

## Order of the experiments, and what each decides

All on MBPP's validation split, 90 problems x 5 samples, never a test set. Ratios are to the baseline (23.8%).

| step | what runs | L4 time | decides |
|---|---|---|---|
| 1 | **Structure oracle** (diagnostic): the realizer plus the bridge, trained on the reference's true line structure | 5,000 steps, about 7 h, + 3 h of evaluation | whether structure, the most a sampled mode can carry, guides parallel decisions |
| 2 | **Latent oracle** (diagnostic), only if step 1 is under 0.5 at K = 8: the same with pooled backbone states of the reference's lines | about 7 h + 3 h | whether any slot plan can drive this realizer at K = 8 |
| 3 | **Planner** (Stage 1 of the paper): f_theta, pi_phi^h, g_phi with L_plan | 20,000 steps, about 3 h (to be measured) | whether structure is predictable from the prompt |
| 4 | **Realizer on predicted plans** (Stage 2 of the paper), planner frozen | 10,000 steps, about 14 h, + 4 h | the method's numbers |

Gates, fixed now:

- **After step 1** (exact kernel, K = 8):
  - at least 0.5: structure is enough in principle. Build the planner (steps 3, 4).
  - between 0.1 and 0.5: H2 is out of reach for a structural plan with this realizer. Steps 3 and 4 still
    measure what the plan buys at K = 32 and K = 128; step 2 says whether richer plans are worth designing.
  - under 0.1: run step 2. If that is also under 0.1, the plan route is closed for this realizer, and the
    choice is between the paper's distillation (Sec. 5.1) and more depth.
- **After step 3:** reported, not gated: the selector's top-1 and top-5 accuracy, the share of held-out programs
  whose line count and whose per-line lengths (within one token) are read correctly from the predicted plan by
  a linear probe, and the same probe on the target plan.
- **After step 4:** H2's ratio at K = 8, exact kernel. At least 0.5: a protocol run on both benchmarks under
  the rule of `docs/stage0.md`, decision 17. Otherwise the stage reports the plan's effect at K = 8, 32, 128
  with its sign, by paired bootstrap, against three references: `runs/stage0-a`; the same model with the null
  plan; the same model with another problem's plan.

**The oracles are diagnostics and never the method.** They condition the realizer on the reference, which is
the leak the anti-leakage rule forbids. They get their own plan type (`OraclePlan`), which the Stage 2 trainer
rejects as it rejects `TargetOutput`; their run directories and reports carry "oracle: not the method"; no
oracle number goes into a results table of the paper except as a labelled upper bound.

## Components

Shapes: B examples, P prompt tokens, N = 256 canvas tokens, M = 32 slots, d = 256, |U| = 1,024.

**Slots: lines.** Slot m is line m of the canvas program (lines from the 32nd on share the last slot). A token
belongs to the line of its first character (tokenizer offsets). Per slot: `kind` (the line's first token: one
of about 24 classes, `def`, `return`, `if`, `for`, an assignment, a call, a string, blank, ...), `depth`
(indentation / 4, at most 7), `start` and `length` in canvas tokens. Lines, not the AST regions of
`data/ast_parser.py`: those cover headers and first bindings only, and a canvas position needs the line it is
in. Preflight reports how many corpus and benchmark programs fit 32 lines.

**Mode: the line skeleton.** The skeleton of a program is the sequence of (kind, depth) of its non-blank
lines, cut at 16. U holds the 1,023 most frequent skeletons of a 200,000-function sample of the training
corpus, and "other". The reference mode is the reference's skeleton; pi_phi^h is the repository's
`ModeSelector` on s_x; at inference u is sampled (T = 1), as in Algorithm 1. Preflight reports the coverage of
the codebook; under 60%, depth is dropped from the skeleton before any training (a choice made on the training
corpus, not on a benchmark).

**Pass 1: f_theta.** The frozen backbone's states of the prompt at block 15, which the realizer's prefill
already computes (the paper's l_p = 15), through a trainable head: a linear map to width 512, two bidirectional
pre-norm blocks, a LayerNorm and a linear map to d. s_x in R^{P x d}. About 7M parameters. The prefill itself
stays frozen (Stage 0 rule 2).

**Target encoder f_theta_bar.** The EMA copy of that head (m from 0.996 to 1, Eq. 4), without gradient, on the
frozen backbone's block-15 states of the program read causally after its prompt (one more prefill, training
only). Slot m's target is the mean of its tokens' outputs plus a fixed sinusoidal code of the slot's start and
length (no parameters), so that layout is part of what g_phi has to predict. s_y in R^{M x d}.

**Pass 3: g_phi.** The repository's `JEPAPredictor` unchanged: slot queries Delta plus the mode embedding, four
blocks of self-attention over slots and cross-attention to s_x, five heads; Z = their mean, sigma_bar their
spread (Eq. 8).

**Loss.** The repository's `PlanLoss` unchanged: invariance per element on the real slots, VICReg's variance
hinge (gamma = 1, epsilon = 1e-4) and covariance term at 25 : 25 : 1, the mode's cross-entropy. Stable rank is
logged; the collapse guard logs only, as in the earlier runs.

**Bridge into the realizer.** Gated cross-attention in each of the 35 blocks, on canvas positions only, after
self-attention: h <- h + tanh(a_l) CA_l(norm(h), Z~), with a_l = 0 at the start, so the realizer is exactly
`runs/stage0-a` before training (tested). CA_l is Eq. 10 at attention width 256, with spectral norm on W_K
and W_V, the 1/gamma scaling, plan dropout 0.1 and the null token of `models/bridge.py`; its keys and values
are computed once per generation and reused by all K passes. About 32M parameters.

**Z~ = [Z ; e(u)] / gamma, without s_x.** The paper's Z~ also holds s_x, because its realizer had no other
view of the prompt. This one attends to the prompt's cached keys and values in every block. A second,
trainable view of the prompt would make "the plan's effect" include something that is not the plan. A row of
the ablation table restores s_x.

**Stage 2.** Starts from `runs/stage0-a/last.pt`. The planner is frozen, out of the optimizer, in eval mode and
fingerprinted, and the realizer accepts a `PredictedPlan` only (the repository's guards, kept). Training states,
data format, loss and optimizer are Stage 0's, so the plan is the only change. g_phi gets the reference mode
in training and a sampled one at inference (the repository's existing decision).

**Oracles.** Structure: Z_m = E_kind[kind_m] + E_depth[depth_m] + a linear map of the sinusoidal code of
(start_m, length_m), trained with the bridge; no backbone state of the reference enters. Latent: Z_m = a
linear map of the mean block-15 state of line m plus the same code.

**Depth.** Prefill with the head, selector, predictor, then K realizer passes: K + 3. The bridge's projections
are fused into pass 4. `realize`'s depth audit is extended to count them and to raise otherwise.

## What is measured in every evaluation

Exact kernel and confidence order at K = 8, 32, 128, and left to right at K = 256 (the plan must not cost
anything at one token per step). For the model of step 4 also: the null plan, a shuffled plan, and the
reference mode in place of the sampled one (a diagnostic: what the selector's errors cost). `analyze.py`'s
outcome and def-line tables for each.

One confound is known in advance: the model of step 4 has 10,000 more steps than `runs/stage0-a`. The null and
shuffled plans are step-matched, but a model trained to lean on a plan is weaker without it than one never
given a plan. Before any number of this stage goes into the paper as "the plan's effect", `runs/stage0-a` is
continued for the same steps without a bridge.

## Choices where the paper is silent (for the appendix)

1. Slots are lines, with their start and length in the target (layout is what breaks random order).
2. Modes are the 1,023 most frequent line skeletons and "other" (Proposition 1(b): only the mode reduces the
   factorization error, so it has to carry structure).
3. f_theta is a trainable head on the frozen backbone's block-15 states; the backbone is not trained.
4. Z~ has no s_x rows (above).
5. Cross-attention is gated and starts at zero; attention width 256; in all 35 blocks.
6. Stage 2 starts from the Stage 0 realizer and keeps its recipe.
7. Gates and thresholds as above, fixed on 21 Nov 2026.

## Deviations from the paper's protocol, recorded

- **No distillation.** Sec. 5.1 trains the Stage 2 realizer with "distillation from a 1,024-step teacher".
  Decided by the author on 21 Nov 2026: not now, an ablation later. It is the paper's own instrument against
  the collapse at small K, so a null result of this stage is a result about the plan without distillation.
- **No verification passes** (R = 0) and **no energies** (none for Python in the Stage 0 design).
- **No Stage 3.**
- **The argument count** (`docs/stage0.md`, second run) is not fixed here. It costs the realizer about 0.06 of
  rho_1 at one token per step and is left alone so that the plan is this stage's only change.

## What gets built

| module | contents | built |
|---|---|---|
| `text_vpa/stage1/slots.py` | line slots, kinds, depths, layout code, skeletons, the mode codebook | yes |
| `text_vpa/stage1/planner.py` | the head and its EMA copy on backbone states; wiring of `ModeSelector`, `JEPAPredictor`, `PlanLoss` | no (step 3) |
| `text_vpa/stage1/bridge.py` | gated cross-attention for `Stage0Realizer`, cached keys and values, null plan | yes |
| `text_vpa/stage1/train.py` | Stage 1 (planner), Stage 2 (realizer on predicted plans), the oracle diagnostics | the structure oracle only |
| `text_vpa/stage0/evaluate.py`, `report.py`, `analyze.py` | `--planner`, the plan variants, the gates of this document | for the oracle: `text_vpa/stage1/evaluate.py` and `report.py` (below) |
| `scripts/ncp/stage1.sh` | resumable pipeline; smoke mode; never shuts the server down | yes (step 1) |

Tests, each against a reference computed independently: slot and layout assignment against an explicit loop
over characters; skeleton and codebook; EMA pairing by name; the realizer with a closed gate equals the Stage 0
realizer token for token; cached keys and values equal recomputed ones; K + 3 depth; the Stage 2 trainer
rejects `OraclePlan` and `TargetOutput`; the planner's fingerprint is unchanged after Stage 2 steps; an oracle
run cannot write a report without its label.

## Step 1 as built (21 Nov 2026)

Code: `text_vpa/stage1/` (`slots.py`, `bridge.py`, `plan.py`, `oracle.py`, `data.py`, `train.py`, `evaluate.py`,
`report.py`, `preflight.py`). Tests: `tests/test_stage1.py`, `tests/test_stage1_cli.py`. Runner:
`scripts/ncp/stage1.sh`. Stage 0's files changed only where a plan has to pass through them (an optional `plan`
argument of `Stage0Realizer.canvas_hidden` and `sampler.realize`, hooks in `Stage0Trainer`); without a plan they
compute what they did, and Stage 0's tests are unchanged and pass.

    SMOKE=1 bash scripts/ncp/stage1.sh     # the pipeline in miniature: server, memory setting, speed
    bash scripts/ncp/stage1.sh             # preflight, 5,000 steps from runs/stage0-a/last.pt, evaluation, report
    bash scripts/ncp/stage1.sh status

| step of the script | output |
|---|---|
| preflight (`text_vpa.stage1.preflight`) | `RUN/preflight.json`: token offsets, line counts, indentation, MBPP's references as the oracle reads them, skeleton coverage; `RUN/skeleton_codebook.json` |
| training (`text_vpa.stage1.train --init runs/stage0-a/last.pt`) | `RUN/last.pt`, `RUN/metrics.jsonl`, `RUN/ORACLE.md` |
| evaluation (`text_vpa.stage1.evaluate`), MBPP validation, 5 samples | `RUN/sweep/mbpp/<tag>/`: `oracle-K{8,32,128}` (the gate's row is K = 8), `oracle-K256-left_to_right`, `oracle-K*-confidence`, `null-K*`, `null-K256-left_to_right` |
| report (`text_vpa.stage1.report --reference runs/stage0-a/sweep`) | `RUN/sweep/report.md`: the table, the gate after step 1, the delivery check |

**What is the same as Stage 0** (so that the plan is the only change): the trainer's loop, the training states
(the init run's `prefix_fraction` = 0.5), the loss, AdamW with Stage 0's schedule (500 warm-up steps, cosine from
2e-4 to 2e-5 at step 5,000), 32 examples per step, the prompts, the token draw, the truncation, the seeds. Step 0
is the Stage 0 checkpoint exactly: with closed gates the realizer with a plan equals the realizer without one bit
for bit (tested), so the evaluation at step 0 must repeat the last evaluation of `runs/stage0-a`
(`stage1.sh status` prints the two side by side).

### Choices made while building it (the design was silent or wrong)

1. **MBPP's references are normalised before the oracle reads them.** Of the 90 validation references, 30 are
   indented in multiples of four spaces; 13 use tabs, the rest one, two or three spaces; 55 have trailing blanks.
   Read as written, two thirds of the oracle's plans would carry wrong depths (indentation // 4) and a token
   layout of a formatting the realizer was not trained to write, and the "upper bound" would be too low. The
   oracle reads `ast.unparse` of the reference's syntax tree instead: the same program in four-space formatting.
   Checked on the benchmark file: all 90 parse, all 90 still pass their tests, none has more than 23 lines.
   Training plans are read from the corpus text as it is.
2. **The bridge's queries know their canvas position.** Q = H W_Q + P W_P, with P the fixed sinusoidal code of
   the canvas index (64 wide). Eq. 10 has no such term; the paper's realizer has learned positions. Gemma 4 has
   rotary attention only, so a hidden state does not hold its canvas index in a form a content query can match
   against a slot's (start, length), and a slot plan helps a position only if it can find its own slot.
   `query_position=false` is Eq. 10 as written.
3. **The layout code** of a slot is the sinusoidal code of start, of length and of end = start + length (wavelengths
   2 to 1,024 tokens): a linear map of the codes of start and length cannot form the end, and "position i lies in
   slot m" is start <= i < end.
4. **Kinds** are 27 lexical classes read from a line's own characters (20 statement keywords with `from` as an
   import; blank, comment, decorator, string, close, assign, call, other), so they exist for continuation lines
   and for broken samples; tested against Python's own tokenizer on standard-library modules. A tab counts as four
   columns of indentation.
5. **The oracle's plan has no mode row.** A skeleton is a function of the kinds and depths its slots already
   hold. The bridge has the e(u) row for the method (`num_modes` > 0, tested).
6. **Start of the run.** The realizer's adapters are `runs/stage0-a`'s and keep training; the bridge (32.8M
   parameters on E2B) and the oracle's embeddings are new and use the same learning rate; the optimizer is new;
   the corpus stream continues where `runs/stage0-a` stopped (functions its adapters have not seen).
7. **In the bridge**: a LayerNorm on the residual stream, 8 heads of 32, spectral norm on W_K and W_V only, keys
   and values in float32. It enters each block by wrapping two of its frozen norms (`bridge.py` says how); a test
   compares the result with a Gemma 4 block written out by hand.
8. **The oracle's rows have their own evaluator and report** (`text_vpa.stage1.evaluate`, `report`); Stage 0's
   `evaluate` and `train` refuse a `stage1-oracle` checkpoint, so an oracle number cannot appear under a Stage 0
   tag. MBPP only: the oracle reads a whole reference program.

### Reading the oracle (fixed on 21 Nov 2026, before any run)

The gate is the one above. Three checks decide whether its number can be read:

1. **Step 0 equals `runs/stage0-a`.** Held-out losses and the K = 8 probe at step 0 against that run's last
   evaluation. A difference is a bug; stop.
2. **Delivery.** At one token per step (`oracle-K256-left_to_right`) the samples must follow their plans: the
   mean share of lines that have the planned kind (`kind_match`; a missing or extra line counts as a miss) is at
   least 0.9. Parallel decisions play no part there, so a lower value means the structure did not reach the
   realizer (the bridge, or 5,000 steps at this learning rate), and then a ratio under 0.5 at the gate is **not**
   read as "structure is not enough": the two lower bands wait until that is repaired, and the report prints
   "NOT READ" in place of the band. A ratio of at least 0.5 needs no such check. The null plan's `kind_match` on
   the same row is printed next to it. The metric is strict by construction: lines are compared by index, so a
   sample that drops or adds one line early misses on every later line.
3. **The plan's effect is read within the same weights.** The oracle run has 5,000 more adapter steps than
   `runs/stage0-a`; `null-K*` (the null plan) is the comparison that holds the weights fixed, with the caveat
   this document already names: a model trained to lean on a plan is weaker without it.

While it trains, `plan_gain_k0` (null minus oracle loss with every token masked, nats per masked token) and the
probe's `kind_match` say whether the plan is being used; both should be clearly above zero by step 1,000 to
2,000. If they are still rising at step 5,000, the result is budget-limited, and the remedy is a longer run from
the start (a resumed run continues at the floor learning rate, which is too low for new modules).

### What step 1 cannot tell

- It is an upper bound on *structural* plans, and a loose one for the method: the oracle's layout (the number of
  tokens of every line) is more than a mode carries. A skeleton holds kinds and depths of at most 16 lines and no
  lengths; in the method the layout has to come out of g_phi, a deterministic prediction given (c, u). A ratio of
  at least 0.5 here says the planner is worth building, not that it will pass.
- One seed, 90 problems: at the gate's thresholds the interval will be wide (Stage 0's rho_1 was 0.69 with
  [0.45, 0.97]).
- Not verified off the server: the real tokenizer's character offsets (the preflight checks them against decoded
  prefixes and stops the run if they disagree), memory and speed with the bridge (the smoke run measures both).

### Smoke run on the L4 (21 Nov 2026, `runs/stage1-oracle-smoke`, commit `a75cae2`)

30 steps from `runs/stage0-a/last.pt`, 8 MBPP validation problems x 2 samples. A check of the server, not a result.

- Unit tests and the preflight pass on the server.
- **Memory:** micro-batch 8 without activation checkpointing peaks at 22.2 of 23.5 GiB (Stage 0: 21.1), over the
  92% limit, so the trainer switched checkpointing on (11.6 GiB). The math does not change.
- **Speed: 0.152 steps/s** (Stage 0: 0.202), the cost of recomputing each block in the backward pass. 5,000 steps
  take about 9.1 h, plus about 1 h of evaluations during training, then about 4 h of evaluation (11 configurations;
  the gate's three rows and the delivery row come first, about 1.5 h). About 14 h in all, not the 10 h estimated
  above.
- After 30 steps the gates are at 0.001 (max 0.003; under Adam a parameter moves at most about the learning rate
  per step), so the oracle and null rows are identical and `plan_gain_k0` is 0.000, as expected.
- **The delivery metric's floor:** without a working plan, `kind_match` at one token per step is 0.140 (oracle and
  null alike; 16 samples). At K = 8 it is 0.26, higher because long garbage canvases match the most frequent kind,
  `assign`, by chance.
- The step-0 comparison with `runs/stage0-a` is not possible in a smoke run (32 held-out functions and 1 probe
  sample instead of 256 and 5); the full run makes it.
- Fixed after the smoke run: its report printed "clearly under 0.1: run step 2" above a delivery check that had
  not passed. The report now prints "NOT READ" there (`report.gate_reading`), and the gate line names the number of
  problems and samples. `stage1.sh status` now prints the preflight's numbers and says when the step-0 comparison
  is not possible.

## Step 1: result (`runs/stage1-oracle`, 21 to 22 Nov 2026; oracle: not the method)

5,000 steps from `runs/stage0-a/last.pt` (adapters trained too), MBPP validation, 90 problems x 5 samples.

| Pass@1 (parses) | K = 8 | K = 32 | K = 128 | one token per step |
|---|---|---|---|---|
| `runs/stage0-a`, no plan, exact kernel / left to right | 0.4% | 1.1% | 0.7% | 17.1% (98.2%) |
| same weights as the oracle, null plan, exact kernel / left to right | 0.0% (5.3%) | 0.2% (11.3%) | 0.0% (14.7%) | 13.8% (97.6%) |
| structure oracle, exact kernel / left to right | 0.0% (12.4%) | 0.2% (14.7%) | 0.4% (22.2%) | 7.3% (77.1%) |
| oracle - null, 95% paired bootstrap, points | +0.0 [+0.0, +0.0] | +0.0 [-0.7, +0.7] | +0.4 [+0.0, +1.1] | **-6.4 [-12.9, -0.2]** |
| `runs/stage0-a`, no plan, confidence order | 0.0% | 2.0% | 9.8% | |
| structure oracle, confidence order | 0.0% (0.2%) | 0.0% (5.6%) | 0.2% (27.6%) | |
| kind match: structure oracle / null plan, exact kernel and left to right | 0.719 / 0.230 | 0.760 / 0.278 | 0.775 / 0.297 | 0.791 / 0.276 |

- **Step 0 equalled `runs/stage0-a`** exactly (val_loss 2.7907, val_loss_next 0.9332, probe 0.44%).
- **The plan was used heavily.** Held-out loss with every token masked, oracle / null / another function's plan, nats
  per token: step 0 4.134 / 4.134 / 4.134; step 5,000 2.984 / 4.166 / 12.739 (plan gain 0.33, 0.71, 0.90, 1.09,
  1.18 at steps 1,000 to 5,000).
- **Following the structure did not make samples pass.** At K = 8, 188 of 450 samples had the planned kind on at
  least 90% of their lines; none passed and 20% parsed (inside a line: brackets opened and not closed, one variable
  under two names in two lines, wrong parameter counts). At one token per step the best-following samples passed
  11.0%, below the realizer without a plan (17.1%): the samples copy the reference's shape and fill it with their
  own, wrong content (MBPP/511: the reference's ten lines, kind for kind, with another loop start).
- The expectation stated above ("lift the exact kernel clearly at K = 32 and K = 128") did not hold.
- **Delivery check: 0.791 < 0.9, so the report printed NOT READ.** Author's decision (22 Nov 2026): the gate is read
  as **clearly under 0.1** anyway. The check was meant to catch a structure that did not reach the realizer; the
  kind-match buckets show that where it fully arrived it was not enough, and a longer run would not change that.
  By the gate: step 2.
- **Within the same weights the plan changes nothing at parallel K and costs Pass@1 at one token per step:** -6.4
  points [-12.9, -0.2] (parses 97.6% to 77.1%). A layout the realizer trusts (another function's plan: 12.7 nats)
  pushes it into the reference's form, which it then fills with its own content.
- **The extra training cost a little without a plan, within noise:** null plan 13.8% against `runs/stage0-a`'s
  17.1% at one token per step, difference -3.3 points [-8.7, +1.8] (95% paired bootstrap over the 90 problems). It
  includes the known confound: a model trained to lean on a plan is weaker without one.
- **Confidence order collapses with the plan:** 0.0 / 0.0 / 0.2% at K = 8 / 32 / 128, against `runs/stage0-a`'s
  0.0 / 2.0 / 9.8% (no null-plan row in this order). A likely explanation, not tested: with the layout known the
  realizer is confident about structural tokens all over the canvas, so confidence order stops growing a prefix.
- Speed 0.150 steps/s with activation checkpointing (11.6 GiB); the whole run, tests to report, took about 13.5 h.

**What this implies for steps 3 and 4 as designed** (an argument from the measurements, not a measurement). A line
skeleton, the mode of step 3, is a function of the structure oracle's plan (its kinds and depths, without the
layout). The predicted plan g_phi(s_x, u, Delta) is a function of the prompt and the mode, so beyond what the
realizer already has from the prompt it can carry at most what the mode carries (Proposition 1(a)), at most ln 1024
= 6.9 nats per program. The structure oracle gave the realizer all of that and more: 1.18 nats per masked token at
the all-masked state, well over an order of magnitude more per program. Within the same weights it changed nothing
at K = 8 (+0.0 [+0.0, +0.0]) and cost Pass@1 at one token per step. So the planner of steps 3 and 4, as designed,
cannot meet H2 with this realizer, whatever step 2 shows. The argument assumes that a plan holding less about the
program does not do better at K = 8 than one holding more, which is what the K = 8 rows show between null and
oracle. Step 2 asks about a different route.

## Step 2 as built (22 Nov 2026): the latent oracle

The realizer is told what each line of the reference *says*:

    Z_m = [len_m > 0] W_s LN( mean over line m of H_k ) + W code(start_m, length_m)

H_k is the frozen backbone's hidden state after reference token k, read causally after the problem's prompt
through blocks 0 to 14 (`latent_layers` = 15: the KV-storing blocks, the paper's l_p = 15, what the prefill computes
for the prompt; `Stage0Realizer.frozen_states`, tested against the released model's own hidden states). LN has its
own scale and shift (the residual stream's dimensions differ by orders of magnitude, and the bridge's keys are
Lipschitz in Z). A line without a token has its layout code only. The bridge, the training loop, the recipe and the
evaluation are step 1's; only the plan changes. In training the reference is the example's own program; at
evaluation the normalised MBPP reference, after the same prompt the realizer sees.

What it decides: whether *any* slot plan (32 rows of width 256 about the reference) can drive this realizer at K = 8.
It carries more than any mode the method could sample, so it is an upper bound for plans, not a candidate method.
After step 1 (above) the planner of steps 3 and 4 is answered, so step 2's question is the route the paper names
for modes that cannot be enumerated (Sec. 2): a stochastic latent predictor that samples the content of the plan.
The latent oracle is close to conditioning the realizer on the paper's own JEPA target (slot-pooled states of y),
which the anti-leakage rule forbids for the method: the best case for the paper's plan representation.

**Gate after step 2** (written 22 Nov 2026, before it ran; `report.step2_gate`), the same rho on its `oracle-K8`:

- at least 0.5: a slot plan can drive this realizer at K = 8. A plan route stays open only for a plan that carries
  content: the paper's stochastic latent predictor (skeleton modes cover under half the programs, preflight above,
  and a deterministic g_phi(c) cannot lower the factorization error, Proposition 1(a)).
- between 0.1 and 0.5: even the reference's content does not bring this realizer to H2 at K = 8; the plan route
  cannot meet H2 here. What content buys at K = 32 and K = 128 is reported.
- under 0.1: the plan route is closed for this realizer. The choice is between the paper's distillation (Sec. 5.1)
  and more depth.

**Reading rule** (replaces step 1's delivery check; revised on 22 Nov 2026 before the run, `report.latent_reading`).
A band under 0.5 is read only if the content reached the realizer on MBPP itself: **at one token per step, at least
half of the samples reproduce the reference program** (the same syntax tree after normalisation). Then the realizer,
given the plan, can write the program out, and a ratio under 0.5 is about parallel decisions, not about missing
content. Otherwise NOT READ. A band of at least 0.5 needs no check.

The first version (all-masked held-out loss below the structure oracle's 2.984) is withdrawn: any plan that carries
content passes it, so it could not support "the plan route is closed", the conclusion the gate draws from a band
under 0.1.

Reported without a threshold: K = 8 Pass@1 on the problems where the content arrived (at least half of their
one-token-per-step samples reproduce the reference), against the baseline on the same problems (the check that
settled step 1); the share of samples reproducing the reference and the kind match at each K; the all-masked
held-out loss against the structure oracle's and against the plan-free realizer's next-token loss with the full left
context (this run's step 0, 0.933; the all-masked mean includes the <eos> filling at weight 0.1).

The gate and this reading rule were **confirmed by the author on 22 Nov 2026, before the run.**

**Realizer adapters** (`TRAIN_ADAPTERS`): 1 trains them as in step 1 (one change between the two steps, and the plan
gets the best chance: an upper bound). 0 keeps them at `runs/stage0-a`'s values, so that "no plan" is exactly the
Stage 0 realizer. Rule fixed now: keep 1, unless the step-1 null-plan row at one token per step is below
`runs/stage0-a`'s 17.1% with its 95% paired-bootstrap interval of the difference entirely below zero (the extra
training itself degraded the realizer); then 0. **Applied at launch:** -3.3 points [-8.7, +1.8] on 90 problems; the
interval includes zero, so `TRAIN_ADAPTERS` = 1.

    SMOKE=1 PLAN=oracle-latent bash scripts/ncp/stage1.sh        # runs/stage1-latent-smoke: memory and speed
    PLAN=oracle-latent bash scripts/ncp/stage1.sh                # runs/stage1-latent; TRAIN_ADAPTERS=0 to freeze
    RUN=runs/stage1-latent bash scripts/ncp/stage1.sh status

Expected cost (measured by the run itself; the smoke run is optional, since the full run reaches every new code
path at step 0): one more forward pass through 15 of 35 blocks per micro-batch, without gradient, about 10% of a
training step; 5,000 steps about 10.3 h, with evaluation about 15 to 16 h.

## Step 2: result (`runs/stage1-latent`, 22 to 23 Nov 2026; oracle: not the method)

5,000 steps from `runs/stage0-a/last.pt` (adapters trained, `TRAIN_ADAPTERS` = 1), 0.142 steps/s with activation
checkpointing; launched 22 Nov about 14:55 KST, report written 23 Nov 05:22 KST. MBPP validation, 90 problems x 5
samples.

| Pass@1 (parses) | K = 8 | K = 32 | K = 128 | one token per step |
|---|---|---|---|---|
| `runs/stage0-a`, no plan, exact kernel / left to right | 0.4% | 1.1% | 0.7% | 17.1% (98.2%) |
| same weights as the oracle, null plan | 0.2% (6.9%) | 0.2% (10.7%) | 0.4% (12.7%) | 14.2% (96.7%) |
| latent oracle | 1.8% (17.6%) | 2.2% (22.9%) | 2.2% (26.4%) | 6.4% (51.1%) |
| oracle - null, 95% paired bootstrap, points | +1.6 [+0.0, +3.3] | +2.0 [+0.7, +3.8] | +1.8 [+0.0, +4.0] | **-7.8 [-14.7, -1.1]** |
| `runs/stage0-a`, no plan, confidence order | 0.0% | 2.0% | 9.8% | |
| latent oracle, confidence order | 0.0% (3.8%) | 1.1% (23.6%) | 2.0% (50.7%) | |
| kind match: oracle / null | 0.795 / 0.229 | 0.818 / 0.262 | 0.821 / 0.271 | 0.814 / 0.286 |
| the reference's line count: oracle / null | 70.0% / 9.6% | 72.4% / 10.9% | 72.4% / 10.9% | 68.7% / 12.0% |
| samples reproducing the reference | 0.0% | 0.0% | 0.0% | 0.0% |

- **Step 0 equalled `runs/stage0-a`** exactly (2.7907 / 0.9332 / 0.44%).
- **Gate: rho = 0.075 [0.018, 0.144], NOT READ** by the reading rule: no sample reproduces the reference at one
  token per step (the rule needs half), so there is no problem on which the content arrived.
- **Held-out loss with every token masked**, oracle / null / another function's plan, nats per token at step
  5,000: 2.194 / 4.177 / 15.109 (structure oracle: 2.984 / 4.166 / 12.739). Plan gain (null - oracle) at k = 0, 2,
  4, 6 of K = 8 (0, 8, 29 and 62% of the canvas revealed): +1.982, +1.373, +0.558, +0.116. Next-token loss with the
  plan and the full left context: 0.702, accuracy 0.852 (step 0: 0.933, 0.797).
- **Probe at K = 8** with the oracle plan: 0.4, 0.0, 0.4, 0.7, 0.9, 1.8% at steps 0 to 5,000; kind match 0.24 to
  0.80; reproduced 0.0% throughout. Gates: mean |tanh(a_l)| 0.073, largest 0.25.

Reading (Claude's, 23 Nov 2026):

1. **By the rule: not read.** The latent plan did not deliver the content. Reproducing a ~70-token program exactly
   at one token per step needs the reference token on top at about 99% of positions; with the plan it is 85%
   (held-out corpus, full left context: 0.852^70 is about 1e-5). So step 2 does not show that content cannot help
   at K = 8, and neither of the gate's conclusions is drawn ("the plan route is closed", "even the content does
   not reach H2").
2. **What both oracles show.** The realizer uses a plan heavily and trusts it: another function's plan costs 8.6
   (structure) and 10.9 (latent) nats per token above the null plan. It takes the plan's form (kind match 0.8, the
   reference's line count in about 70% of samples against 10 to 12% without) and fills it with its own content. At
   one token per step that costs 6 to 8 points (-6.4 and -7.8, both intervals below zero); at K = 8 to 128 the gain
   is at most +2 points; with a plan the confidence order collapses. The plan's information is used where little is
   revealed (1.98 nats per token with everything masked, 0.12 with 62% revealed).
3. **A train-test gap of the kind Sec. 5.1 names.** In training the plan always describes the program being
   written; in sampling the realizer's own tokens soon depart from the plan's reference, and the plan's layout
   (absolute start and length codes) then pulls it into the reference's form. Sec. 5.1 rejects conditioning on the
   target encoder's view because it "opens a train-test gap of the same kind as exposure bias"; both oracles
   condition on the reference and show that gap. A plan the method could use (predicted or sampled) disagrees with
   the realizer's samples more, so a realizer meant to use one has to be trained on such plans, as the paper's
   Stage 2 is, not on oracles. The gap also makes the oracles a weaker upper bound than intended.
4. **Without distillation.** Every plan of steps 1 and 2 was tested on a realizer trained without the paper's
   distillation (recorded above as a deviation). Sec. 5.1 combines the plan with distillation in Stage 2.

**Next (the author's decision, open on 23 Nov 2026).** The options written before the run for NOT READ: a longer or
faster-learning bridge run (a learning-rate multiplier for the new modules, not built), or a decision between
distillation and more depth on step 1's result and the steps 3 and 4 argument. Claude's recommendation: stop the
oracles. A plan that did carry the content would let the realizer nearly copy the program at K = 8, which says
nothing about whether a planner can sample such content from the prompt; the measured gap says an imperfect content
plan hurts unless the realizer is trained on imperfect plans. Distillation is the paper's own instrument for the
parallel step and is untested here.

**Decision (the author's, 23 Nov 2026).** The oracles stop, and the implication for steps 3 and 4 is accepted:
the planner as designed cannot meet H2 with this realizer. Text-VPA's H2 stays as it was tested (K = 8 on a whole
canvas) and failed. The work continues as Chat-VPA (semi-autoregressive block discrete flow matching with
autoregressive trajectory distillation; preprint draft of 22 Nov 2026) in a new repository,
https://github.com/ylee-sling/chat-VPA, initialized from this repository's Stage 0 code at this commit's parent
(`ff747c9`). Its first experiment, E0, runs on this harness: the same corpus, prompts, draw and MBPP validation
split as `runs/stage0-a`.

One correction for the Chat-VPA draft, measured on this repository's sampler: at K = 128 the left-to-right order
reveals one token per pass for the first 42 passes and two afterwards (cosine schedule, N = 256), so over a
typical program of about 48 tokens it decides one token per pass, not two. At K = 32 it reveals 1, 1, 2, 2, 3, 3,
4, 5, 5, 6, 6, 7, 7 and 7 tokens over the first 59 positions; at K = 8, 5, 15, 24, 32, 39, 44, 48 and 49.
