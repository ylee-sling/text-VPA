# Text-VPA Stage 1: a planner on the Stage 0 realizer (design)

Written on 21 Nov 2026, after the second Stage 0 run. It fixes the order of experiments, their gates and the open
choices before any of them exists, in the way `docs/stage0.md` did for Stage 0.

**Status (21 Nov 2026, evening).** The author confirmed the design's three open choices: slots are lines, modes
are line skeletons, the bridge has no s_x rows. Step 1, the structure oracle, is built (branch `stage1-planner`;
"Step 1 as built" at the end of this document). **Nothing has been run on the real model**, and steps 2 to 4 are
not built. The gates below are unchanged.

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
   read as "structure is not enough": the two lower bands wait until that is repaired. A ratio of at least 0.5
   needs no such check. The null plan's `kind_match` on the same row is printed next to it.
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
