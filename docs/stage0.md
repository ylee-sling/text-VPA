# Text-VPA Stage 0: a plan-free Gemma 4 E2B realizer against its own autoregressive decoding

Stage 0 asks one question before any planner exists: can a pretrained autoregressive backbone be turned into a
few-step parallel realizer at all? Gemma 4 E2B is adapted into the Text-VPA realizer without a plan, generates
at K = 2, 4 and 8 parallel steps, and is compared with the same checkpoint's own left-to-right decoding in the
repository's Pass@k harness on HumanEval and MBPP.

**H2.** At K = 8 the realizer reaches at least half of the autoregressive backbone's Pass@1.

Code: `text_vpa/stage0/`. Tests: `tests/test_stage0.py`, `tests/test_stage0_cli.py`. Runner:
`scripts/ncp/stage0.sh`.

## What runs

| Step | Command (inside `scripts/ncp/stage0.sh`) | Output |
|---|---|---|
| preflight | `python -m text_vpa.stage0.preflight` | `RUN/preflight.json`: the paper's E2B facts against `config.json`, tokenizer, canvas coverage, corpus yield |
| harness check | `python -m text_vpa.stage0.evaluate --decoder reference` | canonical solutions through Stage 0's truncation: 164/164 and 499/500 |
| baseline | `python -m text_vpa.stage0.evaluate --decoder ar` | `RUN/eval/<benchmark>/ar/` |
| training | `python -m text_vpa.stage0.train` | `RUN/last.pt`, `RUN/step-*.pt`, `RUN/metrics.jsonl` |
| realizer | `python -m text_vpa.stage0.evaluate --decoder realizer --k-steps 2 4 8` | `RUN/eval/<benchmark>/realizer-K{2,4,8}[-confidence]/` |
| verdict | `python -m text_vpa.stage0.report` | `RUN/eval/report.md`, `report.json` |
| reference row | `python -m text_vpa.stage0.evaluate --decoder ar --chat` | `RUN/eval/<benchmark>/ar-chat/` (instruction-tuned checkpoint; not part of H2) |

## The realizer

One sample is laid out as

    prefix  c_1 .. c_{P-1}        frozen backbone, causal, cached once
    canvas  a = c_P, w^1 .. w^N   realizer, refined in parallel

- **Prompt cache.** The frozen backbone reads the prompt prefix exactly as released; only the 15 blocks that
  store keys and values run (the upper 20 reuse them). No adapter acts on prompt positions and prompt rows never
  see the canvas, so the cache is the one any other role of the backbone computes. Tested: prefill plus a canvas
  pass under causal masks reproduces `Gemma4ForCausalLM` to 1e-5, and the cache is unchanged by the adapters.
- **AR shift, clamped anchor.** The output at canvas index j - 1 is the posterior of w^j. The anchor a is the last
  prompt token, held fixed at canvas index 0.
- **Bidirectional canvas.** Canvas rows attend to every valid prompt position and every canvas position; local
  blocks keep positions at distance < 512 on either side (on the left, the backbone's own window).
- **Canvas-only LoRA.** Rank 64 on the attention and MLP projections of all 35 blocks; B = 0 at initialisation.
- **Mask source.** [MASK] positions take one trainable vector instead of a token embedding.
- **(t, h) modulation.** y (1 + scale) + shift on the outputs of the two pre-norms of each block, from a
  zero-initialised projection of c = MLP_t(sin t) + MLP_h(sin h) plus zero per-block tables.
- Not in Stage 0: the gated cross-attention to the plan, energies, repair, verification passes (R = 0).

At initialisation the realizer is the backbone with a bidirectional canvas (tested: adapters on = adapters off).

## Decisions where the paper is silent

Each line: the choice, the reason, and the alternative that was not taken.

1. **Checkpoint: `google/gemma-4-E2B` (pre-trained), not `-it`.** The corpus and both prompt formats are raw code
   continuation, which is what the pre-trained checkpoint was trained on, and the baseline can then use the very
   same prompt as the realizer. The instruction-tuned checkpoint is measured through its chat template as a
   reference row, so a weak baseline cannot pass unnoticed. Alternative: adapt `-it` and prompt both decoders
   through the chat template.
2. **Prompt formats.** HumanEval: the prompt as is, the canvas is the body (`complete`). MBPP: the task text and its
   first test inside a module docstring followed by a blank line, the canvas is the whole program (`write`). The
   same strings go to both decoders. The blank line is a measured fix: with the BigCode harness's zero-shot format
   (one newline after the docstring) Gemma 4 E2B wrote MBPP functions without indentation and none of the smoke
   run's baseline programs parsed; with the blank line its greedy output on MBPP/11 is indented and correct, as it
   is with the 3-shot format of Austin et al. (700 prompt tokens instead of 39, and unlike the zero-shot training
   data).
3. **Canvas length N = 256, fixed.** Program tokens, one `<eos>`, then `<eos>` up to N. With the Gemma 3
   SentencePiece vocabulary (262,144 tokens; the preflight repeats the count with the Gemma 4 tokenizer) 99.4% of
   HumanEval's canonical bodies and 98.8% of MBPP's programs fit. Training examples that do not fit are dropped,
   not truncated. The autoregressive baseline gets the same 256-token budget. Alternative: a length head or
   length buckets (the PR-JEPA code took N from the plan's readout; Stage 0 has no plan).
4. **Anchor.** The last prompt token moves into the canvas as the clamped anchor; the prompt cache covers the
   prompt without it. Alternative: a dedicated anchor token.
5. **[MASK].** The tokenizer's `<mask>` id is used for the per-layer embedding lookup; the main input embedding at
   masked positions is a separate trainable vector initialised to `<mask>`'s embedding, because the embedding
   matrix is tied to the output head and stays frozen.
6. **LoRA: rank 64, alpha 64 (scale 1), no dropout,** on q, k, v, o and the three MLP projections. The upper 20
   blocks have no k and v projections (they share the lower blocks' keys and values), so LoRA acts there on q, o
   and the MLP only. About 98.7M trainable parameters with the modulation (4% of the 2.3B effective). Alternative:
   rank 16 on attention only (DiffuLLaMA-style), or full fine-tuning (does not fit canvas-only training of a
   shared backbone).
7. **Modulation: "adaLN-single", zero-initialised,** as in the PR-JEPA realizer, applied to the frozen blocks'
   two pre-norms. Feature width 256.
8. **Schedule: cosine, kappa_t = 1 - cos(pi t / 2).** With the mask source the exact kernel reveals a fraction
   kappa_{t+h} - kappa_t of the canvas per step: at K = 8, 1.9, 5.7, 9.2, 12.4, 15.1, 17.3, 18.8 and 19.5%. Few
   positions are committed while the context is weakest. The preprint's default is linear (12.5% per step);
   changing it needs retraining, because training states follow the schedule.
9. **Training states.** K drawn uniformly from {2, 4, 8}, t = k/K on K's grid (the states the sampler evaluates,
   as in the PR-JEPA code), each canvas position kept with probability kappa_t, else [MASK].
10. **Loss.** Cross-entropy of the clean token at masked positions, read through the AR shift; weight 1 on program
    tokens and the first `<eos>`, 0.1 on the `<eos>` filling the canvas (about two thirds of a canvas is filling,
    and it is trivially predictable once the length is known); token-weighted mean per micro-batch. No
    kappa_dot / (1 - kappa) ELBO weight. Alternative: weight 1 everywhere (LLaDA, Dream).
11. **Optimisation.** AdamW (0.9, 0.999), no weight decay, learning rate 2e-4 with 500 warm-up steps and cosine
    decay to 2e-5 at step 25,000, gradient clipping at 1.0, bfloat16 autocast, 32 examples per step (8 x 4).
12. **Budget: 25,000 steps = 800K examples = 0.2B canvas tokens** (0.4 of one pass over the 2M-function corpus),
    estimated at 14 to 28 hours on the L4. This is what about a day on the server buys, and it is two to three
    orders of magnitude below published adaptations (DiffuLLaMA: under 200B tokens; DiffuCoder: 65B + 65B). The
    learning curve below exists for that reason.
13. **Data mix.** Every corpus row gives a `complete` example; a documented top-level function gives a `write`
    example instead with probability 0.5 (about 11% of all examples: 71% of the corpus are methods, which have no
    description-only form; a higher probability would take the most HumanEval-like functions away from
    `complete`): description = its docstring plus the line `assert name(...) == ...`, canvas = imports + the
    function without its docstring. MBPP names the function in an assert and the corpus has no tests, so the line
    carries the name and nothing else. In the first run it listed the parameter names
    (`assert name(a, b) == ...`), and the realizer learned to copy an assert's arguments into its `def` line
    ("A flaw in the first run's training data" below). Alternatives not taken: literal arguments made up from
    annotations and defaults (nothing says what values a function accepts), asserts taken from doctests (few
    docstrings have them). Prompts over 512 tokens are dropped.
14. **Sampler.** Primary, and the one H2 is decided on: the exact jump kernel on the mask source, i.e. each masked
    position is revealed with probability pi_{t,h}, positions chosen at random. Secondary, reported next to it:
    the same number of positions per step, most confident first (MaskGIT). A third order for diagnostics only,
    `left_to_right`: the same number of positions per step, leftmost first, stopping once a row's `<eos>` is
    revealed (which changes no output). Token draw for both decoders: temperature 0.2, top-k 1,024,
    top-p 0.95; [MASK], `<pad>` and `<bos>` are never drawn.
15. **Baseline.** The released model's own `generate` (`Gemma4ForConditionalGeneration`, KV cache): same prompt,
    same 256-token budget, same draw, same truncation. It stops early once the truncation rule has found its
    stop, which changes no program.
16. **Truncation, one rule for both decoders.** `complete`: keep lines up to the first non-blank line starting in
    column 0. `write`: keep lines up to the first line starting with `assert `, `print(`, `if __name__`, `"""`,
    `'''`, a code fence or `<`. The canonical solutions pass through it unchanged in effect (164/164, 499/500).
17. **Protocol.** 10 samples per problem, unbiased Pass@1 and Pass@10. H2 holds if
    Pass@1(realizer, K = 8) / Pass@1(AR) >= 0.5 on both benchmarks, fails if it is below on both, and is "mixed"
    otherwise; "clearly" is added when every 95% paired-bootstrap interval (over problems) lies on the same side
    of 0.5. The final checkpoint is evaluated; nothing is selected on a test set. One training seed: the
    interval covers the choice of problems, not the variance between training runs.
18. **Learning curve.** Every 2,500 steps: masked-token loss on held-out corpus functions and Pass@1 on MBPP's
    validation split (90 problems, 5 samples, K = 8). The report says whether the curve was still rising when
    training stopped; a failure with a rising curve is a statement about the budget.
19. **Depth.** Realizer: one prefill and K canvas passes, K + 1. Baseline: one prefill and one pass per generated
    token. (With a planner the paper's count is K + R + 3.)
20. **Memory.** Before the first step a worst-case micro-batch is run; if it does not fit, activation
    checkpointing is switched on, then the micro-batch is halved and the accumulation doubled. The result is in
    `RUN/memory.json`.
21. **Prefix-shaped states (second run, `prefix_fraction` = 0.5; 0 in the first run).** Half of the training
    states are drawn differently: the first masked position L is uniform over the program's positions and its
    first `<eos>`, everything before L is clean and everything from L on is [MASK]. The loss is unchanged
    (masked-token cross-entropy on every masked position, weight 0.1 on the `<eos>` filling). Time input: K from
    {2, 4, 8}, t the point of K's grid whose kappa_t is nearest L / N, h = 1/K, i.e. the pair the sampler feeds
    when that fraction of the canvas is revealed. Reason: these are the states `left_to_right` decodes from, and
    uniformly random masks practically never produce them (at kappa_t = 0.8, 20 clean tokens in a row have
    probability about 1%), so the first run was never trained on the prediction that sequential decoding asks
    for. Half and half gives the exact kernel's states and the sequential ones the same share. Alternatives not
    taken: a loss on the next token only (it trains one token per step and nothing of what K <= 128 needs),
    extra weight near L (one more hyperparameter; to be considered if the gate is missed), block-causal
    attention (direction B, which changes the claim about depth).
22. **The one-token-per-step gate (fixed on 20 Nov 2026, before the second run).** On MBPP's validation split,
    5 samples: rho_1 = Pass@1(trained realizer, `left_to_right`, K = N = 256) / Pass@1(code check). Passes at
    rho_1 >= 0.8, fails at rho_1 <= 0.5, partial in between; "clearly" when the 95% paired-bootstrap interval
    lies on the same side. At one token per step nothing is decided in parallel, so the ratio is the cost of the
    adaptation alone. 0.5 is below the estimate for the first run without its data flaw (0.58), so a fail means
    prefix-shaped states added nothing. It is read with the def-line table (bad and copied `def` lines at the
    baseline's 0%). The held-out next-token loss (`val_loss_next`) is a training check only: the first run's
    realizer was below its backbone on it while keeping 36% of its Pass@1 (see "The second run"). Rows at
    K <= 128 are read after the gate.

## Gemma 4 E2B: what was checked on 17 Nov 2026

From the model page and `config.json` of `google/gemma-4-E2B`:

- Hub ids `google/gemma-4-E2B` (pre-trained) and `google/gemma-4-E2B-it`; Apache 2.0; the page shows no access
  gate (the runner still explains what to do if the Hub refuses the download).
- 2.3B effective parameters, 5.1B with embeddings; 35 blocks of width 1,536; 8 query heads and 1 KV head, head
  size 256 in local and 512 in global blocks; 28 local blocks with a 512-token window and 7 global ones, every
  fifth block, the last block global; `num_kv_shared_layers` = 20; tied 262,144-token vocabulary; per-layer
  embeddings of width 256; final logit soft cap 30; 128K context.
- Not in the paper's notes: `use_double_wide_mlp` is true, so the MLPs of the 20 KV-sharing blocks are 12,288
  wide (6,144 below). This matters for the cost model's FLOP counts.
- The page reports LiveCodeBench v6 44.0% and no HumanEval or MBPP figure, and recommends temperature 1.0,
  top-p 0.95, top-k 64 for chat use. Pass@1 here uses temperature 0.2, the usual setting for code.

`preflight.py` compares these facts with the downloaded `config.json` and stops on a difference.

Related work that appeared after the preprint: Google's DiffusionGemma (10 June 2026) is a block-diffusion model
on Gemma 4's 26B MoE backbone with a causal encoder that fills a KV cache and a bidirectional decoder that
denoises a 256-token canvas in up to 48 steps under uniform-state diffusion. It is the closest published design
to the Text-VPA realizer and belongs in Sec. 2.

## Result of the first run (17 to 19 Nov 2026, `runs/stage0` on the NCP L4)

25,000 steps (about 35 h), final checkpoint, 10 samples per problem, temperature 0.2.

| benchmark | decoder | Pass@1 | Pass@10 | parses | depth |
|---|---|---|---|---|---|
| HumanEval | autoregressive (pre-trained) | 23.8% | 32.3% | 93.5% | 99 |
| HumanEval | realizer K = 2 / 4 / 8, exact kernel | 0.7 / 1.0 / 1.2% | 1.2 / 3.0 / 4.9% | 8.1 / 10.1 / 17.3% | 3 / 5 / 9 |
| HumanEval | realizer K = 2 / 4 / 8, confidence | 0.7 / 0.9 / 1.0% | 1.2 / 1.2 / 1.2% | 7.3 / 7.5 / 7.3% | 3 / 5 / 9 |
| HumanEval | instruction-tuned, chat (reference) | 74.4% | 82.3% | 89.7% | 343 |
| MBPP | autoregressive (pre-trained) | 26.7% | 43.8% | 90.4% | 116 |
| MBPP | realizer K = 2 / 4 / 8, exact kernel | 0.0 / 0.1 / 0.2% | 0.0 / 0.8 / 1.2% | 0.7 / 3.0 / 7.3% | 3 / 5 / 9 |
| MBPP | realizer K = 2 / 4 / 8, confidence | 0.0 / 0.0 / 0.0% | 0.0 / 0.0 / 0.0% | 0.5 / 0.7 / 1.5% | 3 / 5 / 9 |
| MBPP | instruction-tuned, chat (reference) | 51.1% | 58.2% | 82.3% | 352 |

**H2 clearly fails.** Pass@1 ratio of the realizer at K = 8 (exact kernel) to the autoregressive backbone:
HumanEval 0.051 [0.011, 0.104], MBPP 0.006 [0.002, 0.012] (95% paired bootstrap over problems); threshold 0.5.
The probe (MBPP validation, K = 8) stayed between 0.0% and 0.7% from step 5,000 to 25,000, and the held-out loss
was still falling slowly at every state (all-masked 4.26 -> 4.08 nats from step 2,500 to 17,500; mostly revealed
0.72 -> 0.53). The instruction-tuned row used its chat template and a 768-token budget; it is not part of H2.

**The realizer's MBPP rows are affected by a flaw in this run's training data** (found on 20 Nov, described
below): 57% of its MBPP samples at K = 8 have a `def` line that is not Python. The verdict does not depend on
it. HumanEval's prompts contain the signature and no assert, so its rows are untouched, and there the ratio is
0.051. On MBPP, the samples at K = 8 whose `def` line is valid pass at about 0.5%, a ratio of about 0.02.

## Diagnostic sweep (20 Nov 2026, `runs/stage0/sweep`)

MBPP validation (90 problems, 5 samples each), the final checkpoint; outside the protocol, no verdict.

| decoder | Pass@1 | parses | mean depth |
|---|---|---|---|
| baseline, the released model's `generate` | 23.8% | 87.1% | 120 |
| backbone through the realizer's code path (adapters off, causal, one token per step) | 24.7% | 89.8% | 118 |
| realizer, left to right, K = 256 (one token per step) | 8.9% | 58.7% | 67 |
| realizer, left to right, K = 256, (t, h) as in training | 8.0% | 59.3% | 65 |
| realizer, left to right, K = 256, causal canvas masks (adapters on) | 8.7% | 66.2% | 57 |
| realizer, left to right, K = 128 (one token per step for 42 positions, then 2 to 3) | 7.6% | 44.2% | 51 |
| realizer, confidence order, K = 128 | 4.9% | 36.2% | 129 |
| realizer, exact kernel, K = 128 | 0.7% | 20.7% | 129 |
| realizer, exact kernel, K = 32 | 0.9% | 16.7% | 33 |
| realizer, exact kernel, K = 32, (t, h) as in training | 0.7% | 16.0% | 33 |
| realizer, confidence order, K = 32 | 0.0% | 5.1% | 33 |
| realizer, left to right, K = 32 (blocks of about 4 to 6) | 0.0% | 3.1% | 16 |
| realizer, exact kernel, K = 8 | 0.2% | 6.9% | 9 |
| realizer, confidence order, K = 8 | 0.0% | 1.3% | 9 |
| realizer, left to right, K = 8 (blocks of 5, 15, 24, 32) | 0.0% | 0.0% | 5 |

What it shows:

- **The implementation is verified on the real model.** The realizer's code path with the adapters off reproduces
  the backbone (24.7% against 23.8%, within noise).
- **At one token per step the trained realizer keeps 36% of the backbone's Pass@1 (8.9 / 24.7), and most of
  that loss is a flaw in the training data** (next section): 38% of its samples fail at the `def` line. Among the
  others it keeps about 58%. Inputs (t, h) taken from training change nothing (8.0%), and neither do causal
  canvas masks at inference (8.7%): what remains is in the trained weights, not in the step-size input and not
  in what the decoder attends to. (The adapters were trained under bidirectional masks, so the causal run is
  off their training distribution as well; it rules out a fix at inference and says nothing about what training
  on prefix-shaped states gives.)
- **Parallel decisions cost a further factor of about 10 at K = 32 and 40 at K = 8** over all samples (about 30
  at K = 8 among samples with a valid `def` line), and the cost depends on the order. The exact jump kernel (random positions) barely improves from K = 32 to K = 128 (0.9% and 0.7%): tokens
  committed early far from any known context are guesses that are never revised. Committing contiguous blocks of
  4 or more tokens fails completely (left to right at K = 32 and K = 8), because neighbours decided together
  repeat each other.

### A flaw in the first run's training data (found on 20 Nov 2026)

The first run's `write` examples ended in `assert name(a, b) == ...` with the function's parameter names. In
training, the parameters of the `def` line were therefore always the arguments of the assert, and the realizer
learned to copy them. MBPP's asserts hold literal values, so on MBPP the copy is not Python:
`def count_element_freq((5, 6, (5, 6), 7, (8, 9), 9)):`, `def add_str((5, 6, 7, 4, 9) , "FDF"):`, and at K = 8
a garbled copy, `def add_str((5,6,7):`.

| MBPP split | decoder | samples | no def | bad def | literal | copied |
|---|---|---|---|---|---|---|
| validation | baseline | 450 | 0.0% | 0.0% | 0.0% | 0.0% |
| validation | code check (adapters off) | 450 | 0.0% | 0.0% | 0.0% | 0.0% |
| validation | realizer, left to right, K = 256 | 450 | 0.0% | 38.4% | 36.9% | 31.1% |
| validation | realizer, left to right, K = 128 | 450 | 0.0% | 37.3% | 34.9% | 29.8% |
| validation | realizer, exact kernel, K = 8 | 450 | 0.2% | 57.3% | 46.4% | 9.6% |
| test | baseline | 5,000 | 0.1% | 0.0% | 0.1% | 0.2% |
| test | realizer, exact kernel, K = 8 | 5,000 | 1.4% | 57.1% | 49.2% | 15.2% |

Shares of all samples, from `python -m text_vpa.stage0.analyze --eval RUN/sweep --mbpp data2/eval/mbpp.jsonl`:
no top-level `def` line; a first `def` line that is not valid Python; a digit or quote in its parameter list;
a parameter list equal to the arguments of the test's assert. The baseline's 0.2% are tests whose argument is
itself a name (`is_tree_balanced(root)`).

- **Nearly all syntax errors at one token per step are this flaw.** 41.3% of those samples do not parse and
  38.4% have a bad `def` line. Among the 61.6% with a valid one, about 95% parse (the backbone: 90%) and about
  14% pass (8.9 / 0.616), which is 58% of the code check. The model chose which samples those are, so 58% is an
  estimate of the adaptation's cost without the flaw, not a measurement of it.
- **MBPP's realizer rows understate the realizer everywhere,** in the first run's table and in this sweep.
  HumanEval's are not affected (the prompt gives the signature), and H2's verdict stands (see the first run).
- **Fixed on 20 Nov** in `data.write_example`: the line is `assert name(...) == ...`. The first run's numbers
  stay in the tables as measured.

## The second run: direction A (prefix-shaped states), decided on 20 Nov 2026

The design is kept (one canvas, constant depth); the training changes in two ways: the data flaw is fixed, and
half of the training states are prefix-shaped (decision 21). A validation run comes first:

    VALIDATION_ONLY=1 RUN=runs/stage0-a STEPS=10000 TRAIN_SET="prefix_fraction=0.5" bash scripts/ncp/stage0.sh

- **Training:** from the released checkpoint (not from the first run's adapters, which carry the copying),
  10,000 steps of 32 examples, otherwise the first run's recipe. About 14 h on the L4.
- **While it trains:** every 2,500 steps the held-out next-token loss in prefix states, `val_loss_next`, next to
  the frozen backbone's own on the same tokens, `val_loss_next_backbone` (`bash scripts/ncp/stage0.sh status`
  prints the curve). It is a check that training works, not a forecast of the gate. Measured on the first run's
  checkpoints before the second run started (20 Nov, same 256 functions, same positions):

  | step | 2,500 | 5,000 | 7,500 | 10,000 | 25,000 |
  |---|---|---|---|---|---|
  | first run, `val_loss_next` (nats) | 1.028 | 1.075 | 1.066 | 1.071 | 0.990 |
  | first run, next-token accuracy | 76.2% | 73.8% | 73.4% | 75.4% | 77.3% |

  The backbone: 1.187 nats, 71.5%. The first run's realizer predicted the next token in prefix-shaped states
  *better* than its backbone from step 2,500 on, without prefix-shaped training, yet kept 36% of its Pass@1 at
  one token per step (about 58% without the data flaw). So this loss does not measure what Pass@1 loses: part of
  the realizer's lead is the corpus format it was trained on (the `<eos>` after the function, which the backbone
  does not expect), and a mean over tokens hides the rare token that breaks a program. If the second run's curve
  matches the first run's, prefix-shaped states did not change next-token prediction, and a gain at the gate
  belongs mostly to the data fix; separating the two would take a run with the fix alone (`prefix_fraction` = 0).
  A curve above the backbone's 1.187 would mean something broke.
- **After training,** on MBPP's validation split (90 problems, 5 samples), in this order: the trained realizer at
  one token per step; the code check; left to right at K = 8, 32, 128; the exact kernel and the confidence order
  at K = 8, 32, 128; the baseline; and as a control the rows with (t, h) as in training. `RUN/sweep/report.md`
  carries the gate (decision 22), `analysis.md` the def-line table. About 5 h.
- **Not touched:** HumanEval and MBPP's test split. No H2 verdict comes out of this run. If the gate passes, the
  protocol run of the same recipe (25,000 steps, both benchmarks, the rule of decision 17) follows.

How the result is read, fixed before the run:

1. **Def lines.** Bad and copied `def` lines at the baseline's level (0%): the fix worked. Anything else: stop
   and read samples.
2. **The gate.** rho_1 >= 0.8: one token per step works, and the question becomes what fewer steps cost.
   rho_1 <= 0.5: prefix-shaped states added nothing to the fix; the adaptation recipe is the problem (rank,
   budget, weight near the first masked position). In between: continue the run or reweight.
3. **K = 8, 32, 128.** This is what direction A is about. Prefix-shaped states teach the realizer the states of
   sequential decoding. They do not remove the cost of deciding many tokens in one pass from the same context
   (repeated neighbours, contiguous blocks of 4 or more failing completely in the first run). The report prints
   H2's ratio on this split for reference (first run: 0.2 / 23.8). If the gate passes and K = 8 stays near
   zero, the remaining gap is the factorization error the plan is meant to remove (Proposition 1), and Stage 1
   gets a baseline to be measured against at K = 32 to 128, where this realizer is not at the floor.
4. **Control.** The one-token-per-step row with (t, h) as in training made no difference in the first run
   (8.0% against 8.9%). If it differs from the gate's row by more than 4 points here, the network has started to
   use its time input, both rows are reported, and what the sampler feeds above K = 8 becomes a decision.

### Result of the second run (20 to 21 Nov 2026, `runs/stage0-a`)

10,000 steps in 13.8 h (0.202 steps/s), then the sweep on MBPP's validation split (90 problems, 5 samples). The
first run's value for the same configuration is in parentheses.

| decoder | Pass@1 | parses | mean depth |
|---|---|---|---|
| baseline, released model's `generate` | 23.8% | 87.1% | 120 |
| code check (adapters off, causal, one token per step) | 24.7% | 89.8% | 118 |
| realizer, left to right, K = 256 (one token per step) | 17.1% (8.9) | 98.2% (58.7) | 49 |
| realizer, left to right, K = 256, (t, h) as in training | 18.4% (8.0) | 98.2% (59.3) | 48 |
| realizer, left to right, K = 128 | 17.1% (7.6) | 88.0% (44.2) | 42 |
| realizer, confidence, K = 128 | 9.8% (4.9) | 71.1% (36.2) | 129 |
| realizer, exact kernel, K = 128 | 0.7% (0.7) | 17.6% (20.7) | 129 |
| realizer, confidence, K = 32 | 2.0% (0.0) | 13.8% (5.1) | 33 |
| realizer, left to right, K = 32 | 1.6% (0.0) | 8.2% (3.1) | 15 |
| realizer, exact kernel, K = 32 | 1.1% (0.9) | 14.9% (16.7) | 33 |
| realizer, exact kernel, K = 32, (t, h) as in training | 1.3% (0.7) | 15.6% (16.0) | 33 |
| realizer, exact kernel, K = 8 | 0.4% (0.2) | 7.1% (6.9) | 9 |
| realizer, confidence, K = 8 | 0.0% (0.0) | 1.3% (1.3) | 9 |
| realizer, left to right, K = 8 | 0.0% (0.0) | 3.1% (0.0) | 5 |

**Gate: rho_1 = 0.694 [0.450, 0.965], partial.** H2's pair on this split, for reference: 0.019 [0.000, 0.048].
The K = 8 probe during training: 0.0, 0.0, 0.0, 0.2, 0.4% at steps 0 to 10,000. Held-out next-token loss: 5.179
at step 0, then 1.010, 0.967, 0.960, 0.933 (first run at the same steps: 1.028, 1.075, 1.066, 1.071; backbone
1.187).

Read by the rules fixed before the run:

1. **Def lines: the fix worked.** 0.0% bad and 0.0% copied at K = 256 and K = 128, like the backbone. At
   K = 8 (left to right) 7.1% of the `def` lines are not Python and none is a copy: parallel decisions, not the
   flaw.
2. **The gate is partial, and its interval spans both thresholds:** 90 problems cannot tell 0.6 from 0.9. What is
   left of the gap is not syntax. At one token per step the realizer's samples are 17.1% passed, 1.8% syntax
   errors, 58.4% wrong answers, 22.7% runtime errors (code check: 24.7 / 10.2 / 48.2 / 16.9). Paired over the 90
   problems: both solve 17, only the code check 17, only the realizer 2, neither 54; the first run also failed
   13 of those 17. Two causes are visible in the samples:
   - *Argument count.* 58 of the realizer's 450 samples fail on a call with the wrong number of arguments
     (12.9%; the code check 29, 6.4%; the first run 39 of its 277 samples with a valid `def` line, 14%). MBPP
     often shows a parameter only in the test's arguments, and the realizer of either run does not read the
     count from them. The new assert line neither caused this nor cured it. If the 29 extra failures passed at
     the rate of the realizer's other samples, rho_1 would be about 0.75.
   - *Shorter, simpler programs.* 4.8 lines on average against the code check's 12.2, with plain logic errors
     (`return string.upper() if string.lower() else string.lower()` for a case toggle). The adapters moved the
     model toward the corpus's short functions.

   Neither cause is about the shape of the training masks.
3. **Fewer steps: unchanged in kind.** The exact kernel at K = 128, two tokens per step at random positions,
   stays at 0.7% in both runs while left to right at the same K reaches 17.1%: generation in random order is
   broken whatever the number of steps, and what breaks it is deciding tokens far from known context. The
   confidence order improved (9.8% at K = 128, 2.0% at K = 32) and still fails by a factor of ten at K = 32.
   Contiguous blocks of 4 to 6 tokens collapse (left to right at K = 32: 1.6%). At K = 8 nothing exceeds 0.4%.
4. **Control:** 18.4% against 17.1% with (t, h) as in training, 1.3 points, under the 4 points fixed beforehand.
   The time input is still unused.

What cannot be said: how much of the gain at one token per step belongs to the data fix and how much to the
prefix-shaped states (two changes, one run). The next-token curve is below the first run's at every step, but
the first run's model was scored on prompts in the new format. And the realizer's depth of 49 against the
baseline's 120 is the stopping rule (its own `<eos>` against the truncation rule's look-ahead), not the method.

**Decisions of 21 Nov (the author's):** `runs/stage0-a` is the Stage 0 decoder. No distillation for now, although
the paper's Stage 2 names it (Sec. 5.1: "distillation from a 1,024-step teacher"); it stays available as an
ablation. Next: the planner on top of this realizer, `docs/stage1.md`. The argument count is not fixed in that
stage, so that the plan is its only change.

## Reading the result

- `python -m text_vpa.stage0.analyze --eval RUN/eval --mbpp data2/eval/mbpp.jsonl` sorts every sample into
  passed, empty, syntax error, wrong answer, runtime error or timeout, lists the most frequent syntax errors,
  counts adjacent repeated identifiers (a symptom of positions decided in parallel from the same context), and
  for MBPP checks the first `def` line of every sample (missing, not Python, a literal in the parameter list, a
  copy of the test's arguments).
- `bash scripts/ncp/stage0_sweep.sh` (after the run) evaluates on MBPP's validation split, next to the baseline
  on the same split; outside the protocol, no verdict. `STAGES` chooses and orders its stages; with the code
  check and the one-token-per-step row both present, `report.md` decides the gate of decision 22. It splits the
  gap to the baseline into three parts:
  1. **Code check:** the realizer's code path with the adapters off, causal masks and `left_to_right` at
     K = 256 (exactly one token per step) is the backbone's own decoding; it must match the baseline (on the tiny
     Gemma 4 it reproduces greedy `generate` token for token). If it does not, there is a bug and every Stage 0
     number has to be redone.
  2. **Cost of the adaptation:** the trained realizer, `left_to_right` at K = 256 (one token per step), against
     the code check. Training used uniformly random masks and never a revealed prefix with a masked suffix, so
     part of this cost can be that mismatch. Training also gave the network only (t, h) pairs with t a multiple
     of 1/8 and h in {1/2, 1/4, 1/8}, entered as sinusoidal features of 1000 t and 1000 h; K = 256 gives
     h = 1/256 (3.9 against 125 to 500). The same run with `--time-input matched` (h = 1/8 and the grid time
     whose kappa_t matches the fraction already revealed: a pair seen in training) separates the two.
  3. **Cost of parallel decisions:** the trained realizer at fewer steps: `left_to_right` at K = 128 (one token
     per step for the first 42 positions, then 2 to 3) and K = 8 (blocks of 5, 15, 24, 32), and all three orders
     at K = 8, 32, 128.
  The realizer was trained only at the (t, h) values of K in {2, 4, 8}, so K > 8 is also off its training
  distribution. A left-to-right row stops at its `<eos>` or once the truncation rule has found its stop in the
  revealed prefix; neither changes the program.

- `report.md` lists Pass@1, Pass@10, the fraction of programs that parse, the sequential depth and the mean
  number of tokens for every configuration, then the H2 verdict with its intervals and the learning curve.
- `samples.jsonl` holds every sample: the raw text, the program that was run, whether it parsed, and the test
  outcome. A realizer that fails by syntax errors and one that fails by wrong logic call for different next steps.
- If the verdict is "fails" and the curve is still rising, continue the same run with a larger `STEPS` (the
  learning rate stays at its floor) before concluding anything about the method.
- A control that is not implemented: LoRA fine-tuning of the backbone on the same corpus with causal masks,
  decoded left to right. It would separate what the corpus taught from what parallel decoding costs, and is worth
  running if H2 holds narrowly.
