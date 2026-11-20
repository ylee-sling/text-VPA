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
    `complete`): description = its docstring plus the line
    `assert name(parameters) == ...` (MBPP names the function in an assert; the corpus has no tests), canvas =
    imports + the function without its docstring. Prompts over 512 tokens are dropped.
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

## Diagnostic sweep (20 Nov 2026, `runs/stage0/sweep`)

MBPP validation (90 problems, 5 samples each), the final checkpoint; outside the protocol, no verdict.

| decoder | Pass@1 | parses | mean depth |
|---|---|---|---|
| baseline, the released model's `generate` | 23.8% | 87.1% | 120 |
| backbone through the realizer's code path (adapters off, causal, one token per step) | 24.7% | 89.8% | 118 |
| realizer, left to right, K = 256 (one token per step) | 8.9% | 58.7% | 67 |
| realizer, left to right, K = 256, (t, h) as in training | 8.0% | 59.3% | 65 |
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
- **The adaptation is the first bottleneck.** Decoding one token per step, the trained realizer keeps about 36% of
  the backbone's Pass@1 (8.9 / 24.7). Inputs (t, h) taken from training change nothing (8.0%), so the cause is
  not the step-size input. With this recipe H2 is out of reach at any K, before parallel decisions enter.
- **Parallel decisions cost a further factor of about 10 at K = 32 and 40 at K = 8**, and the cost depends on the
  order. The exact jump kernel (random positions) barely improves from K = 32 to K = 128 (0.9% and 0.7%): tokens
  committed early far from any known context are guesses that are never revised. Committing contiguous blocks of
  4 or more tokens fails completely (left to right at K = 32 and K = 8), because neighbours decided together
  repeat each other.

## Reading the result

- `python -m text_vpa.stage0.analyze --eval RUN/eval` sorts every sample into passed, empty, syntax error, wrong
  answer, runtime error or timeout, lists the most frequent syntax errors, and counts adjacent repeated
  identifiers (a symptom of positions decided in parallel from the same context).
- `bash scripts/ncp/stage0_sweep.sh` (after the run) evaluates on MBPP's validation split, next to the baseline
  on the same split; outside the protocol, no verdict. It splits the gap to the baseline into three parts:
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
