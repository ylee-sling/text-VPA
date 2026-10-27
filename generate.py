# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
generate.py -- few-step non-autoregressive generation with a trained text-VPA checkpoint (Algorithm 1).

    python generate.py --prompt "def fibonacci(n: int) -> int:" --k_steps 4 --realizer fs_dfm \\
        --checkpoint ./checkpoints/best_stage2.pt

Prints the generated text, the sampled design mode u, the plan uncertainty sigma_bar, the audited
sequential depth (K + 3) and a per-stage latency breakdown (planner passes 1-3, the fused readout/bridge
projections, every realizer pass and its energy-tilted kernel, the final repair).
"""

from __future__ import annotations

import argparse
import sys

import torch

from text_vpa.configs.schema import RealizerConfig, SamplerConfig
from text_vpa.data.ast_parser import ModeExtractor
from text_vpa.inference.loading import build_sampler, load_model
from text_vpa.utils.device import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PR-JEPA few-step generation (Algorithm 1)")
    p.add_argument("--prompt", required=True, help="specification / signature (the context c)")
    p.add_argument("--checkpoint", required=True, help="Stage-2 or Stage-3 checkpoint")
    p.add_argument("--k_steps", "--k-steps", dest="k_steps", type=int, default=None, help="K (1..8); default: config")
    p.add_argument("--realizer", choices=("fs_dfm", "logit_kl"), default=None,
                   help="must match the checkpoint's realizer (checked)")
    p.add_argument("--length", type=int, default=None, help="canvas length N (default: the readout's length head)")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=None, help="energy temperature T")
    p.add_argument("--c-max", type=float, default=None, help="safety clip c_max in (0, 1]")
    p.add_argument("--guidance", type=float, default=None, help="predictor-free guidance scale omega")
    p.add_argument("--mode-strategy", choices=("sample", "greedy", "gumbel"), default=None)
    p.add_argument("--no-energy", action="store_true", help="disable the constraint energies and the repair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    loaded = load_model(args.checkpoint, device)
    cfg = loaded.cfg
    if args.realizer is not None and args.realizer != cfg.realizer.kind:
        print(f"error: the checkpoint was trained with the {cfg.realizer.kind} realizer, not {args.realizer}",
              file=sys.stderr)
        return 2
    rupdate = {k: v for k, v in (("temperature", args.temperature), ("c_max", args.c_max),
                                 ("guidance_scale", args.guidance)) if v is not None}
    realizer_cfg = RealizerConfig.model_validate({**cfg.realizer.model_dump(), **rupdate})
    sampler_cfg = cfg.sampler
    if args.mode_strategy is not None:
        sampler_cfg = SamplerConfig.model_validate({**cfg.sampler.model_dump(), "mode_strategy": args.mode_strategy})
    sampler = build_sampler(loaded, use_energy=not args.no_energy)
    sampler.rcfg, sampler.scfg = realizer_cfg, sampler_cfg
    k_steps = args.k_steps or realizer_cfg.num_steps
    if k_steps not in realizer_cfg.step_budgets:
        print(f"warning: K = {k_steps} is not a trained step budget {list(realizer_cfg.step_budgets)}; "
              "the step-size conditioning h = 1/K is outside training", file=sys.stderr)

    ids = loaded.tokenizer.encode(args.prompt)[: cfg.data.max_context_tokens]
    context = torch.tensor([ids] * args.num_samples, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = sampler.sample(context, length=args.length, num_steps=k_steps, generator=generator, profile=True)

    modes = ModeExtractor(cfg.data.mode_scheme)
    print(f"checkpoint: stage {loaded.stage}, step {loaded.step}, realizer {cfg.realizer.kind}, K = {k_steps}, "
          f"N = {result.tokens.shape[1]}, device {device}")
    for i, text in enumerate(result.texts):
        u = int(result.modes[i])
        label = modes.describe(u) if u < modes.num_modes else "unassigned mode"
        print(f"\n--- sample {i}: mode u = {u} ({label}), sigma_bar = {float(result.sigma_bar[i]):.4f}")
        print(text)
    print(f"\nsequential network evaluations: {result.depth} (K + 3 = {k_steps + 3}, audited)")
    if not args.no_energy:
        print(f"hard-constraint violations after repair: {result.hard_violations} (unrepairable: {result.unrepaired})")
        if cfg.realizer.kind == "fs_dfm":
            print(f"positions that never left the source token (refilled by the repair): {result.unrealized}")
    print("\nlatency breakdown (ms):")
    total = 0.0
    for stage, ms in result.timings_ms.items():
        total += ms
        print(f"  {stage:<26} {ms:9.2f}")
    print(f"  {'total':<26} {total:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
