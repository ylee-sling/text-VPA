# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sampler.py -- Algorithm 1: few-step non-autoregressive sampling with PR-JEPA.

    1   s_x <- f_theta(c)                                                     pass 1
    2   u ~ pi_phi^h(. | s_x)                                                 pass 2
    3-4 {s^(e)} <- g_phi(s_x, u, Delta); Z <- mean; sigma_bar <- spread      pass 3
    5   (K_l, V_l) <- bridge projections of [Z; s_x; e(u)]/gamma              once; fused into pass 4
    6   w <- N tokens drawn from p_0
    7   for k = 0, ..., K-1:                                                  passes 4 .. K+3
    8       t <- k/K; h <- 1/K; pi <- (kappa_{t+h} - kappa_t)/(1 - kappa_t)
    9       P <- p^psi_{1|t}(. | w, t, h, Z)                                  one forward pass
    10-12   for all i in parallel: rho^i(j) = min(P_i(j) e^{-E^i(j|w,Z)/T}, c_max), j != w^i;
            with probability pi sum_j rho^i(j): w^i <- j ~ rho^i / sum rho^i
    15  return w

Runtime guarantees, enforced by assertions (``SamplerConfig``):
  * depth audit: forward hooks record the network evaluations; the sequential chain must be exactly
    [f_theta, pi_phi^h, g_phi] + [F_psi] * K, with the bridge and the readout called once each between
    g_phi and the first F_psi pass (fused into pass 4). Predictor-free guidance batches the conditional
    and null passes into one F_psi call, so the depth stays K + 3.
  * Proposition 4(d): at every parallel update, a position admissible under the plan-mediated hard
    energies stays admissible (``ConstraintEnergyEvaluator.check_parallel_update``).

Optional: the source w_0 is drawn from the plan-admissible tokens (so the premise of Prop. 4(d) holds from
t = 0), and a network-free left-to-right repair enforces the draft-dependent hard terms at the end.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from text_vpa.configs.schema import PLANNER_PASSES, RealizerConfig, SamplerConfig
from text_vpa.data.ast_parser import TreeSitterASTExtractor
from text_vpa.data.tokenization import TextVPATokenizer
from text_vpa.inference.energy_terms import ConstraintEnergyEvaluator, PlanConstraints
from text_vpa.models.realizers.kernels import repair_aware_multiplier, sample_jump, tilted_jump_kernel
from text_vpa.models.realizers.logit_kl import LogitKLRealizer
from text_vpa.models.realizers.paths import KappaSchedule, uniform_time_grid
from text_vpa.models.text_vpa_model import TextVPAModel

__all__ = ["DepthAudit", "DepthAuditError", "PRJEPASampler", "SampleResult"]

COUNTED = ("f_theta", "pi_phi", "g_phi", "F_psi")
FUSED = ("bridge", "readout")


class DepthAuditError(RuntimeError):
    """The generation did not run exactly K + 3 sequential network evaluations."""


class DepthAudit:
    """Records calls of the generation networks with forward hooks (``module(...)`` calls only)."""

    def __init__(self, model: TextVPAModel) -> None:
        self.modules: dict[str, nn.Module] = {
            "f_theta": model.planner.context_encoder,
            "pi_phi": model.planner.selector,
            "g_phi": model.planner.predictor,
            "F_psi": model.realizer,
            "bridge": model.realizer.bridge,
        }
        if model.readout is not None:
            self.modules["readout"] = model.readout
        self.events: list[str] = []
        self._handles: list = []

    def __enter__(self) -> DepthAudit:
        self.events = []
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(lambda *_a, n=name: self.events.append(n)))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def sequential(self) -> list[str]:
        return [e for e in self.events if e in COUNTED]

    @property
    def depth(self) -> int:
        return len(self.sequential)

    def verify(self, num_steps: int) -> None:
        """Raise ``DepthAuditError`` unless the chain is [f_theta, pi_phi^h, g_phi] + [F_psi] * K and the
        fused heads (bridge exactly once, readout at most once) ran between g_phi and the first F_psi."""
        expected = ["f_theta", "pi_phi", "g_phi"] + ["F_psi"] * num_steps
        if self.sequential != expected:
            raise DepthAuditError(f"sequential evaluations {self.sequential} != {expected} (K + 3 = {num_steps + 3})")
        g_index = self.events.index("g_phi")
        first_realizer = self.events.index("F_psi")
        for name, required in (("bridge", True), ("readout", False)):
            calls = [i for i, e in enumerate(self.events) if e == name]
            if len(calls) > 1 or (required and not calls):
                raise DepthAuditError(f"{name} must run {'exactly' if required else 'at most'} once, ran {len(calls)}")
            if any(not g_index < i < first_realizer for i in calls):
                raise DepthAuditError(f"{name} must run after g_phi and before the first F_psi pass (fused, pass 4)")


@dataclass
class SampleResult:
    tokens: Tensor                      # [B, N] int64
    texts: list[str]
    modes: Tensor                       # [B]
    sigma_bar: Tensor                   # [B]
    plans: list[PlanConstraints] | None
    num_steps: int
    depth: int                          # sequential network evaluations (K + 3)
    events: list[str]
    timings_ms: dict[str, float] = field(default_factory=dict)
    unrepaired: int = 0
    hard_violations: int = 0            # positions with infinite energy in the final output (after repair)
    unrealized: int = 0                 # FS-DFM positions that never left w_0 (stalls; refilled by the repair)


class PRJEPASampler:
    """Algorithm 1 over a ``TextVPAModel``.

    Args:
        model: trained networks (switched to eval mode).
        tokenizer: the shared tokenizer.
        realizer_cfg: K, schedule, T, c_max and guidance (``RealizerConfig``).
        sampler_cfg: Algorithm 1 options (``SamplerConfig``).
        energy: constraint energies (None disables the tilt and the repair).
        extractor: AST extractor used to parse decoded signatures (readout plans).
        inventory_threshold: readout inventory threshold.
    """

    def __init__(self, model: TextVPAModel, tokenizer: TextVPATokenizer, realizer_cfg: RealizerConfig,
                 sampler_cfg: SamplerConfig | None = None, energy: ConstraintEnergyEvaluator | None = None,
                 extractor: TreeSitterASTExtractor | None = None, inventory_threshold: float = 0.5,
                 plan_source: str = "readout") -> None:
        if plan_source not in ("readout", "none"):
            raise ValueError("plan_source must be 'readout' or 'none' (pass oracle plans to sample() directly)")
        self.model = model
        self.tokenizer = tokenizer
        self.rcfg = realizer_cfg
        self.scfg = sampler_cfg or SamplerConfig()
        self.energy = energy
        self.extractor = extractor
        self.inventory_threshold = inventory_threshold
        self.plan_source = plan_source
        if tokenizer.vocab_size != model.realizer.vocab_size:
            raise ValueError("the tokenizer and the realizer disagree on |V|")

    # -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        context_mask: Tensor | None = None,
        length: int | None = None,
        num_steps: int | None = None,
        plans: Sequence[PlanConstraints] | None = None,
        mode: Tensor | None = None,
        generator: torch.Generator | None = None,
        profile: bool = False,
    ) -> SampleResult:
        """Generate one output per context row.

        Args:
            context: ``[B, M_c]`` token ids.
            context_mask: ``[B, M_c]`` bool.
            length: canvas length N (default: ``SamplerConfig.default_length``, else the readout's length
                head, else the realizer's maximum).
            num_steps: K (default ``RealizerConfig.num_steps``); 1 <= K <= 8.
            plans: symbolic plans for the energies (e.g. oracle plans); None: decoded by the readout.
            mode: force the mode u (pi_phi^h still runs, keeping the chain at K + 3).
            generator: random generator on the model's device.
            profile: synchronise the device around every stage for accurate timings.
        """
        k_steps = self.rcfg.num_steps if num_steps is None else int(num_steps)
        if not 1 <= k_steps <= 8:
            raise ValueError("K must satisfy 1 <= K <= 8")
        model = self.model.eval()
        device = context.device
        timings: dict[str, float] = {}
        clock = _Clock(device, profile, timings)
        audit = DepthAudit(model)
        with audit:
            with clock("pass1-3_planner"):
                plan = model.planner.plan(context, context_mask, mode=mode, strategy=self.scfg.mode_strategy,
                                          temperature=self.scfg.mode_temperature, generator=generator)
            with clock("fused_readout_bridge"):
                lengths = None
                decode_plans = (plans is None and self.energy is not None and self.extractor is not None
                                and self.plan_source == "readout")
                if model.readout is not None and (decode_plans or length is None):
                    readout_out = model.readout(plan.z_target)
                    if decode_plans:
                        plans, lengths = model.readout.decode(readout_out, self.tokenizer, self.extractor,
                                                              self.energy.analysis, self.inventory_threshold)
                    else:
                        lengths = (readout_out.length_logits.argmax(-1) + 1) * model.readout.length_bucket_size
                guidance = self.rcfg.guidance_scale > 0
                cache = model.realizer.prepare(plan, with_null=guidance)
            n = self._canvas_length(length, lengths)
            energy = self.energy
            if energy is not None and plans is None:
                # No symbolic plan: plan-free terms (special tokens, grammar) still apply.
                plans = [PlanConstraints(source="none")] * context.shape[0]
            if plans is not None and len(plans) != context.shape[0]:
                raise ValueError("one plan per context row is required")
            if isinstance(model.realizer, LogitKLRealizer):
                tokens, posterior, unrealized = self._logit_kl(cache, n, k_steps, plans, energy, generator, clock,
                                                               guidance)
            else:
                tokens, posterior, unrealized = self._fs_dfm(cache, n, k_steps, plans, energy, generator, clock,
                                                             guidance)
        if self.scfg.audit_depth:
            audit.verify(k_steps)
        unrepaired = 0
        violations = 0
        if energy is not None and plans is not None:
            source = model.realizer.mask_id if getattr(model.realizer, "source", "") == "mask" else None
            if self.scfg.final_repair:
                with clock("repair"):
                    tokens, unrepaired = energy.repair(tokens, posterior, plans, source_token=source,
                                                       unrealized=unrealized, temperature=self.rcfg.temperature)
            violations = int(energy.hard_violations(tokens, plans).sum())
        texts = [self.tokenizer.decode(row) for row in tokens.tolist()]
        return SampleResult(tokens, texts, plan.mode, plan.sigma_bar, list(plans) if plans is not None else None,
                            k_steps, audit.depth, list(audit.events), timings, unrepaired, violations,
                            0 if unrealized is None else int(unrealized.sum()))

    # -----------------------------------------------------------------------------------------
    def _canvas_length(self, length: int | None, predicted: Tensor | None) -> int:
        max_len = self.model.realizer.max_len
        if length is None:
            length = self.scfg.default_length
        if length is None and predicted is not None:
            length = min(int(predicted.max()), max_len)          # a bucket may end past the canvas
        if length is None:
            length = max_len
        if not 1 <= length <= max_len:
            raise ValueError(f"canvas length must lie in [1, {max_len}], got {length}")
        return length

    def _posterior(self, logits: Tensor, batch: int, guidance: bool) -> Tensor:
        p = torch.softmax(logits.float(), dim=-1)
        if guidance:
            w = self.rcfg.guidance_scale
            p = (1 + w) * p[:batch] - w * p[batch:]                 # lower clip at 0 in the kernel (Eq. 16)
        return p

    def _fs_dfm(self, cache, n: int, k_steps: int, plans, energy, generator, clock: _Clock, guidance: bool):
        realizer = self.model.realizer
        b = cache.mask.shape[0] // (2 if guidance else 1)
        device = cache.mask.device
        schedule = KappaSchedule(self.rcfg.schedule)
        source_token = realizer.mask_id if realizer.source == "mask" else None
        admissible = energy.plan_admissible(plans) if energy is not None else None   # A(Z), [B, S]
        w = realizer.sample_source(b, n, device, generator,
                                   allowed=admissible if self.scfg.restrict_source else None)  # line 6
        realized = torch.zeros_like(w, dtype=torch.bool)       # a jump always changes the token (j != w^i)
        p = None
        for k, (t, h) in enumerate(uniform_time_grid(k_steps)):                       # line 7
            pi = schedule.exact_jump_mass(t, h)                                        # line 8
            t_vec = torch.full((b,), t, device=device)
            h_vec = torch.full((b,), h, device=device)
            w_in = torch.cat([w, w]) if guidance else w
            with clock(f"pass{k + 4}_realizer"):
                logits = realizer(w_in, torch.cat([t_vec, t_vec]) if guidance else t_vec,
                                  torch.cat([h_vec, h_vec]) if guidance else h_vec, cache)   # line 9
            with clock(f"step{k}_energy_kernel"):
                p = self._posterior(logits, b, guidance)
                multiplier = tilt = None
                if energy is not None:
                    result = energy(w, plans)
                    tilt = result.multiplier(self.rcfg.temperature).to(device)          # exp(-E/T) <= 1
                    multiplier = (repair_aware_multiplier(result.dense().to(device), w, self.rcfg.temperature)
                                  if self.scfg.repair_aware else tilt)
                kernel = tilted_jump_kernel(p, w, pi, multiplier, self.rcfg.c_max)    # lines 10-11
                w_next = sample_jump(kernel, w, generator)                             # line 12
                if energy is not None and self.scfg.check_admissibility:
                    energy.check_parallel_update(w, w_next, plans, source_token=source_token, allowed=admissible)
                realized |= w_next != w
                w = w_next
        # The repair ranks candidates with the last posterior and re-applies the tilt on the repaired prefix.
        return w, p.clamp_min(0.0), ~realized

    def _logit_kl(self, cache, n: int, k_steps: int, plans, energy, generator, clock: _Clock, guidance: bool):
        realizer = self.model.realizer
        b = cache.mask.shape[0] // (2 if guidance else 1)
        device = cache.mask.device
        l_state = realizer.sample_source(b, n, device, generator)
        posterior: Tensor | None = None
        for k, (t, h) in enumerate(uniform_time_grid(k_steps)):
            t_vec = torch.full((b,), t, device=device)
            h_vec = torch.full((b,), h, device=device)
            l_in = torch.cat([l_state, l_state]) if guidance else l_state
            with clock(f"pass{k + 4}_realizer"):
                logits = realizer(l_in, torch.cat([t_vec, t_vec]) if guidance else t_vec,
                                  torch.cat([h_vec, h_vec]) if guidance else h_vec, cache)
            with clock(f"step{k}_energy_euler"):
                posterior = self._posterior(logits, b, guidance).clamp_min(0.0)
                p = posterior
                if energy is not None:
                    draft = l_state.argmax(dim=-1)
                    tilted = p * energy(draft, plans).multiplier(self.rcfg.temperature).to(device)
                    norm = tilted.sum(dim=-1, keepdim=True)
                    p = torch.where(norm > 0, tilted / norm.clamp_min(1e-30), p)    # p~ ∝ p e^{-E/T}
                else:
                    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-30)
                l_state = realizer.euler_step(l_state, p, t, h)                     # Eq. 21
        # Untilted posterior for the repair, which recomputes the tilt on the repaired prefix.
        return l_state.argmax(dim=-1), posterior, None


class _Clock:
    """Wall-clock timings per stage, synchronising the device when profiling."""

    def __init__(self, device: torch.device, profile: bool, out: dict[str, float]) -> None:
        self.device, self.profile, self.out = device, profile, out

    def _sync(self) -> None:
        if not self.profile:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.out[name] = self.out.get(name, 0.0) + (time.perf_counter() - start) * 1e3


if PLANNER_PASSES != 3:  # pragma: no cover - the audit's expected chain hard-codes three planner passes
    raise ImportError("Algorithm 1 has exactly three planner passes")
