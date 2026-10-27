# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Networks of the K + 3 generation chain (PR-JEPA Sec. 3, Algorithm 1).

    encoders.py   f_theta (pass 1) and its frozen EMA copy f_theta_bar (training only)
    ema.py        EMA update theta_bar <- m_k theta_bar + (1 - m_k) theta (Eq. 4)
    selector.py   pi_phi^h, the discrete mode selector (pass 2)
    predictor.py  g_phi with n_ens ensemble heads (pass 3; Eqs. 3, 8)
    bridge.py     cross-attention bridge with spectrally normalised, cached K/V (Sec. 3.2, Eq. 10)
    readout.py    the plan's slot decoder: symbolic plan content read out of Z_target (Sec. 3.3)
    plan.py       PredictedPlan, the only conditioning the realizer accepts (Sec. 5.1 anti-leakage)
    text_vpa_model.py  TextVPAPlanner (passes 1-3) and TextVPAModel (planner + realizer + readout)
    layers.py     shared attention / transformer building blocks
    realizers/    FS-DFM (Eqs. 11-17) and Logit-KL (Eqs. 18-21) realizers, AdaLN backbone, jump kernel
"""

from text_vpa.models.bridge import BridgeConditioning, BridgeKVCache, CrossAttentionBridge
from text_vpa.models.ema import EMATargetEncoder, ema_momentum
from text_vpa.models.encoders import ContextEncoder, TargetEncoder, TargetOutput
from text_vpa.models.layers import EnsembleLinear, MultiHeadAttention, TransformerBlock, embed_modes, masked_mean
from text_vpa.models.predictor import JEPAPredictor, PredictorOutput, ensemble_mean, ensemble_uncertainty
from text_vpa.models.selector import ModeSelection, ModeSelector, choose_mode

__all__ = [
    "BridgeConditioning",
    "BridgeKVCache",
    "ContextEncoder",
    "CrossAttentionBridge",
    "EMATargetEncoder",
    "EnsembleLinear",
    "JEPAPredictor",
    "ModeSelection",
    "ModeSelector",
    "MultiHeadAttention",
    "PredictorOutput",
    "TargetEncoder",
    "TargetOutput",
    "TransformerBlock",
    "choose_mode",
    "ema_momentum",
    "embed_modes",
    "ensemble_mean",
    "ensemble_uncertainty",
    "masked_mean",
]
