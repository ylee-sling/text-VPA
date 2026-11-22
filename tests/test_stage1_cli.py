# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 1, step 1 end to end through its command lines, on the tiny Gemma 4 checkpoint of ``test_stage0_cli``:
a Stage 0 run, the preflight, the structure oracle started from it (with a resume), its evaluation with the
three plans, and the report with the gate."""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_stage0 import _write_corpus  # noqa: E402
from test_stage0_cli import write_benchmarks, write_checkpoint  # noqa: E402

from text_vpa.stage0 import evaluate as evaluate0  # noqa: E402
from text_vpa.stage0 import train as train0  # noqa: E402
from text_vpa.stage1 import evaluate, preflight, report, train  # noqa: E402
from text_vpa.stage1.plan import ORACLE_LABEL  # noqa: E402
from text_vpa.stage1.slots import Codebook  # noqa: E402


def test_stage1_oracle_pipeline(tmp_path, capsys):
    model, corpus, bench = tmp_path / "model", tmp_path / "corpus", tmp_path / "eval"
    stage0, run = tmp_path / "stage0", tmp_path / "oracle"
    write_checkpoint(model)
    _write_corpus(corpus, functions=24, shards=2)
    write_benchmarks(bench)
    data = ["--corpus", str(corpus), "--eval-dir", str(bench)]
    assert train0.main([*data, "--run", str(stage0), "--device", "cpu", "--set", f"model={model}", "canvas_len=64",
                        "max_prompt_tokens=128", "lora_rank=4", "lora_alpha=4", "steps=4", "micro_batch=2",
                        "grad_accum=2", "warmup_steps=2", "save_every=2", "eval_every=4", "val_examples=4",
                        "probe_samples=2", "probe_steps=4", "log_every=2", "prefix_fraction=0.5"]) == 0
    start = torch.load(stage0 / "last.pt", weights_only=True)

    # Preflight: token offsets, lines, MBPP's references as the oracle reads them, skeleton coverage.
    common = ["--model", str(model), *data, "--canvas-len", "64", "--max-slots", "8", "--codebook-sample", "24",
              "--codebook-size", "4", "--exec-workers", "1"]
    assert preflight.main([*common, "--init", str(stage0 / "last.pt"), "--out", str(run / "preflight.json")]) == 0
    found = json.loads((run / "preflight.json").read_text())
    assert found["init"]["stage"] == "stage0" and found["init"]["step"] == 4 and not found["problems"]
    assert found["offsets"]["corpus_same_slots"] == 1.0 and found["offsets"]["mbpp_same_slots"] == 1.0
    assert found["mbpp_validation"]["problems"] == 2 and found["mbpp_validation"]["pass_normalised"] == 2
    assert found["mbpp_validation"]["indented_in_fours_normalised"] == 1.0
    assert found["lines"]["corpus_complete"]["within_slots"] == 1.0
    assert found["skeletons"]["with_depth"]["held_out"] == 1.0          # every toy function has the same shape
    assert len(Codebook.load(run / "skeleton_codebook.json")) == 3      # "other", the write and the complete shape
    assert preflight.main([*common, "--init", str(tmp_path / "missing.pt")]) == 1
    assert preflight.main([*common, "--canvas-len", "32", "--init", str(stage0 / "last.pt")]) == 1

    # The oracle run: starts from the Stage 0 checkpoint and its data position, resumes, and labels itself.
    small = ["latent_dim=16", "bridge_width=16", "bridge_heads=4", "position_dim=8", "max_slots=8", "steps=4",
             "eval_every=2", "save_every=2"]
    args = [*data, "--run", str(run), "--device", "cpu"]
    with pytest.raises(SystemExit, match="--init"):
        train.main(args)
    assert train.main([*args, "--init", str(stage0 / "last.pt"), "--set", *small]) == 0
    first = torch.load(run / "last.pt", weights_only=True)
    assert first["stage"] == "stage1-oracle" and first["label"] == ORACLE_LABEL and first["step"] == 4
    assert first["config"]["prefix_fraction"] == 0.5 and first["config"]["canvas_len"] == 64        # the recipe
    assert first["config"]["schedule_steps"] == 4 and first["config"]["init"] == str(stage0 / "last.pt")
    assert first["init"] == {"path": str(stage0 / "last.pt"), "stage": "stage0", "step": 4}
    assert first["examples_seen"] == 16 and first["stream"] != start["stream"]
    assert set(first["trainable"]) == set(start["trainable"]) and "gate" in first["bridge"]
    config = json.loads((run / "config.json").read_text())
    assert config["label"] == ORACLE_LABEL and config["trainable_parameters"]["bridge"] > 0
    assert ORACLE_LABEL in (run / "ORACLE.md").read_text()
    assert train.main([*args, "--steps", "6"]) == 0                   # resume and extend
    second = torch.load(run / "last.pt", weights_only=True)
    assert second["step"] == 6 and second["config"]["schedule_steps"] == 4
    metrics = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    evaluations = [m for m in metrics if "eval/plan_gain_k0" in m]
    assert [m["step"] for m in evaluations] == [0, 2, 4, 6]
    assert evaluations[0]["eval/plan_gain_k0"] == 0.0                 # step 0 is the Stage 0 checkpoint exactly
    assert all("eval/probe_pass@1" in m and "eval/probe_kind_match" in m and "eval/val_loss_shuffled_k0" in m
               for m in evaluations)
    assert any("train/gate_mean" in m for m in metrics)
    # Stage 0's tools do not take an oracle checkpoint for one of their own.
    with pytest.raises(SystemExit, match="stage1.evaluate"):
        evaluate0.main(["--decoder", "realizer", "--checkpoint", str(run / "last.pt"), "--model", str(model),
                        "--eval-dir", str(bench), "--out", str(tmp_path / "wrong"), "--device", "cpu"])
    with pytest.raises(SystemExit, match="not a Stage 0 run"):
        train0.main([*data, "--run", str(run), "--device", "cpu"])
    with pytest.raises(SystemExit, match="Stage 0 checkpoint"):
        train.main([*data, "--run", str(tmp_path / "other"), "--device", "cpu", "--init", str(run / "last.pt")])

    # Evaluation on MBPP's validation split with the three plans, then the report with the gate.
    sweep = run / "sweep"
    shared = ["--checkpoint", str(run / "last.pt"), "--model", str(model), "--eval-dir", str(bench), "--out",
              str(sweep), "--num-samples", "2", "--device", "cpu", "--exec-workers", "1"]
    assert evaluate.main([*shared, "--plans", "oracle", "null", "shuffled", "--k-steps", "2", "8"]) == 0
    assert evaluate.main([*shared, "--k-steps", "64", "--orders", "left_to_right"]) == 0
    tags = sorted(d.name for d in (sweep / "mbpp").iterdir())
    assert tags == ["null-K2", "null-K8", "oracle-K2", "oracle-K64-left_to_right", "oracle-K8", "shuffled-K2",
                    "shuffled-K8"]
    k8 = json.loads((sweep / "mbpp" / "oracle-K8" / "summary.json").read_text())
    assert k8["label"] == ORACLE_LABEL and k8["plan"] == "oracle" and k8["stage"] == "stage1-oracle"
    assert k8["split"] == "validation" and k8["problems"] == 2 and k8["step"] == 6 and k8["sequential_depth"] == 9
    assert k8["init"]["step"] == 4 and 0.0 <= k8["kind_match"] <= 1.0
    rows = [json.loads(line) for line in open(sweep / "mbpp" / "oracle-K8" / "samples.jsonl", encoding="utf-8")]
    assert len(rows) == 4 and all(row["plan_lines"] == 2 and "kind_match" in row for row in rows)
    capsys.readouterr()
    before = (sweep / "mbpp" / "null-K8" / "summary.json").stat().st_mtime_ns
    assert evaluate.main([*shared, "--plans", "null", "--k-steps", "8"]) == 0     # finished: skipped
    assert (sweep / "mbpp" / "null-K8" / "summary.json").stat().st_mtime_ns == before

    reference = tmp_path / "stage0-sweep"
    assert evaluate0.main(["--decoder", "ar", "--model", str(model), "--eval-dir", str(bench), "--out",
                           str(reference), "--benchmarks", "mbpp", "--mbpp-split", "validation", "--num-samples",
                           "2", "--k", "1", "--device", "cpu"]) == 0
    assert report.main(["--eval", str(sweep), "--run", str(run), "--reference", str(reference)]) == 0
    result = json.loads((sweep / "report.json").read_text())
    assert result["label"] == ORACLE_LABEL and len(result["rows"]) == 7
    assert "rho" in result["gate"] and (result["gate"]["rho"] is None or "verdict" in result["gate"])
    assert result["gate"].get("reason") is None                      # the baseline was scored on the same problems
    assert [point["step"] for point in result["curve"]] == [0, 2, 4, 6]
    assert result["delivery"]["row"] == "oracle-K64-left_to_right"   # K = N = 64: one token per step
    assert result["delivery"]["kind_match"] is not None and "not evaluated" not in result["delivery"]["verdict"]
    text = (sweep / "report.md").read_text()
    assert text.startswith("# Text-VPA Stage 1, step 1: the structure oracle (oracle: not the method)")
    assert text.count(ORACLE_LABEL) >= 8 and "## Gate after step 1" in text

    # Step 2, the latent oracle, with the realizer's adapters frozen: train, resume, evaluate, report.
    latent = tmp_path / "latent"
    latent_args = [*data, "--run", str(latent), "--device", "cpu"]
    assert train.main([*latent_args, "--init", str(stage0 / "last.pt"), "--set", *small, "plan=oracle-latent",
                       "latent_layers=2", "train_adapters=false"]) == 0
    assert train.main([*latent_args, "--steps", "6"]) == 0
    trained = torch.load(latent / "last.pt", weights_only=True)
    assert trained["plan"] == "oracle-latent" and trained["step"] == 6 and trained["label"] == ORACLE_LABEL
    for name, value in start["trainable"].items():
        assert torch.equal(trained["trainable"][name], value), name      # frozen adapters: the Stage 0 values
    assert json.loads((latent / "config.json").read_text())["trainable_parameters"]["realizer_adapters"] == 0
    assert "states of the reference program's lines" in (latent / "ORACLE.md").read_text()
    latent_sweep = latent / "sweep"
    assert evaluate.main(["--checkpoint", str(latent / "last.pt"), "--model", str(model), "--eval-dir", str(bench),
                          "--out", str(latent_sweep), "--num-samples", "2", "--device", "cpu", "--exec-workers", "1",
                          "--plans", "oracle", "null", "--k-steps", "8"]) == 0
    row = json.loads((latent_sweep / "mbpp" / "oracle-K8" / "summary.json").read_text())
    assert row["oracle"] == "latent" and row["label"] == ORACLE_LABEL and "reproduced" in row
    assert report.main(["--eval", str(latent_sweep), "--run", str(latent), "--reference", str(reference),
                        "--structure-run", str(run)]) == 0
    latent_report = json.loads((latent_sweep / "report.json").read_text())
    assert latent_report["oracle"] == "latent" and latent_report["carried"]["structure_loss"] is not None
    assert (latent_sweep / "report.md").read_text().startswith("# Text-VPA Stage 1, step 2: the latent oracle")
