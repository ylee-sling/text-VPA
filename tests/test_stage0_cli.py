# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 0 end to end through its command lines, on a tiny random Gemma 4 checkpoint saved to disk in the
released layout (``Gemma4ForConditionalGeneration`` + a fast tokenizer with <mask>): preflight, training with a
resume, both decoders in the Pass@k harness, and the report."""

from __future__ import annotations

import gzip
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_stage0 import LAYER_TYPES, VOCAB, WINDOW, _write_corpus  # noqa: E402

from text_vpa.stage0 import evaluate, preflight, report, train  # noqa: E402


def write_checkpoint(path) -> None:
    torch.manual_seed(0)
    text = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=7, num_attention_heads=4,
                num_key_value_heads=1, head_dim=16, global_head_dim=32, sliding_window=WINDOW,
                layer_types=list(LAYER_TYPES), final_logit_softcapping=30.0, vocab_size_per_layer_input=VOCAB,
                hidden_size_per_layer_input=8, num_kv_shared_layers=3, use_double_wide_mlp=True,
                max_position_embeddings=1024)
    config = transformers.Gemma4Config(text_config=text, vision_config=None, audio_config=None)
    transformers.Gemma4ForConditionalGeneration(config).save_pretrained(path)
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "\n": 5}
    vocab.update({chr(c): c - 32 + 6 for c in range(32, 127)})
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Split("", "isolated")
    tok.decoder = tokenizers.decoders.Fuse()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>",
                                         pad_token="<pad>", unk_token="<unk>", mask_token="<mask>"
                                         ).save_pretrained(path)


def write_benchmarks(path) -> None:
    path.mkdir(parents=True)
    with gzip.open(path / "HumanEval.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i in range(3):
            prompt = f'def add{i}(a, b):\n    """Add a, b and {i}."""\n'
            handle.write(json.dumps({"task_id": f"HumanEval/{i}", "prompt": prompt, "entry_point": f"add{i}",
                                     "canonical_solution": f"    return a + b + {i}\n",
                                     "test": f"def check(f):\n    assert f(1, 2) == {3 + i}\n"}) + "\n")
    with open(path / "mbpp.jsonl", "w", encoding="utf-8") as handle:
        for task_id in (11, 12, 511, 512):                           # two test problems, two validation problems
            handle.write(json.dumps({"task_id": task_id, "text": f"Write a function that adds {task_id}.",
                                     "code": f"def plus(x):\n    return x + {task_id}\n",
                                     "test_list": [f"assert plus(1) == {task_id + 1}"], "test_setup_code": ""}) + "\n")


def test_a_lone_surrogate_in_the_corpus_is_dropped_not_fatal(tmp_path):
    """The Hugging Face tokenizer rejects a whole batch over one lone surrogate (it stopped the first L4 run)."""
    from text_vpa.stage0.data import CorpusStream, HFTokenizer

    write_checkpoint(tmp_path / "model")
    _write_corpus(tmp_path / "corpus", functions=8, shards=1)
    bad = {"context": "def bad(a):\n", "target": 'def bad(a):\n    return "\ud83d"\n', "documented": False}
    with gzip.open(tmp_path / "corpus" / "train-00000.jsonl.gz", "at", encoding="utf-8") as handle:
        handle.write(json.dumps(bad) + "\n")                            # json.dumps escapes it as \\ud83d
    tokenizer = HFTokenizer(transformers.AutoTokenizer.from_pretrained(tmp_path / "model"))
    with pytest.raises(TypeError, match="TextEncodeInput"):
        tokenizer.encode_batch([bad["target"]])                         # the failure being guarded against
    stream = CorpusStream(tmp_path / "corpus", tokenizer, 64, 128, write_fraction=0.0, chunk_rows=4)
    assert len(stream.take(16)) == 16 and stream.dropped["invalid_unicode"] >= 1


def test_stage0_pipeline(tmp_path, capsys):
    model, corpus, bench, run = tmp_path / "model", tmp_path / "corpus", tmp_path / "eval", tmp_path / "run"
    write_checkpoint(model)
    _write_corpus(corpus, functions=24, shards=2)
    write_benchmarks(bench)

    # The tiny checkpoint is not E2B: the preflight says so, and passes once that is allowed.
    common = ["--model", str(model), "--corpus", str(corpus), "--eval-dir", str(bench), "--canvas-len", "64"]
    assert preflight.main(common) == 1
    assert preflight.main([*common, "--allow-mismatch", "--out", str(run / "preflight.json")]) == 0
    found = json.loads((run / "preflight.json").read_text())
    assert found["tokenizer"]["mask"] == 4 and found["benchmarks"]["humaneval"]["fit_canvas"] == 1.0
    assert found["benchmarks"]["humaneval"]["round_trip"] == 1.0 and found["benchmarks"]["mbpp"]["round_trip"] == 1.0
    assert any(line.startswith("blocks:") for line in found["architecture_differences"])

    settings = [f"model={model}", "canvas_len=64", "max_prompt_tokens=128", "lora_rank=4", "lora_alpha=4",
                "steps=4", "micro_batch=2", "grad_accum=2", "warmup_steps=2", "save_every=2", "eval_every=2",
                "val_examples=4", "probe_samples=2", "probe_steps=4", "log_every=2"]
    args = ["--corpus", str(corpus), "--eval-dir", str(bench), "--run", str(run), "--device", "cpu"]
    assert train.main([*args, "--set", *settings]) == 0
    first = torch.load(run / "last.pt", weights_only=True)
    assert first["step"] == 4 and first["config"]["schedule_steps"] == 4 and (run / "step-000004.pt").exists()
    assert set(first["trainable"]) and all("lora_" in k or k.startswith(("mask_", "t_mlp", "h_mlp", "mod_"))
                                           for k in first["trainable"])
    assert json.loads((run / "memory.json").read_text())["micro_batch"] == 2
    assert train.main([*args, "--steps", "6"]) == 0                  # resume and extend
    second = torch.load(run / "last.pt", weights_only=True)
    assert second["step"] == 6 and second["config"]["schedule_steps"] == 4 and second["examples_seen"] == 24
    metrics = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    probes = [m["step"] for m in metrics if "eval/probe_pass@1" in m]
    assert probes == [0, 2, 4, 6] and all("eval/val_loss" in m for m in metrics if "eval/probe_pass@1" in m)

    out = run / "eval"
    shared = ["--model", str(model), "--eval-dir", str(bench), "--out", str(out), "--num-samples", "2", "--k", "1",
              "2", "--device", "cpu"]
    assert evaluate.main(["--decoder", "reference", *shared]) == 0
    assert json.loads((out / "humaneval" / "reference" / "summary.json").read_text())["pass@1"] == 1.0
    assert json.loads((out / "mbpp" / "reference" / "summary.json").read_text())["problems"] == 2
    assert evaluate.main(["--decoder", "ar", *shared]) == 0
    assert evaluate.main(["--decoder", "realizer", "--checkpoint", str(run / "last.pt"), "--k-steps", "2", "8",
                          "--orders", "random", "confidence", *shared]) == 0
    tags = sorted(d.name for d in (out / "humaneval").iterdir())
    assert tags == ["ar", "realizer-K2", "realizer-K2-confidence", "realizer-K8", "realizer-K8-confidence",
                    "reference"]
    k8 = json.loads((out / "mbpp" / "realizer-K8" / "summary.json").read_text())
    assert k8["sequential_depth"] == 9 and k8["canvas_len"] == 64 and k8["step"] == 6 and k8["schedule"] == "cosine"
    ar = json.loads((out / "humaneval" / "ar" / "summary.json").read_text())
    assert ar["max_new_tokens"] == 256 and ar["decoder"] == "ar" and ar["samples_per_problem"] == 2
    capsys.readouterr()
    before = (out / "mbpp" / "ar" / "summary.json").stat().st_mtime_ns
    assert evaluate.main(["--decoder", "ar", *shared]) == 0          # finished configurations are skipped
    assert (out / "mbpp" / "ar" / "summary.json").stat().st_mtime_ns == before

    assert report.main(["--eval", str(out), "--run", str(run)]) == 0
    result = json.loads((out / "report.json").read_text())
    assert set(result["ratios"]) == {"humaneval", "mbpp"} and [s for s, _ in result["learning_curve"]] == [0, 2, 4, 6]
    assert result["verdict"].startswith(("undecided", "clearly", "holds", "fails", "mixed"))
    assert "## H2:" in (out / "report.md").read_text()
    assert report.main(["--eval", str(out), "--out", str(run / "diag"), "--no-verdict"]) == 0
    assert "## H2:" not in (run / "diag" / "report.md").read_text()
    assert json.loads((run / "diag" / "report.json").read_text())["verdict"].startswith("not evaluated")
