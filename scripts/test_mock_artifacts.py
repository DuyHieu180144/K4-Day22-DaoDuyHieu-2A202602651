import hashlib
import json
import pyarrow.parquet as pq

from scripts.generate_mock_artifacts import generate
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_mock_bundle_has_lab_shapes_and_is_explicitly_synthetic():
    output = REPO / "submission" / "mock-artifacts"
    generate(output)

    manifest = json.loads((output / "MOCK_ONLY.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert manifest["not_training_or_evaluation_evidence"] is True

    train = pq.read_table(output / "data/pref/train.parquet").to_pylist()
    heldout = pq.read_table(output / "data/pref/eval.parquet").to_pylist()
    assert len(train) == 800
    assert len(heldout) == 100
    assert {"prompt", "chosen", "rejected"} <= train[0].keys()
    assert not (
        {row["prompt"][0]["content"] for row in train}
        & {row["prompt"][0]["content"] for row in heldout}
    )

    lines = (output / "data/eval/side_by_side.jsonl").read_bytes()
    records = [json.loads(line) for line in lines.splitlines()]
    assert len(records) == 58
    assert sum(row["category"] == "heldout" for row in records) == 50
    assert all(row["synthetic"] is True for row in records)

    summary = json.loads((output / "data/eval/judge_summary.json").read_text(encoding="utf-8"))
    assert summary["outputs_sha256"] == hashlib.sha256(lines).hexdigest()
    assert summary["synthetic"] is True
    assert summary["heldout"]["n"] == 50

    stats = json.loads((output / "data/pref/stats.json").read_text(encoding="utf-8"))
    assert stats["n"] == 800
    assert stats["length_unit"].startswith("whitespace word count")

    metrics = json.loads((output / "adapters/dpo/dpo_metrics.json").read_text(encoding="utf-8"))
    assert metrics["synthetic"] is True
    assert metrics["diagnosis"] == "MOCK_ONLY"
    assert {"end_chosen_reward", "end_rejected_reward", "end_reward_gap", "eval_reward_accuracy"} <= metrics.keys()
    assert all((output / "submission/screenshots" / f"{name}.png").is_file() for name in (
        "02-sft-loss", "02b-pref-length", "03-dpo-reward-curves", "04-side-by-side-table"
    ))
