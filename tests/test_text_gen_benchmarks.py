from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmarks import text_gen_benchmarks as tgb


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def benchmark_results() -> dict:
    return tgb.run_text_generation_benchmarks(quick=True)


def test_random_baseline_runs() -> None:
    baseline = tgb.RandomBaseline(seed=1).fit("abcabc")

    generated = baseline.generate("a", max_tokens=12)

    assert len(generated) == 12
    assert set(generated).issubset(set("abc"))


def test_frequency_baseline_learns() -> None:
    baseline = tgb.FrequencyBaseline().fit("aaabbbcaaaa")

    assert baseline.most_common_char == "a"
    assert baseline.predict_next("zzz") == "a"


def test_bigram_baseline_learns() -> None:
    baseline = tgb.BigramBaseline().fit("abababab")

    assert baseline.predict_next("a") == "b"
    assert baseline.predict_probability("a", "b") > baseline.predict_probability("a", "a")


def test_benchmark_suite_runs(benchmark_results: dict) -> None:
    assert "character_prediction_accuracy" in benchmark_results
    assert "summary_table" in benchmark_results


def test_bioarn_beats_random(benchmark_results: dict) -> None:
    accuracy = benchmark_results["character_prediction_accuracy"]["mean_accuracy"]

    assert accuracy["bioarn"] > accuracy["random"]


def test_pattern_learning(benchmark_results: dict) -> None:
    abc_case = benchmark_results["pattern_learning"]["cases"]["abc-cycle"]["bioarn"]

    assert abc_case["score"] >= 0.40
    assert "abc" in abc_case["generated"]


def test_few_shot_advantage(benchmark_results: dict) -> None:
    one_shot = benchmark_results["few_shot_learning"]["shots"]["1"]

    assert one_shot["bioarn"] > one_shot["bigram"]


def test_continual_advantage(benchmark_results: dict) -> None:
    forgetting = benchmark_results["continual_learning"]["bioarn"]["forgetting"]

    assert forgetting < 0.10


def test_diversity_metrics(benchmark_results: dict) -> None:
    for model_metrics in benchmark_results["generation_diversity"].values():
        for key in ("unique_bigram_ratio", "unique_trigram_ratio", "vocabulary_coverage", "repetition_rate"):
            assert 0.0 <= model_metrics[key] <= 1.0


def test_results_json_valid(benchmark_results: dict) -> None:
    path = Path(tgb.RESULTS_PATH)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["timestamp"]
    assert payload["config"]["quick"] is True
    assert payload["summary_table"]["bioarn"]["next_char_accuracy_ctx8"] == pytest.approx(
        benchmark_results["summary_table"]["bioarn"]["next_char_accuracy_ctx8"]
    )


def test_report_generated() -> None:
    path = Path(tgb.REPORT_PATH)

    assert path.exists()
    assert "Bio-ARN Text Generation Benchmarks" in path.read_text(encoding="utf-8")


def test_benchmarks_all_metrics(benchmark_results: dict) -> None:
    expected = {
        "character_prediction_accuracy",
        "approximate_perplexity",
        "generation_diversity",
        "pattern_learning",
        "few_shot_learning",
        "continual_learning",
        "summary_table",
    }

    assert expected.issubset(benchmark_results)
