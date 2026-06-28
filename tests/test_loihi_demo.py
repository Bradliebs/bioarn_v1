from __future__ import annotations

from pathlib import Path

from experiments.loihi_e2e_demo import LoihiDemoConfig, LoihiEndToEndDemo


def test_loihi_end_to_end_demo_smoke(tmp_path: Path) -> None:
    demo = LoihiEndToEndDemo(
        LoihiDemoConfig(
            train_samples=240,
            eval_samples=40,
            camera_frames=24,
            seed=31,
            pca_dim=96,
            concept_dim=192,
            max_pool_size=128,
            preprocessing_warmup_samples=64,
            output_dir=tmp_path / "loihi-demo",
        )
    )

    report = demo.run_full_demo()
    rendered = report.render()

    assert report.training["accuracy"] >= 0.15
    assert report.export["weight_fidelity"] >= 0.999
    assert abs(float(report.comparison["bio_accuracy"]) - float(report.comparison["lava_accuracy"])) <= 0.01
    assert len(report.frame_results) == 24
    assert "Bio-ARN → Loihi 2 End-to-End Demo" in rendered
    assert "Energy Comparison" in rendered
    assert (tmp_path / "loihi-demo" / "loihi_demo_report.txt").exists()
