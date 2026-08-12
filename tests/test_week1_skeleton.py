"""Hafta 1 iskeletinin `pytest` ile çalıştırılabilir versiyonu.
`python -m src.X.Y` çağırıp exit code kontrol etmek yerine, CI'da da
çalışacak gerçek assertion'lar burada.

Çalıştırma: pytest tests/test_week1_skeleton.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.manifest import (
    add_row,
    check_split_leakage,
    load_manifest,
    new_manifest,
    save_manifest,
    summarize,
)
from src.detectors.base import BaseDetector, DetectorOutput
from src.detectors.metadata import MetadataDetector
from src.eval.metrics import (
    compute_image_level_metrics,
    fp_area_rate,
    pixel_f1_iou,
    tpr_at_fpr,
)
from src.eval.report import run_and_report


class DummyDetector(BaseDetector):
    name = "dummy_for_tests"

    def predict(self, image_path):
        return DetectorOutput(score=0.5)


def test_detector_output_validates_score_range():
    with pytest.raises(ValueError):
        DetectorOutput(score=1.5)


def test_detector_output_validates_mask_shape():
    with pytest.raises(ValueError):
        DetectorOutput(score=0.5, mask=np.zeros((3, 4, 5)))


def test_base_detector_predict_batch_default():
    d = DummyDetector()
    results = d.predict_batch(["a.jpg", "b.jpg", "c.jpg"])
    assert len(results) == 3
    assert all(isinstance(r, DetectorOutput) for r in results)


def test_manifest_roundtrip(tmp_path):
    df = new_manifest()
    df = add_row(
        df, source_image_id="x1", path="a.jpg", label="real",
        width=100, height=100, split="train",
    )
    out_path = tmp_path / "m.parquet"
    save_manifest(df, out_path)
    loaded = load_manifest(out_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["label"] == "real"


def test_manifest_rejects_invalid_label():
    df = new_manifest()
    with pytest.raises(ValueError):
        add_row(
            df, source_image_id="x1", path="a.jpg", label="not_a_real_label",
            width=100, height=100,
        )


def test_split_leakage_detection():
    df = new_manifest()
    df = add_row(df, source_image_id="dup", path="a.jpg", label="real",
                 width=10, height=10, split="train")
    df = add_row(df, source_image_id="dup", path="b.jpg", label="real",
                 width=10, height=10, split="test")
    problems = check_split_leakage(df)
    assert len(problems) == 1
    assert "SIZINTI" in problems[0]


def test_split_leakage_clean_case():
    df = new_manifest()
    df = add_row(df, source_image_id="a", path="a.jpg", label="real",
                 width=10, height=10, split="train")
    df = add_row(df, source_image_id="b", path="b.jpg", label="real",
                 width=10, height=10, split="test")
    assert check_split_leakage(df) == []


def test_tpr_at_fpr_bounds():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_score = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, size=200), 0, 1)
    val = tpr_at_fpr(y_true, y_score, 0.05)
    assert 0.0 <= val <= 1.0


def test_compute_image_level_metrics_rejects_single_class():
    with pytest.raises(ValueError):
        compute_image_level_metrics(np.zeros(10), np.random.rand(10))


def test_pixel_f1_iou_perfect_match():
    mask = np.zeros((10, 10))
    mask[2:5, 2:5] = 1.0
    result = pixel_f1_iou(mask, mask)
    assert result["pixel_f1"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)


def test_fp_area_rate_zero_on_empty_mask():
    assert fp_area_rate(np.zeros((10, 10))) == 0.0


def test_metadata_detector_flags_missing_exif(tmp_path):
    path = tmp_path / "no_exif.jpg"
    Image.new("RGB", (50, 50)).save(path, "JPEG")
    detector = MetadataDetector()
    result = detector.predict(path)
    assert result.meta["exif_present"] is False
    assert result.score > 0


def test_run_and_report_end_to_end(tmp_path):
    df = new_manifest()
    for i in range(40):
        label = "real" if i % 2 == 0 else "fully_synthetic"
        profile = "clean" if (i // 2) % 2 == 0 else "whatsapp"
        df = add_row(
            df, source_image_id=f"img_{i}", path=f"fake/{label}_{i}.jpg",
            label=label, width=64, height=64, split="test",
            launder_profile=profile,
        )
    manifest_path = tmp_path / "manifest.parquet"
    save_manifest(df, manifest_path)

    class BiasedDummy(BaseDetector):
        name = "biased_dummy"

        def predict(self, image_path):
            score = 0.8 if "synthetic" in str(image_path) else 0.2
            return DetectorOutput(score=score)

    result = run_and_report(
        detector=BiasedDummy(),
        manifest_path=manifest_path,
        experiment_dir=tmp_path / "exp",
        split="test",
    )
    assert "clean" in result["results_by_launder_profile"]
    assert "whatsapp" in result["results_by_launder_profile"]
    for profile_metrics in result["results_by_launder_profile"].values():
        assert profile_metrics["roc_auc"] > 0.9  # bariz ayrılabilir sahte veri
