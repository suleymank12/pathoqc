"""Uçtan uca duman testi: minik veriyle 1 epoch eğitim hatasız dönüyor mu, kaldığı yerden devam ediyor mu."""

import json

import train_stage1


def test_one_epoch_then_resume(tiny_dataset, tmp_path):
    out_dir = tmp_path / "out"
    common = ["--dataset", "luad", "--data-root", str(tiny_dataset), "--out-dir", str(out_dir),
              "--batch-size", "4", "--workers", "0", "--no-pretrained"]

    first = train_stage1.main(common + ["--epochs", "1"])
    run_dir = out_dir / "luad_stage1_seed0"
    assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file()
    assert len(first["history"]) == 1
    assert 0.0 <= first["history"][0]["val"]["miou"] <= 100.0

    # Aynı klasörle 2 epoch istenince yalnızca eksik olan 2. epoch koşmalı.
    second = train_stage1.main(common + ["--epochs", "2"])
    assert [r["epoch"] for r in second["history"]] == [1, 2]
    saved = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(saved["history"]) == 2
