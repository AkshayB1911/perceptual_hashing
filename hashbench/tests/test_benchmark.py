from pathlib import Path

import pandas as pd
import pytest

from hashbench import benchmark

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
ARGS = dict(algorithms=["phash", "pdq"], attacks=["jpeg", "rotate"], limit=None, max_side=128,
            max_pairs=1000, workers=1, seed=0)


def _outputs(out: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(out / f"{name}.csv") for name in ("genuine", "impostor", "hashes")}


def test_resume_after_interruption_gives_identical_outputs(tmp_path, monkeypatch):
    benchmark.run(SAMPLES, tmp_path / "full", checkpoint_every=4, **ARGS)
    expected = _outputs(tmp_path / "full")

    # Simulate a kill after two chunks have been saved.
    save = benchmark._save_chunk
    saved = []

    def save_then_die(ckpt_dir, k, jobs, results):
        if len(saved) == 2:
            raise KeyboardInterrupt
        save(ckpt_dir, k, jobs, results)
        saved.append(k)

    monkeypatch.setattr(benchmark, "_save_chunk", save_then_die)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(SAMPLES, tmp_path / "resumed", checkpoint_every=4, **ARGS)
    assert len(list((tmp_path / "resumed" / "checkpoints").glob("chunk_*.pkl"))) == 2
    assert not (tmp_path / "resumed" / "genuine.csv").exists()

    monkeypatch.setattr(benchmark, "_save_chunk", save)
    benchmark.run(SAMPLES, tmp_path / "resumed", checkpoint_every=4, **ARGS)
    for name, df in _outputs(tmp_path / "resumed").items():
        pd.testing.assert_frame_equal(df, expected[name], obj=name)
    assert not (tmp_path / "resumed" / "checkpoints").exists()


def test_refuses_checkpoints_of_another_config(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_save_chunk", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(SAMPLES, tmp_path, checkpoint_every=4, **ARGS)
    with pytest.raises(SystemExit, match="different configuration"):
        benchmark.run(SAMPLES, tmp_path, checkpoint_every=4, **{**ARGS, "algorithms": ["phash"]})
