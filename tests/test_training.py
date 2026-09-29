"""Router training on a tiny synthetic feature cache: budgets, teacher, both stages and the CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from modalfidelity.router.oracle import solve_knapsack, uniform_costs
from modalfidelity.router.policy import RouterPolicy
from modalfidelity.router.reward import reward_matrix
from modalfidelity.training import stage1_dagger, stage2_scst
from modalfidelity.training.budgets import fixed_budgets, sample_budget
from modalfidelity.training.features import (FeatureDataset, collate, load_feature_shards,
                                             split_indices)


def write_synthetic_cache(root, videos=12, seed=0):
    """One shard in the mf-cache-features format, 12 videos of 6-15 windows by 6 speakers."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(6, 16, videos)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    feats = rng.standard_normal((offsets[-1], 2048)).astype(np.float16)
    labels = (rng.random((offsets[-1], 2)) < 0.3).astype(np.int8)
    files = [f"id{k // 2:05d}/clip/{k:05d}/real.mp4" for k in range(videos)]
    np.save(os.path.join(root, "shard0of1_feats.npy"), feats)
    np.savez(os.path.join(root, "shard0of1_meta.npz"), labels=labels, offsets=offsets,
             keys=np.array([f.replace("/", "_") for f in files]),
             modify_type=np.array(["real"] * videos), video_file=np.array(files),
             source=np.array(["/".join(f.split("/")[:3]) for f in files]))
    return root


def test_sampled_budgets_are_in_range():
    rng = np.random.default_rng(0)
    for t in (1, 5, 28, 132):
        b = sample_budget(t, rng, n=1000)
        assert b.min() >= 1 and b.max() <= int(np.ceil(0.6 * t))


def test_fixed_budgets():
    assert fixed_budgets(20, [0.05, 0.1, 0.5]).tolist() == [1, 2, 10]


def test_split_is_identity_disjoint(tmp_path):
    store = load_feature_shards(write_synthetic_cache(str(tmp_path)))
    train, held, _ = split_indices(store, fraction=0.3, seed=0)
    ids = lambda idx: {store["file"][i].split("/")[0] for i in idx}
    assert ids(train).isdisjoint(ids(held))
    assert len(train) + len(held) == len(store["keys"])


def test_teacher_tables_are_the_oracle(tmp_path):
    store = load_feature_shards(write_synthetic_cache(str(tmp_path)))
    ds = FeatureDataset(store, false_alarm=0.25, seed=0)
    item = ds[0]
    _, lab, _ = ds.video(0)
    reward = reward_matrix(lab[:, 1] > 0, lab[:, 0] > 0, 0.25)
    for b, table, best in zip(item["budgets"], item["expert_tables"], item["oracle_value"]):
        sol = solve_knapsack(reward, uniform_costs(len(lab)), int(b))
        assert np.array_equal(table, sol.actions.astype(np.int8))
        assert float(best) == pytest.approx(sol.optimal_value)


def test_one_epoch_of_each_stage_runs_and_learns(tmp_path):
    torch.manual_seed(0)
    store = load_feature_shards(write_synthetic_cache(str(tmp_path)))
    loader = DataLoader(FeatureDataset(store, seed=0), batch_size=4, collate_fn=collate)
    model = RouterPolicy()
    before = [p.detach().clone() for p in model.parameters()]
    s1 = stage1_dagger.train_epoch(model, loader, torch.optim.Adam(model.parameters(), 1e-3), "cpu")
    s2 = stage2_scst.train_epoch(model, loader, torch.optim.Adam(model.parameters(), 1e-4), "cpu",
                                 false_alarm=0.25)
    assert np.isfinite(s1["loss"]) and 0 <= s1["teacher_agreement"] <= 1
    assert np.isfinite(s2["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_train_router_cli(tmp_path):
    os.makedirs(tmp_path / "features")
    cache = write_synthetic_cache(str(tmp_path / "features"))
    out = tmp_path / "run"
    cmd = [sys.executable, "-m", "modalfidelity.cli.train_router", "--features", cache,
           "--out", str(out), "--heldout-sources", "", "--stage1-epochs", "1",
           "--stage2-epochs", "1", "--batch-size", "4", "--device", "cpu", "--workers", "0"]
    subprocess.run(cmd, check=True, capture_output=True)
    produced = sorted(os.listdir(out))
    assert any(f.startswith("stage1") for f in produced), produced
    assert "best.pt" in produced or any(f.startswith("stage2") for f in produced), produced
    # a second run into the same directory is refused
    assert subprocess.run(cmd, capture_output=True).returncode != 0
