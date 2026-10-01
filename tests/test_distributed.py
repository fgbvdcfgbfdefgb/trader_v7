"""Regression test for the heterogeneous multi-rank metric reduction.

Background: on a real multi-GPU box, resources.assign_roles() gives each rank a
*different* agent (rank0: predictor, rank1: analyst, rank2+3: trader). The first
real 4xGPU training run hung indefinitely inside Trainer._reduce_metrics, because
that function built its all_reduce tensor from each rank's *local* metrics-dict
keys -- and those keys differ across ranks since each rank only computes the
agent(s) it owns. Different tensor sizes on different ranks is a classic NCCL
collective-mismatch bug: it manifests as a 10-minute hang then a watchdog
timeout, not a clean crash, which makes it dangerous and easy to miss.

This test reproduces exactly that rank/role layout with the lightweight `gloo`
CPU backend (no GPU needed) and asserts that every rank finishes promptly and
that every rank ends up with every expected metric key (including the ones
owned by other ranks), with a hard wall-clock bound so a regression fails fast
instead of hanging the test suite.

The actual per-rank worker lives in tests/_dist_worker.py, a plain module, not
in this file: pytest's assertion-rewrite import hook instruments test modules,
and `multiprocessing`'s "spawn" context re-imports the target function's module
from scratch in the child -- doing that for an instrumented test module reliably
hangs instead of erroring, so the spawn target must come from an ordinary module.
"""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._dist_worker import rank_main  # noqa: E402

JOIN_TIMEOUT = 60  # seconds; generous for CI (slow first-import on a cold, CPU-
                    # starved box), still far below the old 600s NCCL hang


def test_reduce_metrics_heterogeneous_roles_does_not_hang():
    # 2 ranks with disjoint agents (== resources.assign_roles(2) on a real 2-GPU
    # box) is already enough to reproduce the original bug: rank 0's local
    # metrics dict has predictor+analyst keys, rank 1's has only trader keys, so
    # the old code's all_reduce tensor sizes differed between ranks. Kept at 2
    # (rather than mirroring the 4-GPU plan that actually hung in production)
    # so this test is cheap enough to run reliably on small/shared CI boxes too.
    world_size = 2
    roles = [["predictor", "analyst"], ["trader"]]  # == assign_roles(2)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=rank_main, args=(r, world_size, roles, q))
             for r in range(world_size)]
    for p in procs:
        p.start()
        # Stagger spawns: on CPU-starved boxes (e.g. a 2 vCPU sandbox) four
        # concurrent cold imports of torch/pandas/pyarrow/matplotlib can stall
        # each other out for minutes even though nothing is truly deadlocked.
        # This is irrelevant on a real multi-core training box but makes the
        # test itself reliable everywhere.
        time.sleep(4.0)
    for p in procs:
        p.join(timeout=JOIN_TIMEOUT)

    stuck = [p.pid for p in procs if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.terminate()
    assert not stuck, (
        "a rank did not finish within the timeout -- _reduce_metrics likely "
        "hung again on mismatched collective calls across heterogeneous ranks")
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    expected = {
        "equity_mean", "return_pct", "epoch_sec",
        "pred_nll", "pred_ce", "pred_dir_acc", "pred_ic",
        "an_ce", "an_vol", "an_align", "an_risk", "an_acc", "an_conf",
        "pg_loss", "v_loss", "entropy", "kl", "clipfrac", "grad_norm", "ppo_epochs_run",
    }
    seen = {}
    while not q.empty():
        rank, keys = q.get_nowait()
        seen[rank] = set(keys)
    assert set(seen) == set(range(world_size))
    for rank, keys in seen.items():
        assert keys == expected, f"rank {rank}: {keys ^ expected}"
