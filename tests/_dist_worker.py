"""Plain (non-pytest) helper module for test_distributed.py.

Must live outside any `test_*.py` file: pytest's assertion-rewriting import hook
instruments test modules, and re-importing an instrumented module from a
`multiprocessing` "spawn" child can hang instead of raising -- so the actual
worker function that gets spawned has to live in an ordinary module like this one.
"""
from __future__ import annotations

import os


def rank_main(rank: int, world_size: int, roles, q) -> None:
    # Each spawned process otherwise lets torch/OpenMP/MKL grab every visible
    # core for its internal thread pool; with several processes doing that at
    # once on a small CPU box they oversubscribe and stall each other's import
    # for a long time. Pin everything to 1 thread per rank (mirrors what
    # trader/resources.py's cpu_threads budget does for real training runs on a
    # shared/contended box).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)

    from trader.resources import Plan
    from trader.train.loop import AGENT_METRIC_KEYS, AGENTS, Trainer

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size,
                             init_method="tcp://127.0.0.1:29511")
    try:
        plan = Plan(world_size=world_size, backend="gloo", device_type="cpu",
                    size_preset="tiny", roles=roles, devices=["cpu"] * world_size,
                    cpu_threads=1, gpus=[], host_ram_gb=1.0, host_cpus=1, notes=[])

        # Bare-bones stand-in for a Trainer: _reduce_metrics only touches these
        # attributes, so there's no need to build real models/data for this test.
        tr = Trainer.__new__(Trainer)
        tr.world, tr.rank, tr.device, tr.plan = world_size, rank, torch.device("cpu"), plan
        tr.owned = set(roles[rank])
        tr.groups, tr.src = {}, {}
        for a in AGENTS:
            owners = plan.owners_of(a)
            tr.src[a] = owners[0] if owners else 0
            tr.groups[a] = dist.new_group(ranks=owners) if len(owners) > 1 else None

        # Exactly what train_epoch() leaves in `metrics`: common keys every rank
        # always has, plus only the agent-specific keys for what this rank owns.
        m = {"equity_mean": 10.0 + rank, "return_pct": -1.0, "epoch_sec": 0.1}
        for a in AGENTS:
            if a in tr.owned:
                for k in AGENT_METRIC_KEYS[a]:
                    m[k] = 1.0

        out = tr._reduce_metrics(m)
        dist.barrier()
        q.put((rank, sorted(out.keys())))
    finally:
        dist.destroy_process_group()
