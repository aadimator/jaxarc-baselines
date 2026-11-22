"""Simulator throughput: JaxARC (bare or training stack) and ARCLE, random actions.

strategy-v4 §14 (T1a-T1c). One call measures one (backend, dataset, interface,
stack, batch) point:

- **bare:** the env plus its action wrappers (``make_jaxarc_env`` with every
  observation wrapper off), vmapped, reset by hand when an episode ends.
- **training:** the exact Stoix wrapper chain the training runs use
  (``stoix.utils.make_env.make``: answer, input and context channels, episode
  metrics, auto-reset, vmap over the envs).
- **arcle:** ``RawARCEnv`` (raw ops, one trial per episode) behind a bbox
  wrapper, in a Gymnasium Sync or Async vector env, on the same canvas and pairs.

Random actions are drawn inside a ``lax.scan`` of ``steps`` steps per env. Compile
time is reported separately; the timed calls run after one warm-up call. With
several local devices the batch is split over them with ``pmap``; across hosts
(``--distributed``) every process runs its own devices and the totals combine
over the slowest process.

CLI: ``python -m jaxarc_baselines.throughput --backend jaxarc --dataset concept
--interface point --stack training --powers 0-16 --out FILE.jsonl``
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    # name: (hydra dataset, scenario, canvas)
    "mini": ("mini_arc", "Mini-all", 5),
    "concept": ("concept_arc", "Concept-all", 30),
}


def _config(dataset: str, interface: str, stack: str, num_envs: int):
    from hydra import compose, initialize_config_dir
    from jaxarc.registration.subset_loader import load_all_subsets_for_dataset
    from omegaconf import OmegaConf

    hydra_dataset, scenario, _ = DATASETS[dataset]
    load_all_subsets_for_dataset(scenario.split("-")[0])
    bare = stack == "bare"
    overrides = [
        f"dataset={hydra_dataset}",
        f"env.scenario.name={scenario}",
        f"env.action.mode={interface}",
        f"env.observation_wrappers.answer_grid={str(not bare).lower()}",
        f"env.observation_wrappers.input_grid={str(not bare).lower()}",
        f"env.observation_wrappers.contextual={str(not bare).lower()}",
        f"arch.total_num_envs={num_envs}",
    ]
    with initialize_config_dir(str(ROOT / "experiments" / "configs"), version_base="1.2"):
        config = compose("default_ppo_jaxarc", overrides=overrides)
    OmegaConf.set_struct(config, False)
    config.num_devices = 1
    config.arch.num_envs = num_envs
    return config


def jaxarc_env(dataset: str, interface: str, stack: str, num_envs: int):
    """The env to time and its action count. ``training`` is batched over num_envs."""
    config = _config(dataset, interface, stack, num_envs)
    if stack == "bare":
        from jaxarc.stoix_adapter import make_jaxarc_env

        env, _ = make_jaxarc_env(config)
    elif stack == "training":
        from stoix.utils import make_env as environments

        env, _ = environments.make(config=config)
    else:
        raise ValueError(f"Unknown stack {stack!r}; use 'bare' or 'training'")
    return env, int(env.action_space().num_values)


def _runner(env, stack: str, num_actions: int, num_envs: int, steps: int):
    """A function of one PRNG key that runs num_envs envs for `steps` random steps."""

    def bare(key):
        reset_key, loop_key = jax.random.split(key)
        state, ts = jax.vmap(env.reset)(jax.random.split(reset_key, num_envs))

        def body(carry, key):
            state, ts = carry
            k_reset, k_action = jax.random.split(key)
            fresh, fresh_ts = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
            last = ts.step_type >= 2
            pick = lambda new, old: jnp.where(
                last.reshape(last.shape + (1,) * (new.ndim - 1)), new, old
            )
            state = jax.tree.map(pick, fresh, state)
            action = jax.random.randint(k_action, (num_envs,), 0, num_actions)
            state, ts = jax.vmap(env.step)(state, action)
            return (state, ts), None

        (_, ts), _ = jax.lax.scan(body, (state, ts), jax.random.split(loop_key, steps))
        return ts.reward.sum()

    def training(key):
        reset_key, loop_key = jax.random.split(key)
        state, ts = env.reset(jax.random.split(reset_key, num_envs))

        def body(carry, key):
            state = carry
            action = jax.random.randint(key, (num_envs,), 0, num_actions)
            state, ts = env.step(state, action)
            return state, ts.reward.sum()

        _, rewards = jax.lax.scan(body, state, jax.random.split(loop_key, steps))
        return rewards.sum()

    return bare if stack == "bare" else training


def _block(x):
    jax.tree_util.tree_leaves(x)[0].block_until_ready()


def _summary(steps_timed: int, times: list[float]) -> dict:
    sps = [steps_timed / t for t in times]
    return {
        "steps_timed": steps_timed,
        "times_s": times,
        "sps_median": float(np.median(sps)),
        "sps_min": float(np.min(sps)),
        "sps_max": float(np.max(sps)),
    }


def combine_processes(steps_per_process: list[int], seconds_per_process: list[float]) -> float:
    """Total SPS of processes that ran at the same time: all steps over the slowest one."""
    return float(sum(steps_per_process) / max(seconds_per_process))


def measure_jaxarc(
    dataset: str,
    interface: str,
    stack: str,
    num_envs: int,
    steps: int = 128,
    repeats: int = 3,
    devices: int | None = None,
    seed: int = 0,
) -> dict:
    """Time JaxARC on ``num_envs`` envs, split over ``devices`` local devices (default 1)."""
    local = jax.local_devices()
    ndev = devices or 1
    if ndev > len(local) or num_envs % ndev:
        raise ValueError(f"num_envs={num_envs} over {ndev} of {len(local)} local devices")
    per_device = num_envs // ndev
    env, num_actions = jaxarc_env(dataset, interface, stack, per_device)
    run = _runner(env, stack, num_actions, per_device, steps)
    keys = jax.random.split(jax.random.PRNGKey(seed), ndev)

    t0 = time.perf_counter()
    if ndev == 1:
        compiled = jax.jit(run).lower(keys[0]).compile()
        call = lambda: compiled(keys[0])
    else:
        fn = jax.pmap(run, devices=local[:ndev])
        _block(fn(keys))  # pmap compiles on first call
        call = lambda: fn(keys)
    compile_s = time.perf_counter() - t0
    _block(call())  # warm-up
    times = timeit.repeat(lambda: _block(call()), number=1, repeat=repeats)
    return {
        "backend": "jaxarc",
        "dataset": dataset,
        "canvas": DATASETS[dataset][2],
        "interface": interface,
        "stack": stack,
        "num_envs": num_envs,
        "devices": ndev,
        "steps": steps,
        "num_actions": num_actions,
        "compile_s": compile_s,
        **_summary(num_envs * steps, times),
    }


# --- ARCLE ---------------------------------------------------------------------


def _pairs(dataset: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The demo pairs of every task in the dataset."""
    if dataset == "mini":
        root = ROOT / "data" / "MiniARC" / "data" / "MiniARC"
        files = sorted(root.glob("*.json"))
    else:
        root = ROOT.parent / "JaxARC" / "data" / "ConceptARC" / "corpus"
        files = sorted(root.glob("*/*.json"))
    pairs = []
    for f in files:
        for pair in json.loads(f.read_text())["train"]:
            pairs.append((np.array(pair["input"], np.int8), np.array(pair["output"], np.int8)))
    return pairs


class _Loader:
    """ARCLE loader that hands out the dataset's pairs in turn."""

    def __init__(self, pairs):
        self.data = [([i], [o], [], [], {"id": str(k)}) for k, (i, o) in enumerate(pairs)]
        self.next = 0

    def pick(self, data_index: int | None = None, **_kwargs):
        if data_index is None:
            data_index = self.next
            self.next = (self.next + 1) % len(self.data)
        return self.data[data_index]


def _arcle_env(pairs, canvas: int):
    import gymnasium as gym
    from arcle.envs import RawARCEnv
    from gymnasium import spaces

    class BBox(gym.ActionWrapper):
        """(r1, c1, r2, c2, op) -> ARCLE's dict action, as JaxARC's bbox mode."""

        def __init__(self, env):
            super().__init__(env)
            h, w = self.unwrapped.H, self.unwrapped.W
            n = len(self.unwrapped.operations)
            self.action_space = spaces.Tuple(
                (spaces.Discrete(h), spaces.Discrete(w), spaces.Discrete(h), spaces.Discrete(w), spaces.Discrete(n))
            )

        def action(self, action):
            r1, c1, r2, c2, op = action
            selection = np.zeros((self.unwrapped.H, self.unwrapped.W), dtype=np.int8)
            selection[min(r1, r2) : max(r1, r2) + 1, min(c1, c2) : max(c1, c2) + 1] = 1
            return {"selection": selection, "operation": op}

    def make():
        env = RawARCEnv(
            data_loader=_Loader(pairs), max_grid_size=(canvas, canvas), max_trial=1, render_mode=None
        )
        return gym.wrappers.TimeLimit(BBox(env), max_episode_steps=150)

    return make


def measure_arcle(
    dataset: str, num_envs: int, steps: int = 128, repeats: int = 3, mode: str = "sync"
) -> dict:
    """Time ARCLE on ``num_envs`` envs (Gymnasium Sync or Async vector env)."""
    import gymnasium as gym

    canvas = DATASETS[dataset][2]
    make = _arcle_env(_pairs(dataset), canvas)
    t0 = time.perf_counter()
    vector = {"sync": gym.vector.SyncVectorEnv, "async": gym.vector.AsyncVectorEnv}[mode]
    env = vector([make for _ in range(num_envs)])
    env.reset(seed=0)
    env.action_space.seed(0)
    setup_s = time.perf_counter() - t0

    def run():
        for _ in range(steps):
            env.step(env.action_space.sample())

    run()  # warm-up
    times = timeit.repeat(run, number=1, repeat=repeats)
    env.close()
    return {
        "backend": "arcle",
        "dataset": dataset,
        "canvas": canvas,
        "interface": "bbox",
        "stack": f"gym-{mode}",
        "num_envs": num_envs,
        "devices": 0,
        "steps": steps,
        "setup_s": setup_s,
        **_summary(num_envs * steps, times),
    }


# --- CLI -------------------------------------------------------------------------


def _powers(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def system_info() -> dict:
    info = {
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.local_devices()],
    }
    try:
        model = [l for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")]
        info["cpu"] = model[0].split(":", 1)[1].strip()
        info["cpu_count"] = len(model)
    except OSError:
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("jaxarc", "arcle"), default="jaxarc")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--interface", choices=("point", "bbox"), default="point")
    parser.add_argument("--stack", choices=("bare", "training"), default="training")
    parser.add_argument("--mode", choices=("sync", "async"), default="sync", help="ARCLE vector env")
    parser.add_argument("--powers", default="0-12", help="batch sizes 2^lo..2^hi (total, over every device)")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", type=int, default=1, help="local devices to pmap over")
    parser.add_argument("--distributed", action="store_true", help="jax.distributed over SLURM tasks")
    parser.add_argument("--max-seconds", type=float, default=600, help="stop the sweep after a point this slow")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.distributed:
        from jax.experimental import multihost_utils

        jax.distributed.initialize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    info = system_info()
    for power in _powers(args.powers):
        n = 2**power
        try:
            if args.backend == "arcle":
                row = measure_arcle(args.dataset, n, args.steps, args.repeats, args.mode)
            else:
                if n % args.devices:
                    continue
                if args.distributed:
                    multihost_utils.sync_global_devices(f"start-{power}")
                row = measure_jaxarc(
                    args.dataset, args.interface, args.stack, n, args.steps, args.repeats,
                    devices=args.devices, seed=jax.process_index(),
                )
                if args.distributed:
                    times = multihost_utils.process_allgather(np.array(row["times_s"]))
                    per_call = times.max(axis=0)  # every process ran its call together
                    row["processes"] = jax.process_count()
                    row["num_envs_total"] = n * jax.process_count()
                    row["sps_total_median"] = combine_processes(
                        [row["steps_timed"]] * jax.process_count(), [float(np.median(per_call))]
                    )
        except Exception as error:  # noqa: BLE001 - an OOM is a result, not a crash
            row = {"backend": args.backend, "dataset": args.dataset, "num_envs": n,
                   "interface": args.interface, "stack": args.stack, "devices": args.devices,
                   "error": f"{type(error).__name__}: {str(error)[:300]}"}
        row["system"] = info
        if jax.process_index() == 0:
            with args.out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "system"}), flush=True)
        if "error" in row and "RESOURCE_EXHAUSTED" in row["error"]:
            break
        if "times_s" in row and np.median(row["times_s"]) > args.max_seconds:
            break


if __name__ == "__main__":
    main()
