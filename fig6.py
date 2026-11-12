# -*- coding: utf-8 -*-
"""Figure 6: mean cooperator fractions with the original lines and markers.

Each mean is calculated from the independent runs' late-window averages. Use --plot-only with a saved figure6_raw_runs.npz to replot.
Dependencies: numpy, numba, matplotlib, tqdm.
"""
import os

# 1. 解决 JIT 线程限制缺失问题：在导入 numpy/numba 之前全局应用环境变量，确保子进程不产生线程争用
for _name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from numba import njit
from tqdm import tqdm
import multiprocessing as mp

# ==========================================
# 1. 参数设定与模型常数
# ==========================================
L = 100
N = L * L
K = 0.1  # 费米公式中的噪声 K
EPSILON = 0.01  # 正则化常数 epsilon

MCS_TOTAL = 20000 # 模拟总步数
MCS_TRANSIENT = 17000 # 暂态步数
NUM_RUNS = 100  # 独立运行次数

# 固定 alpha, beta 和 u。横坐标保留为 b，图例的多条线改为不同的 gamma
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_U = 0.00

GAMMA_VALUES = [0.00, 0.10, 0.20, 0.50, 1.00]
# b 的区间为 [1.0, 5.0]，取 81 个点（保持步长 0.05）
B_VALUES = np.linspace(1.0, 5.0, 81)
BASE_SEED = 20261007  # 固定基础种子以便复现


# ==========================================
# 2. 核心逻辑函数
# ==========================================

# 2. 解决 JIT 缓存缺失问题：使用 @njit(cache=True)
# Model update functions used by Figure 6.
PROTOCOL_ID = 'replacement_current_global_end_mcs_ema_v1'


@njit(cache=True)
def current_weights(strategies, reputations, epsilon):
    total, cooperative = 0.0, 0.0
    for x in range(len(strategies)):
        weight = reputations[x] + epsilon
        total += weight
        if strategies[x] == 1:
            cooperative += weight
    return total, cooperative


@njit(cache=True)
def fermi_probability(focal_fitness, neighbor_fitness, noise):
    # Stable logistic: no hard truncation of finite probabilities.
    exponent = (focal_fitness - neighbor_fitness) / noise
    if exponent >= 0.0:
        value = np.exp(-exponent)
        return value / (1.0 + value)
    return 1.0 / (1.0 + np.exp(exponent))


@njit(cache=True)
def fixed_fitness(x, strategies, reputations, neighbors, total, cooperative,
                  alpha, beta, epsilon, reward, sucker, temptation, punishment):
    strategy = strategies[x]
    numerator = reputations[x] + epsilon
    denominator = numerator
    payoff = 0.0
    for edge in range(neighbors.shape[1]):
        y = neighbors[x, edge]
        weight = reputations[y] + epsilon
        denominator += weight
        if strategies[y] == strategy:
            numerator += weight
        if strategy == 1:
            payoff += reward if strategies[y] == 1 else sucker
        else:
            payoff += temptation if strategies[y] == 1 else punishment
    if beta == 0.0:
        return payoff
    # Only remove roundoff outside [0,1], not genuine small support.
    global_c = min(1.0, max(0.0, cooperative / total))
    support = alpha * numerator / denominator + (1.0 - alpha) * (
        global_c if strategy == 1 else 1.0 - global_c)
    return payoff * support ** beta


@njit(cache=True)
def fixed_event(x, strategies, reputations, neighbors, total, cooperative,
                noise, alpha, beta, epsilon, u, reward, sucker, temptation, punishment):
    old = strategies[x]
    new = old
    if np.random.random() < u:
        new = np.random.randint(0, 2)
    else:
        y = neighbors[x, np.random.randint(0, neighbors.shape[1])]
        if old != strategies[y]:
            focal = fixed_fitness(x, strategies, reputations, neighbors, total,
                                  cooperative, alpha, beta, epsilon,
                                  reward, sucker, temptation, punishment)
            target = fixed_fitness(y, strategies, reputations, neighbors, total,
                                   cooperative, alpha, beta, epsilon,
                                   reward, sucker, temptation, punishment)
            if np.random.random() < fermi_probability(focal, target, noise):
                new = strategies[y]
    if new != old:
        strategies[x] = new
        cooperative += (reputations[x] + epsilon) * (new - old)
    return cooperative


@njit(cache=True)
def mcs_fixed(strategies, reputations, neighbors, noise, b, alpha, beta,
              gamma, epsilon, u, reward=1.0, sucker=0.0, punishment=0.0):
    total, cooperative = current_weights(strategies, reputations, epsilon)
    for _ in range(len(strategies)):
        x = np.random.randint(0, len(strategies))
        cooperative = fixed_event(x, strategies, reputations, neighbors,
                                  total, cooperative, noise, alpha, beta,
                                  epsilon, u, reward, sucker, b, punishment)
    for x in range(len(strategies)):
        reputations[x] = (1.0 - gamma) * reputations[x] + gamma * strategies[x]


def prepare_run_directory(script_path, parameters=None):
    """Keep new relative outputs separate from all historical result files."""
    from pathlib import Path
    from datetime import datetime
    import os, json, hashlib, platform, sys
    import numba
    script = Path(script_path).resolve()
    root = script.parent / ('results_' + PROTOCOL_ID)
    output = root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol=PROTOCOL_ID, source=str(script), arguments=sys.argv[1:],
                    source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                    core_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python=platform.python_version(), numpy=np.__version__, numba=numba.__version__)
    if parameters is not None:
        captured = {}
        for name, value in parameters.items():
            if name.isupper() and isinstance(value, (str, int, float, bool, list, tuple, Path, np.ndarray)):
                captured[name] = value.tolist() if isinstance(value, np.ndarray) else str(value) if isinstance(value, Path) else value
        manifest['script_defaults'] = captured
    (output / 'protocol_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


@njit(cache=True)
def build_neighbor_map(L):
    """ 构建 von Neumann 邻域 (上下左右) """
    N = L * L
    neighbors = np.empty((N, 4), dtype=np.int64)
    for i in range(N):
        row, col = i // L, i % L
        neighbors[i, 0] = ((row - 1 + L) % L) * L + col  # North
        neighbors[i, 1] = ((row + 1) % L) * L + col  # South
        neighbors[i, 2] = row * L + ((col - 1 + L) % L)  # West
        neighbors[i, 3] = row * L + ((col + 1) % L)  # East
    return neighbors


@njit(cache=True)
def calculate_payoff(s_i, s_j, b):
    """
    收益矩阵 M:
    C(1) D(0)
    C 1    0
    D b    0
    """
    if s_i == 1:  # Si = C
        return 1.0 if s_j == 1 else 0.0
    else:  # Si = D
        return b if s_j == 1 else 0.0


@njit(cache=True)
def simulation_step_optimized(N, K, b, alpha, beta, gamma, epsilon, u, strategies, reputations, neighbor_map):
    mcs_fixed(strategies, reputations, neighbor_map, K, b,
              alpha, beta, gamma, epsilon, u)
    return strategies, reputations


@njit(cache=True)
def run_single_experiment(N, K, b, alpha, beta, gamma, epsilon, u, neighbor_map, seed, mcs_total, mcs_transient):
    # 重置单个任务的 Numpy 种子，供模拟中使用
    np.random.seed(seed)

    strategies = (np.random.rand(N) < 0.5).astype(np.int64)
    reputations = np.full(N, 0.5, dtype=np.float64)

    cooperation_sum = 0.0
    sample_count = mcs_total - mcs_transient

    for mcs in range(mcs_total):
        strategies, reputations = simulation_step_optimized(
            N, K, b, alpha, beta, gamma, epsilon, u,
            strategies, reputations, neighbor_map
        )

        cooperation = np.mean(strategies)
        if mcs >= mcs_transient:
            cooperation_sum += cooperation

        # 3. 解决非必要执行完整 20000 MCS 问题 (Early stopping)：
        # 当突变率 u=0 且模型达到全合作或全背叛的吸收态时，直接分析计算剩余周期的采样平均
        if u == 0.0 and (cooperation == 0.0 or cooperation == 1.0):
            remaining_samples = mcs_total - max(mcs + 1, mcs_transient)
            if remaining_samples > 0:
                cooperation_sum += remaining_samples * cooperation
            return cooperation_sum / sample_count

    return cooperation_sum / sample_count


# ==========================================
# 3. Statistical settings and worker helpers
# ==========================================
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json

CPU_FRACTION = 0.60
OUTPUT_ROOT = Path(__file__).resolve().parent / "figure6_mean_results"

_WORKER_CONFIG = None
_WORKER_NEIGHBORS = None


def default_config():
    """Production settings: identical to the supplied simulation."""
    return {
        "MODEL_PROTOCOL": PROTOCOL_ID,
        "L": int(L), "K": float(K), "EPSILON": float(EPSILON),
        "MCS_TOTAL": int(MCS_TOTAL), "MCS_TRANSIENT": int(MCS_TRANSIENT),
        "NUM_RUNS": int(NUM_RUNS), "FIXED_ALPHA": float(FIXED_ALPHA),
        "FIXED_BETA": float(FIXED_BETA), "FIXED_U": float(FIXED_U),
        "GAMMA_VALUES": [float(value) for value in GAMMA_VALUES],
        "B_VALUES": [float(value) for value in B_VALUES],
        "BASE_SEED": int(BASE_SEED),
    }


def validate_config(config):
    if config["L"] < 2 or config["NUM_RUNS"] < 2:
        raise ValueError("L must be >= 2 and NUM_RUNS must be >= 2.")
    if not 0 <= config["MCS_TRANSIENT"] < config["MCS_TOTAL"]:
        raise ValueError("Require 0 <= MCS_TRANSIENT < MCS_TOTAL.")
    if config["K"] <= 0 or config["EPSILON"] <= 0:
        raise ValueError("K and EPSILON must be positive.")
    if not 0 <= config["FIXED_U"] <= 1:
        raise ValueError("FIXED_U must lie in [0, 1].")
    if not config["GAMMA_VALUES"] or not config["B_VALUES"]:
        raise ValueError("The gamma and b grids must not be empty.")
    if not all(0 <= value <= 1 for value in config["GAMMA_VALUES"]):
        raise ValueError("GAMMA_VALUES must lie in [0, 1].")


def initialize_worker(config):
    global _WORKER_CONFIG, _WORKER_NEIGHBORS
    _WORKER_CONFIG = config
    _WORKER_NEIGHBORS = build_neighbor_map(config["L"])


def execute_task(task):
    gamma_index, b_index, run_index, gamma, b, seed = task
    config = _WORKER_CONFIG
    value = run_single_experiment(
        config["L"] * config["L"], config["K"], b,
        config["FIXED_ALPHA"], config["FIXED_BETA"], gamma,
        config["EPSILON"], config["FIXED_U"],
        _WORKER_NEIGHBORS, seed,
        config["MCS_TOTAL"], config["MCS_TRANSIENT"],
    )
    return gamma_index, b_index, run_index, value


def make_tasks(config):
    """Preserve the original task order and SeedSequence seed mapping."""
    gamma_array = np.asarray(config["GAMMA_VALUES"], dtype=np.float64)
    b_array = np.asarray(config["B_VALUES"], dtype=np.float64)
    shape = (len(gamma_array), len(b_array), config["NUM_RUNS"])
    total_tasks = int(np.prod(shape))
    child_sequences = np.random.SeedSequence(config["BASE_SEED"]).spawn(total_tasks)
    seeds = np.empty(shape, dtype=np.uint32)
    tasks = []
    task_number = 0
    for g_idx, gamma in enumerate(gamma_array):
        for b_idx, b in enumerate(b_array):
            for r_idx in range(config["NUM_RUNS"]):
                seed = int(child_sequences[task_number].generate_state(1)[0]) % (2 ** 32)
                seeds[g_idx, b_idx, r_idx] = seed
                tasks.append((g_idx, b_idx, r_idx, float(gamma), float(b), seed))
                task_number += 1
            if len(np.unique(seeds[g_idx, b_idx])) != config["NUM_RUNS"]:
                raise ValueError("Repeated seed within one parameter point; choose another BASE_SEED.")
    return tasks, seeds


def save_raw(path, run_values, seeds, config):
    """Keep full precision, seeds, configuration, and partial results if interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        run_values=run_values,
        seeds=seeds,
        gamma_values=np.asarray(config["GAMMA_VALUES"], dtype=np.float64),
        b_values=np.asarray(config["B_VALUES"], dtype=np.float64),
        config_json=np.array(json.dumps(config, ensure_ascii=False)),
    )
    os.replace(temporary, path)


def load_raw(path):
    with np.load(path, allow_pickle=False) as data:
        run_values = data["run_values"].copy()
        seeds = data["seeds"].copy()
        config = json.loads(data["config_json"].item())
    if config.get("MODEL_PROTOCOL") != PROTOCOL_ID:
        raise ValueError("Input data use an old or undocumented updating protocol; rerun simulations with this version.")
    return run_values, seeds, config


def simulate(config, workers, output_dir):
    validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, seeds = make_tasks(config)
    run_values = np.full(seeds.shape, np.nan, dtype=np.float64)
    raw_path = output_dir / "figure6_raw_runs.npz"
    workers = max(1, min(int(workers), len(tasks)))
    print(f"L={config['L']}, MCS={config['MCS_TOTAL']}, "
          f"averaging window={config['MCS_TOTAL'] - config['MCS_TRANSIENT']}, "
          f"runs per point={config['NUM_RUNS']}")
    print(f"Workers: {workers}; the worker count is not a real-time CPU usage limit.")
    print(f"Raw results: {raw_path}")
    # Each task returns its own late-window time average; keep the run axis.
    try:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers, initializer=initialize_worker, initargs=(config,),
        ) as pool:
            iterator = pool.imap_unordered(execute_task, tasks, chunksize=1)
            for completed, result in enumerate(
                tqdm(iterator, total=len(tasks), desc="Simulations", unit="run"), start=1,
            ):
                g_idx, b_idx, r_idx, value = result
                run_values[g_idx, b_idx, r_idx] = value
                if completed % 1000 == 0:
                    save_raw(raw_path, run_values, seeds, config)
    except BaseException:
        save_raw(raw_path, run_values, seeds, config)
        raise
    save_raw(raw_path, run_values, seeds, config)
    return run_values, seeds


# ==========================================
# 4. Confidence intervals across independent runs
# ==========================================
def summarize(run_values, config):
    validate_config(config)
    run_values = np.asarray(run_values, dtype=np.float64)
    expected_shape = (len(config["GAMMA_VALUES"]), len(config["B_VALUES"]), config["NUM_RUNS"])
    if run_values.shape != expected_shape or not np.all(np.isfinite(run_values)):
        raise ValueError("Raw results are incomplete or their shape does not match the configuration.")
    if np.any(run_values < 0.0) or np.any(run_values > 1.0):
        raise ValueError("Raw cooperation averages lie outside [0, 1].")
    mean = np.mean(run_values, axis=2)  # Same average as the original code.
    return {"mean": mean}



def export_results(run_values, seeds, config, stats, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gamma_values = config["GAMMA_VALUES"]
    b_values = config["B_VALUES"]
    with (output_dir / "figure6_run_values.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gamma", "b", "run", "seed", "late_window_mean_fc"])
        for g_idx, gamma in enumerate(gamma_values):
            for b_idx, b in enumerate(b_values):
                for r_idx in range(config["NUM_RUNS"]):
                    writer.writerow([
                        gamma, b, r_idx + 1, int(seeds[g_idx, b_idx, r_idx]),
                        float(run_values[g_idx, b_idx, r_idx]),
                    ])
    with (output_dir / "figure6_statistics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "gamma", "b", "n_runs", "mean_fc",
        ])
        for g_idx, gamma in enumerate(gamma_values):
            for b_idx, b in enumerate(b_values):
                writer.writerow([
                    gamma, b, config["NUM_RUNS"],
                    float(stats["mean"][g_idx, b_idx]),
                ])
    # Preserve the original three-column mean-data format as a separate file.
    rows = [[gamma, b, stats["mean"][g_idx, b_idx]]
            for g_idx, gamma in enumerate(gamma_values)
            for b_idx, b in enumerate(b_values)]
    np.savetxt(
        output_dir / (f"u={config['FIXED_U']:g}α={config['FIXED_ALPHA']:g}"
                      "final_simulation_data_gamma_lines.txt"),
        np.asarray(rows), fmt="%.6f", header="gamma b fc", comments="",
    )
    metadata = dict(config)
    metadata.update({
        "sample_unit": "one independent realization's late-window time average",
        "updating_protocol": [
            "independent uniform focal sampling with replacement: N events per MCS",
            "global endorsement follows current strategies after each strategy change",
            "asynchronous imitation within each MCS",
            "synchronous reputation update at the end of each MCS",
            "original absorbing-configuration early return applies only when u=0",
        ],
        "interpretation": "Lines and markers show the ensemble mean of independent late-window averages.",
    })
    (output_dir / "figure6_settings.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    window = config["MCS_TOTAL"] - config["MCS_TRANSIENT"]
    caption = (
        f"Each point shows the mean cooperator fraction over {config['NUM_RUNS']} independent "
        f"realizations, with each realization averaged over the final {window} MCS of "
        f"{config['MCS_TOTAL']} MCS. Lines connect the means at adjacent sampled b values."
    )
    (output_dir / "figure6_caption_en.txt").write_text(caption + "\n", encoding="utf-8")
    (output_dir / "figure6_caption_en.tex").write_text(
        caption.replace("%", r"\%") + "\n", encoding="utf-8",
    )


# ==========================================
# 5. Original mean-curve style
# ==========================================
def load_theory_curves(gamma_values):
    """Read the predictions in the same gamma order as the simulation."""
    path = Path(__file__).resolve().parent / "图6理论预测_含γ0.50.csv"
    if not path.is_file():
        raise FileNotFoundError(f"请将理论CSV与本脚本放在同一文件夹：{path}")
    columns = ["b", *(f"pair_approximation_fc_gamma_{gamma:.2f}" for gamma in gamma_values)]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
        missing = set(columns) - set(header)
        if missing:
            raise ValueError(f"理论CSV缺少列：{sorted(missing)}")
        data = np.loadtxt(handle, delimiter=",", ndmin=2,
                          usecols=[header.index(column) for column in columns])
    if len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("理论CSV须包含至少两行有限数值。")
    data = data[np.argsort(data[:, 0], kind="stable")]
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("理论CSV中的b值必须互不重复。")
    if np.any((data[:, 1:] < 0) | (data[:, 1:] > 1)):
        raise ValueError("理论合作比例必须位于[0, 1]。")
    return data


def plot_figure(config, stats, output_dir, show=False, theory=None):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    styles = {
        "0.00": ("#e4d00a", "*", (0, (5, 2))),
        "0.05": ("#00ced1", "+", (0, (1, 1.5))),
        "0.10": ("#0000CD", "o", (0, (5, 2, 1, 2))),
        "0.20": ("#DC143C", "o", (0, (5, 2, 1, 2, 1, 2))),
        "0.50": ("#00ced1", "+", (0, (1, 1.5))),
        "1.00": ("#228B22", "s", (0, (8, 3))),
    }
    if theory is None:
        theory = load_theory_curves(config["GAMMA_VALUES"])
    b_values = np.asarray(config["B_VALUES"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for j, gamma in enumerate(config["GAMMA_VALUES"]):
        color, marker, theory_style = styles[f"{gamma:.2f}"]
        ax.plot(b_values, stats["mean"][j], label=rf"$\gamma={gamma:.2f}$",
                color=color, linestyle="-", marker=marker,
                markersize=8 if marker == "*" else 6,
                linewidth=1.2, markeredgecolor=color,
                markerfacecolor="none" if color == "#0000CD" else color,
                zorder=3)
        ax.plot(theory[:, 0], theory[:, j + 1],
                color=color, linestyle=theory_style, linewidth=1.2,
                label=rf"$\gamma={gamma:.2f}$ (Theory)", zorder=4)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_xlim(1.0, 5.0)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.margins(0)
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=12)
    ax.set_xlabel(r"$b$", fontsize=14)
    ax.set_ylabel(r"$f_c$", fontsize=14)
    ax.legend(
        fontsize=9, frameon=True, framealpha=1.0, edgecolor="black",
        loc="lower center", bbox_to_anchor=(0.63, 0.06), ncol=1,
        handlelength=2.4, handletextpad=0.50, labelspacing=0.40, borderpad=0.35,
    )
    fig.tight_layout()
    output_dir = Path(output_dir)
    for extension in ("png", "eps"):
        path = output_dir / f"figure6.{extension}"
        fig.savefig(path, dpi=600, bbox_inches="tight", format=extension)
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Figure 6: current-state replacement-sampling simulations and mean cooperation curves across independent runs.",
    )
    parser.add_argument("--plot-only", type=Path, help="Replot an existing figure6_raw_runs.npz without simulations.")
    parser.add_argument("--output-dir", type=Path, help="Explicit directory for this run's outputs.")
    parser.add_argument("--workers", type=int, default=max(1, int((mp.cpu_count() or 1) * CPU_FRACTION)))
    parser.add_argument("--show", action="store_true", help="Also display the figure after saving.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be >= 1")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or OUTPUT_ROOT / ("replot_" if args.plot_only else "run_") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        run_values, seeds, config = load_raw(args.plot_only)
        print(f"Replotting: {args.plot_only.resolve()}")
    else:
        config = default_config()
    theory = load_theory_curves(config["GAMMA_VALUES"])
    if not args.plot_only:
        run_values, seeds = simulate(config, args.workers, output_dir)
    stats = summarize(run_values, config)
    export_results(run_values, seeds, config, stats, output_dir)
    plot_figure(config, stats, output_dir, show=args.show, theory=theory)
    print(f"Finished. Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    prepare_run_directory(__file__, globals())
    mp.freeze_support()
    main()
