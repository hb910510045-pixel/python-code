# -*- coding: utf-8 -*-
"""Figure 6 with pointwise 95% percentile-bootstrap confidence intervals.

The four original Numba functions and production parameters are preserved.
Run averages, seeds, SD, SEM, confidence limits and settings are exported.
Run normally in PyCharm, or use --plot-only with a saved raw-runs NPZ file.
Dependencies: numpy, numba, matplotlib, tqdm (no scipy dependency).
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

MCS_TOTAL = 20000  # 模拟总步数
MCS_TRANSIENT = 17000  # 暂态步数
NUM_RUNS = 100  # 独立运行次数

# 固定 alpha, beta 和 u。横坐标保留为 b，图例的多条线改为不同的 gamma
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_U = 0.00

# 在 GAMMA_VALUES 中增加了 1.00
GAMMA_VALUES = [0.00, 0.05, 0.10, 0.20, 1.00]
# 将 b 的区间调整为 [1.0, 6.0]，取 101 个点（保持步长 0.05）
B_VALUES = np.linspace(1.0, 6.0, 101)
BASE_SEED = 20261007  # 固定基础种子以便复现


# ==========================================
# 2. 核心逻辑函数
# ==========================================

# 2. 解决 JIT 缓存缺失问题：使用 @njit(cache=True)
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
    # --- [GPop 计算] 全局流行度 (在 MCS 开始时计算一次快照) ---
    total_rep = np.sum(reputations) + N * epsilon
    sum_c = 0.0
    for i in range(N):
        if strategies[i] == 1:
            sum_c += reputations[i] + epsilon

    gpop_C = sum_c / total_rep
    gpop_D = 1.0 - gpop_C

    # --- 策略更新 (MCS 循环) ---
    indices = np.random.permutation(N)
    for k in range(N):
        i = indices[k]

        # 1. 突变
        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)

        # 2. 选择与模仿
        else:
            j = neighbor_map[i, np.random.randint(0, 4)]

            if strategies[i] != strategies[j]:
                # --- 计算 F_i ---
                p_i = 0.0
                num_i, den_i = reputations[i] + epsilon, reputations[i] + epsilon

                for idx in range(4):
                    nb = neighbor_map[i, idx]
                    p_i += calculate_payoff(strategies[i], strategies[nb], b)
                    w = reputations[nb] + epsilon
                    den_i += w
                    if strategies[nb] == strategies[i]:
                        num_i += w

                lpop_i = num_i / den_i
                tpop_i = alpha * lpop_i + (1.0 - alpha) * (gpop_C if strategies[i] == 1 else gpop_D)

                if tpop_i < 1e-9: tpop_i = 1e-9
                fit_i = p_i * (tpop_i ** beta)

                # --- 计算 F_j ---
                p_j = 0.0
                num_j, den_j = reputations[j] + epsilon, reputations[j] + epsilon

                for idx in range(4):
                    nb = neighbor_map[j, idx]
                    p_j += calculate_payoff(strategies[j], strategies[nb], b)
                    w = reputations[nb] + epsilon
                    den_j += w
                    if strategies[nb] == strategies[j]:
                        num_j += w

                lpop_j = num_j / den_j
                tpop_j = alpha * lpop_j + (1.0 - alpha) * (gpop_C if strategies[j] == 1 else gpop_D)

                if tpop_j < 1e-9: tpop_j = 1e-9
                fit_j = p_j * (tpop_j ** beta)

                # --- 费米规则 ---
                exp_val = (fit_i - fit_j) / K
                if exp_val > 20:
                    prob = 0.0
                elif exp_val < -20:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + np.exp(exp_val))

                if np.random.random() < prob:
                    strategies[i] = strategies[j]

    # --- 声誉更新 ---
    for i in range(N):
        reputations[i] = (1.0 - gamma) * reputations[i] + gamma * float(strategies[i])

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
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_BATCH = 500
BOOTSTRAP_SEED_NAMESPACE = 610061
OUTPUT_ROOT = Path(__file__).resolve().parent / "figure6_uncertainty_results"

_WORKER_CONFIG = None
_WORKER_NEIGHBORS = None


def default_config():
    """Production settings: identical to the supplied simulation."""
    return {
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
def bootstrap_mean_ci(values, rng, n_resamples=BOOTSTRAP_RESAMPLES,
                      confidence=CONFIDENCE_LEVEL, batch=BOOTSTRAP_BATCH):
    """Pointwise percentile-bootstrap CI for the mean of RUN averages.

    Do not pass the 3000 correlated time points from one run here.
    No normality assumption is imposed on the distribution of run averages.
    This is an approximate bootstrap interval, not a simultaneous confidence band.
    Identical observations produce a zero-width empirical interval.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Need at least two finite independent run averages.")
    if n_resamples < 1000 or batch < 1 or not 0 < confidence < 1:
        raise ValueError("Use >= 1000 resamples, batch >= 1 and 0 < confidence < 1.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Cooperator fractions must lie in [0, 1].")
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    resampled_means = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch):
        stop = min(start + batch, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)), dtype=np.int32)
        resampled_means[start:stop] = np.mean(values[indices], axis=1)
    tail_percent = 100.0 * (1.0 - confidence) / 2.0
    lower, upper = np.percentile(
        resampled_means, [tail_percent, 100.0 - tail_percent], method="linear",
    )
    return float(lower), float(upper)


def summarize(run_values, config, n_resamples=BOOTSTRAP_RESAMPLES, progress=True):
    validate_config(config)
    run_values = np.asarray(run_values, dtype=np.float64)
    expected_shape = (len(config["GAMMA_VALUES"]), len(config["B_VALUES"]), config["NUM_RUNS"])
    if run_values.shape != expected_shape or not np.all(np.isfinite(run_values)):
        raise ValueError("Raw results are incomplete or their shape does not match the configuration.")
    if np.any(run_values < 0.0) or np.any(run_values > 1.0):
        raise ValueError("Raw cooperation averages lie outside [0, 1].")
    mean = np.mean(run_values, axis=2)  # Same average as the original code.
    sd = np.std(run_values, axis=2, ddof=1)
    sem = sd / np.sqrt(config["NUM_RUNS"])
    low = np.empty_like(mean)
    high = np.empty_like(mean)
    identical = np.all(run_values == run_values[:, :, :1], axis=2)
    for g_idx, b_idx in tqdm(
        np.ndindex(mean.shape), total=mean.size,
        desc="Pointwise bootstrap CIs", unit="point", disable=not progress,
    ):
        # A separate RNG makes statistical resampling independent of simulation RNGs.
        ci_seed = np.random.SeedSequence([
            config["BASE_SEED"], BOOTSTRAP_SEED_NAMESPACE, g_idx, b_idx,
        ])
        low[g_idx, b_idx], high[g_idx, b_idx] = bootstrap_mean_ci(
            run_values[g_idx, b_idx], np.random.default_rng(ci_seed), n_resamples=n_resamples,
        )
    return {
        "mean": mean, "sd": sd, "sem": sem,
        "ci95_lower": low, "ci95_upper": high,
        "identical_runs": identical,
    }


def export_results(run_values, seeds, config, stats, output_dir,
                   n_resamples=BOOTSTRAP_RESAMPLES):
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
            "gamma", "b", "n_runs", "mean_fc", "sd_across_runs", "sem",
            "ci95_lower", "ci95_upper", "all_run_averages_identical",
        ])
        for g_idx, gamma in enumerate(gamma_values):
            for b_idx, b in enumerate(b_values):
                writer.writerow([
                    gamma, b, config["NUM_RUNS"],
                    *[float(stats[key][g_idx, b_idx]) for key in
                      ("mean", "sd", "sem", "ci95_lower", "ci95_upper")],
                    bool(stats["identical_runs"][g_idx, b_idx]),
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
        "error_bar": "pointwise 95% percentile-bootstrap CI of the mean across run averages",
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed_namespace": BOOTSTRAP_SEED_NAMESPACE,
        "sample_unit": "one independent realization's late-window time average",
        "intervals_are_simultaneous": False,
        "normality_assumption_for_run_averages": False,
        "all_identical_parameter_points": int(np.count_nonzero(stats["identical_runs"])),
        "unchanged_dynamics": [
            "random permutation: each player is selected once per MCS",
            "global endorsement is a snapshot at the beginning of each MCS",
            "asynchronous imitation within each MCS",
            "synchronous reputation update at the end of each MCS",
            "original absorbing-configuration early return applies only when u=0",
        ],
        "interpretation": (
            "Intervals quantify observed variation in the estimated mean across independent runs. "
            "They are not SD bars, not a simultaneous band, and not proof of statistical significance. "
            "An identical set of run averages gives a zero-width empirical interval; this does not "
            "establish absence of rare outcomes or of convergence/finite-size effects."
        ),
    })
    (output_dir / "figure6_settings.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    window = config["MCS_TOTAL"] - config["MCS_TRANSIENT"]
    caption = (
        f"Each point shows the mean cooperator fraction over {config['NUM_RUNS']} independent "
        f"realizations, with each realization averaged over the final {window} MCS of "
        f"{config['MCS_TOTAL']} MCS. Error bars indicate pointwise 95% percentile-bootstrap "
        f"confidence intervals for the mean, calculated from {n_resamples} resamples of the "
        "independent realization averages. No curve smoothing is applied. Intervals narrower "
        "than the markers may not be visible; identical realization averages give zero-width intervals."
    )
    (output_dir / "figure6_caption_en.txt").write_text(caption + "\n", encoding="utf-8")
    (output_dir / "figure6_caption_en.tex").write_text(
        caption.replace("%", r"\%") + "\n", encoding="utf-8",
    )
    print(f"Points with identical run averages: {np.count_nonzero(stats['identical_runs'])}/{stats['mean'].size}")


# ==========================================
# 5. Original curve style plus confidence bars
# ==========================================
def plot_figure(config, stats, output_dir, show=False):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    colors = ["#e4d00a", "#00ced1", "#0000CD", "#DC143C", "#228B22", "#800080"]
    markers = ["*", "+", "o", "o", "s", "^"]
    b_values = np.asarray(config["B_VALUES"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for j, gamma in enumerate(config["GAMMA_VALUES"]):
        color = colors[j % len(colors)]
        marker = markers[j % len(markers)]
        lower = stats["ci95_lower"][j]
        upper = stats["ci95_upper"][j]
        # Draw the interval itself, then place mean markers separately.
        # This also supports asymmetric intervals without clipping endpoints.
        ax.errorbar(
            b_values, (lower + upper) / 2.0, yerr=(upper - lower) / 2.0,
            fmt="none", ecolor=color, elinewidth=0.85,
            capsize=2.0, capthick=0.85, zorder=2, label="_nolegend_",
        )
        ax.plot(
            b_values, stats["mean"][j], label=rf"$\gamma={gamma:.2f}$",
            color=color, marker=marker, markersize=8 if marker == "*" else 6,
            linewidth=1.2, markeredgecolor=color,
            markerfacecolor="none" if color == "#0000CD" else color, zorder=3,
        )
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_xlim(1.0, 6.0)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax.margins(0)
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=12)
    ax.set_xlabel(r"$b$", fontsize=14)
    ax.set_ylabel(r"$f_c$", fontsize=14)
    ax.legend(
        fontsize=10, frameon=True, framealpha=1.0, edgecolor="black",
        loc="upper right", bbox_to_anchor=(0.85, 0.96), handlelength=1.5,
    )
    fig.tight_layout()
    output_dir = Path(output_dir)
    for extension in ("png", "eps", "pdf"):
        path = output_dir / f"figure6.{extension}"
        fig.savefig(path, dpi=600, bbox_inches="tight", format=extension)
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Figure 6: unchanged simulations with confidence intervals across independent runs.",
    )
    parser.add_argument("--plot-only", type=Path, help="Replot an existing figure6_raw_runs.npz without simulations.")
    parser.add_argument("--output-dir", type=Path, help="Explicit directory for this run's outputs.")
    parser.add_argument("--workers", type=int, default=max(1, int((mp.cpu_count() or 1) * CPU_FRACTION)))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--show", action="store_true", help="Also display the figure after saving.")
    args = parser.parse_args()
    if args.workers < 1 or args.bootstrap_resamples < 1000:
        parser.error("workers must be >= 1 and bootstrap-resamples must be >= 1000")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or OUTPUT_ROOT / ("replot_" if args.plot_only else "run_") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        run_values, seeds, config = load_raw(args.plot_only)
        print(f"Replotting: {args.plot_only.resolve()}")
    else:
        config = default_config()
        run_values, seeds = simulate(config, args.workers, output_dir)
    stats = summarize(run_values, config, n_resamples=args.bootstrap_resamples)
    export_results(run_values, seeds, config, stats, output_dir, args.bootstrap_resamples)
    plot_figure(config, stats, output_dir, show=args.show)
    print(f"Finished. Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
