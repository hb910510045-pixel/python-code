# Updating protocol fixed to the manuscript; historical results require recomputation.
import csv
import datetime as dt
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from scipy.stats import t as student_t
from tqdm import tqdm


# ============================================================
# 1. Model parameters
# ============================================================
L = 100
N = L * L
K = 0.1
FIXED_B = 1.11

# Current user-selected run length. The paper-scale setting is 20000 MCS.
MCS_TOTAL = 2000
MCS_TRANSIENT = 1700
INIT_COOP_RATE = 0.5
NUM_RUNS = 50

LAMBDA_VAL = 0.1
MUTATION_RATE = 0.05
EPSILON = 0.01

BETA_VALUES = [0.30, 0.50, 0.80, 1.00]
ALPHA_VALUES = [0.00, 0.50, 0.80, 1.00]

# Reproducibility and resource control.
BASE_SEED = 20261007
CPU_FRACTION = 0.80

# Optional short execution test; applied in Windows spawn children as well.
SMOKE_TEST = "--smoke-test" in sys.argv[1:]
if SMOKE_TEST:
    L = 8
    N = L * L
    MCS_TOTAL = 40
    MCS_TRANSIENT = 30
    NUM_RUNS = 3
    CPU_FRACTION = 1.0 / (os.cpu_count() or 1)
    REQUESTED_CONVERGENCE_WINDOW = 10

# Convergence diagnostics compare two adjacent windows at the end.
REQUESTED_CONVERGENCE_WINDOW = 3000
CONVERGENCE_WINDOW = min(REQUESTED_CONVERGENCE_WINDOW, max(1, MCS_TOTAL // 4))
CONVERGENCE_TOL = 0.01

# The full trajectory is kept in each worker, while only these points
# are retained by the parent process for plotting and output.
PLOT_STRIDE = 100
PLOT_POINT_COUNT = 200
REQUESTED_SNAPSHOT_TIMES = (1, 10, 100, 1000, 10000, 20000)
SNAPSHOT_TIMES = tuple(sorted(set(
    [t for t in REQUESTED_SNAPSHOT_TIMES if 1 <= t <= MCS_TOTAL] + [MCS_TOTAL]
)))
N_BOOTSTRAP = 2000
# Student-t critical values are computed from each condition's actual count.

WORKER_NEIGHBOR_MAP = None


# ============================================================
# 2. JIT-compiled model functions
# ============================================================
@jit(nopython=True)
def build_neighbor_map(l_size):
    n = l_size * l_size
    neighbors = np.empty((n, 4), dtype=np.int64)
    for i in range(n):
        row, col = i // l_size, i % l_size
        neighbors[i, 0] = ((row - 1 + l_size) % l_size) * l_size + col
        neighbors[i, 1] = ((row + 1) % l_size) * l_size + col
        neighbors[i, 2] = row * l_size + ((col - 1 + l_size) % l_size)
        neighbors[i, 3] = row * l_size + ((col + 1) % l_size)
    return neighbors


@jit(nopython=True)
def calculate_payoff(s_i, s_j, b):
    if s_i == 1:
        return 1.0 if s_j == 1 else 0.0
    return b if s_j == 1 else 0.0


@jit(nopython=True)
def seed_numba(seed):
    """Seed Numba's independent random-number generator state."""
    np.random.seed(seed)


# Complete simulation kernel: no model_rules.py dependency.
PROTOCOL_ID = 'replacement_current_global_end_mcs_ema_v1'

@njit
def current_weights(strategies, reputations, epsilon):
    total, cooperative = 0.0, 0.0
    for x in range(len(strategies)):
        weight = reputations[x] + epsilon
        total += weight
        if strategies[x] == 1:
            cooperative += weight
    return total, cooperative



@njit
def fermi_probability(focal_fitness, neighbor_fitness, noise):
    # Stable logistic: no hard truncation of finite probabilities.
    exponent = (focal_fitness - neighbor_fitness) / noise
    if exponent >= 0.0:
        value = np.exp(-exponent)
        return value / (1.0 + value)
    return 1.0 / (1.0 + np.exp(exponent))



@njit
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



@njit
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



@njit
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
    os.chdir(output)
    return output


@jit(nopython=True)
def single_mcs_step(n, k, b, alpha, beta, lam, epsilon, u, strategies, reputations, neighbor_map):
    mcs_fixed(strategies, reputations, neighbor_map, k, b,
              alpha, beta, lam, epsilon, u)
    return strategies, reputations


# ============================================================
# 3. Multiprocessing and reproducibility
# ============================================================
def make_seed(beta_index, alpha_index, run_index):
    """Generate a stable, distinct 32-bit seed for every independent run."""
    sequence = np.random.SeedSequence(
        [BASE_SEED, int(beta_index), int(alpha_index), int(run_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def initialize_worker(l_size):
    """Build the neighbor map once per worker instead of once per run."""
    global WORKER_NEIGHBOR_MAP
    WORKER_NEIGHBOR_MAP = build_neighbor_map(l_size)

    # Compile the Numba kernels in each worker before timed tasks begin.
    population = l_size * l_size
    dummy_s = np.ones(population, dtype=np.int64)
    dummy_r = np.full(population, 0.5, dtype=np.float64)
    seed_numba(13579)
    single_mcs_step(
        population, K, FIXED_B, 0.5, 0.5, LAMBDA_VAL, EPSILON,
        MUTATION_RATE, dummy_s, dummy_r, WORKER_NEIGHBOR_MAP
    )


def run_single_simulation(task):
    """Run one independent trajectory and return raw diagnostics."""
    beta_index, alpha_index, beta, alpha, run_index, seed = task
    start = time.perf_counter()

    # Seed both Python/NumPy and Numba random states.
    np.random.seed(seed)
    seed_numba(seed)

    # Independent Bernoulli initialization, not a fixed cooperator count.
    strategies = (np.random.random(N) < INIT_COOP_RATE).astype(np.int64)
    reputations = np.full(N, 0.5, dtype=np.float64)
    trajectory = np.empty(MCS_TOTAL + 1, dtype=np.float64)
    trajectory[0] = np.mean(strategies)

    for mcs in range(1, MCS_TOTAL + 1):
        single_mcs_step(
            N, K, FIXED_B, alpha, beta, LAMBDA_VAL, EPSILON,
            MUTATION_RATE, strategies, reputations, WORKER_NEIGHBOR_MAP
        )
        trajectory[mcs] = np.mean(strategies)

    last_start = MCS_TOTAL - CONVERGENCE_WINDOW + 1
    previous_start = MCS_TOTAL - 2 * CONVERGENCE_WINDOW + 1
    last_mean = float(np.mean(trajectory[last_start:MCS_TOTAL + 1]))
    previous_mean = float(np.mean(trajectory[previous_start:last_start]))
    change = last_mean - previous_mean

    elapsed = time.perf_counter() - start
    return {
        "beta_index": beta_index,
        "alpha_index": alpha_index,
        "beta": beta,
        "alpha": alpha,
        "run_index": run_index,
        "seed": seed,
        "trajectory": trajectory,
        "previous_mean": previous_mean,
        "last_mean": last_mean,
        "change": change,
        "converged": bool(abs(change) <= CONVERGENCE_TOL),
        "elapsed_seconds": elapsed,
    }


# ============================================================
# 4. Statistics, confidence intervals, and output
# ============================================================
def bootstrap_mean_ci(values, seed):
    """Pointwise 95% percentile bootstrap CI for the sample mean."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    bootstrap_means = values[indices].mean(axis=1)
    return tuple(np.percentile(bootstrap_means, [2.5, 97.5], axis=0))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalize_condition_statistics(condition_stats):
    """Convert online sums and per-run snapshots to output statistics."""
    finalized = {}
    for key, state in condition_stats.items():
        n = state["count"]
        mean = state["mean"]
        if n > 1:
            variance = np.maximum(state["m2"] / (n - 1), 0.0)
            std = np.sqrt(variance)
        else:
            std = np.zeros_like(mean)

        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
        ci_low = mean - critical * std / np.sqrt(max(n, 1))
        ci_high = mean + critical * std / np.sqrt(max(n, 1))
        ci_low = np.clip(ci_low, 0.0, 1.0)
        ci_high = np.clip(ci_high, 0.0, 1.0)

        snapshot_values = np.vstack(state["snapshot_values"])
        boot_low, boot_high = bootstrap_mean_ci(
            snapshot_values,
            seed=BASE_SEED + 100000 + 100 * key[0] + key[1]
        )

        snapshot_mean = snapshot_values.mean(axis=0)
        snapshot_std = (
            snapshot_values.std(axis=0, ddof=1)
            if n > 1 else np.zeros(snapshot_values.shape[1], dtype=np.float64)
        )

        finalized[key] = {
            "beta": state["beta"],
            "alpha": state["alpha"],
            "count": n,
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "snapshot_values": snapshot_values,
            "snapshot_mean": snapshot_mean,
            "snapshot_std": snapshot_std,
            "bootstrap_low": np.asarray(boot_low),
            "bootstrap_high": np.asarray(boot_high),
            "runtime_seconds": state["runtime_seconds"],
        }
    return finalized


def save_outputs(output_dir, condition_stats, finalized, runtime_report,
                 convergence_rows, seed_rows):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "L": L,
            "N": N,
            "K": K,
            "FIXED_B": FIXED_B,
            "MCS_TOTAL": MCS_TOTAL,
            "MCS_TRANSIENT": MCS_TRANSIENT,
            "INIT_COOP_RATE": INIT_COOP_RATE,
            "initialization": "independent_Bernoulli",
            "strategy_sampling": "uniform_with_replacement",
            "model_protocol": PROTOCOL_ID,
            "NUM_RUNS": NUM_RUNS,
            "LAMBDA_VAL": LAMBDA_VAL,
            "MUTATION_RATE": MUTATION_RATE,
            "EPSILON": EPSILON,
            "BETA_VALUES": BETA_VALUES,
            "ALPHA_VALUES": ALPHA_VALUES,
            "BASE_SEED": BASE_SEED,
            "CPU_FRACTION": CPU_FRACTION,
            "CONVERGENCE_WINDOW": CONVERGENCE_WINDOW,
            "CONVERGENCE_TOL": CONVERGENCE_TOL,
            "SNAPSHOT_TIMES": SNAPSHOT_TIMES,
            "N_BOOTSTRAP": N_BOOTSTRAP,
            "global_endorsement_update": "incremental_after_each_strategy_change",
            "reputation_update": "synchronous_at_MCS_end",
        }, handle, indent=2)

    with (output_dir / "runtime_report.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime_report, handle, indent=2)

    write_csv(
        output_dir / "seeds.csv",
        ["beta", "alpha", "run_index", "seed"],
        seed_rows,
    )
    write_csv(
        output_dir / "convergence_diagnostics.csv",
        ["beta", "alpha", "run_index", "seed", "previous_window_mean",
         "last_window_mean", "change", "converged", "elapsed_seconds"],
        convergence_rows,
    )

    runtime_rows = []
    trajectory_rows = []
    snapshot_rows = []
    for key, state in finalized.items():
        beta, alpha = state["beta"], state["alpha"]
        runtime_rows.append({
            "beta": beta,
            "alpha": alpha,
            "runs": state["count"],
            "sum_run_seconds": state["runtime_seconds"],
            "mean_run_seconds": state["runtime_seconds"] / state["count"],
        })

        for index, mcs in enumerate(PLOT_TIMES):
            trajectory_rows.append({
                "MCS": int(mcs), "beta": beta, "alpha": alpha,
                "mean_fc": state["mean"][index],
                "std_fc": state["std"][index],
                "ci95_low": state["ci_low"][index],
                "ci95_high": state["ci_high"][index],
                "runs": state["count"],
            })

        for index, mcs in enumerate(SNAPSHOT_TIMES):
            snapshot_rows.append({
                "MCS": int(mcs), "beta": beta, "alpha": alpha,
                "mean_fc": state["snapshot_mean"][index],
                "std_fc": state["snapshot_std"][index],
                "bootstrap_ci95_low": state["bootstrap_low"][index],
                "bootstrap_ci95_high": state["bootstrap_high"][index],
                "runs": state["count"],
            })

    write_csv(output_dir / "runtime_by_condition.csv",
              ["beta", "alpha", "runs", "sum_run_seconds", "mean_run_seconds"],
              runtime_rows)
    write_csv(output_dir / "mean_trajectory_ci.csv",
              ["MCS", "beta", "alpha", "mean_fc", "std_fc",
               "ci95_low", "ci95_high", "runs"], trajectory_rows)
    write_csv(output_dir / "snapshot_statistics.csv",
              ["MCS", "beta", "alpha", "mean_fc", "std_fc",
               "bootstrap_ci95_low", "bootstrap_ci95_high", "runs"],
              snapshot_rows)

    # A compact text file in the original style, now with uncertainty columns.
    with (output_dir / "mean_trajectory_with_uncertainty.txt").open(
            "w", encoding="utf-8") as handle:
        handle.write("MCS\tBeta\tAlpha\tfc_mean\tfc_std\tCI95_low\tCI95_high\n")
        for row in trajectory_rows:
            handle.write(
                f"{row['MCS']}\t{row['beta']:.2f}\t{row['alpha']:.2f}\t"
                f"{row['mean_fc']:.8f}\t{row['std_fc']:.8f}\t"
                f"{row['ci95_low']:.8f}\t{row['ci95_high']:.8f}\n"
            )


def plot_results(finalized, output_dir):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"

    colors = ["#76EE00", "#FFD700", "#4169E1", "#FF0000"]
    sub_labels = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for panel, beta in enumerate(BETA_VALUES):
        ax = axes[panel]
        for alpha_index, alpha in enumerate(ALPHA_VALUES):
            state = finalized[(panel, alpha_index)]
            color = colors[alpha_index]
            x = PLOT_TIMES

            # Use many logarithmically spaced points for a smooth mean
            # trajectory. The shaded band is the pointwise 95% CI.
            ax.fill_between(x, state["ci_low"], state["ci_high"],
                            color=color, alpha=0.12, linewidth=0)
            ax.plot(x, state["mean"], color=color, linewidth=2.0,
                    label=rf"$\alpha = {alpha:.3f}$")

            # Display the 95% bootstrap confidence interval at the same
            # logarithmic time points.
            snap_mean = state["snapshot_mean"]
            snap_low = state["bootstrap_low"]
            snap_high = state["bootstrap_high"]
            yerr = np.vstack((snap_mean - snap_low, snap_high - snap_mean))
            ax.errorbar(SNAPSHOT_TIMES, snap_mean, yerr=yerr,
                        fmt="none", ecolor=color, elinewidth=0.8,
                        capsize=2.0, alpha=0.65)

        ax.set_xscale("log")
        ax.set_xlim(1, MCS_TOTAL)
        # Make f_c=0 and f_c=1 coincide exactly with the plot boundaries.
        ax.set_ylim(0.0, 1.0)
        ax.margins(y=0.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=14)
        ax.set_ylabel(r"$f_c$", fontsize=16, fontstyle="italic")
        ax.set_xlabel(r"$MCS$", fontsize=16, fontstyle="italic")
        ax.text(0.04, 0.92, sub_labels[panel], transform=ax.transAxes,
                fontsize=16, fontweight="bold")
        ax.text(0.96, 0.08, rf"$\beta = {beta:.3f}$",
                transform=ax.transAxes, fontsize=14, fontstyle="italic",
                ha="right", va="bottom")

        # ====================== 修改在这里 ======================
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 0.90),
            frameon=True,
            fontsize=12,
            labelspacing=0.2
        )
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)
        # =======================================================

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, wspace=0.2, hspace=0.25)
    # ========== 修改点：同时输出 PNG + EPS ==========
    png_path = output_dir / f"3581Final_Evolution_u{MUTATION_RATE}_Fig3Style_CI.png"
    eps_path = output_dir / f"3581Final_Evolution_u{MUTATION_RATE}_Fig3Style_CI.eps"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, eps_path


# ============================================================
# 5. Main program
# ============================================================
PLOT_TIMES = np.unique(np.concatenate([
    np.geomspace(1, MCS_TOTAL, PLOT_POINT_COUNT).round().astype(np.int64),
    np.asarray(SNAPSHOT_TIMES, dtype=np.int64),
])).astype(np.int64)


def main():
    # Compile once in the parent so a compilation error fails promptly,
    # rather than repeatedly restarting failed Pool initializer processes.
    if MCS_TOTAL < 2 or NUM_RUNS < 2 or N != L * L:
        raise ValueError("Require MCS_TOTAL>=2, NUM_RUNS>=2 and N=L*L.")
    initialize_worker(L)
    start_time = time.perf_counter()
    started_at = dt.datetime.now().astimezone().isoformat()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(f"3581Final_Data_u{MUTATION_RATE}_audit") / f"run_{timestamp}"

    total_cores = os.cpu_count() or 1
    worker_count = max(1, int(total_cores * CPU_FRACTION))
    print(f"演化模拟开始: u={MUTATION_RATE}, Runs={NUM_RUNS}")
    print(f"模拟时长: {MCS_TOTAL} MCS; 快照时刻: {SNAPSHOT_TIMES}")
    print(f"全局认可度: 轮内随策略变化增量更新，声誉在轮末同步更新")
    print(f"CPU 配额: {CPU_FRACTION:.0%}，使用 {worker_count}/{total_cores} 个逻辑核心")

    condition_stats = {}
    for beta_index, beta in enumerate(BETA_VALUES):
        for alpha_index, alpha in enumerate(ALPHA_VALUES):
            condition_stats[(beta_index, alpha_index)] = {
                "beta": beta,
                "alpha": alpha,
                "count": 0,
                "mean": np.zeros(PLOT_TIMES.size, dtype=np.float64),
                "m2": np.zeros(PLOT_TIMES.size, dtype=np.float64),
                "snapshot_values": [],
                "runtime_seconds": 0.0,
            }

    tasks = []
    seed_rows = []
    for beta_index, beta in enumerate(BETA_VALUES):
        for alpha_index, alpha in enumerate(ALPHA_VALUES):
            for run_index in range(NUM_RUNS):
                seed = make_seed(beta_index, alpha_index, run_index)
                tasks.append((beta_index, alpha_index, beta, alpha, run_index, seed))
                seed_rows.append({"beta": beta, "alpha": alpha,
                                  "run_index": run_index, "seed": seed})

    convergence_rows = []
    pool_start = time.perf_counter()
    with mp.Pool(processes=worker_count,
                 initializer=initialize_worker,
                 initargs=(L,)) as pool:
        iterator = pool.imap_unordered(run_single_simulation, tasks, chunksize=1)
        for result in tqdm(iterator, total=len(tasks), desc="独立模拟"):
            key = (result["beta_index"], result["alpha_index"])
            state = condition_stats[key]

            values = result["trajectory"][PLOT_TIMES]
            state["count"] += 1
            count = state["count"]
            delta = values - state["mean"]
            state["mean"] += delta / count
            state["m2"] += delta * (values - state["mean"])
            state["snapshot_values"].append(
                result["trajectory"][np.asarray(SNAPSHOT_TIMES, dtype=np.int64)]
            )
            state["runtime_seconds"] += result["elapsed_seconds"]

            convergence_rows.append({
                "beta": result["beta"],
                "alpha": result["alpha"],
                "run_index": result["run_index"],
                "seed": result["seed"],
                "previous_window_mean": result["previous_mean"],
                "last_window_mean": result["last_mean"],
                "change": result["change"],
                "converged": result["converged"],
                "elapsed_seconds": result["elapsed_seconds"],
            })

    pool_elapsed = time.perf_counter() - pool_start
    finalized = finalize_condition_statistics(condition_stats)
    total_elapsed = time.perf_counter() - start_time
    finished_at = dt.datetime.now().astimezone().isoformat()

    runtime_report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "total_wall_seconds": total_elapsed,
        "pool_wall_seconds": pool_elapsed,
        "sum_worker_seconds": float(sum(s["runtime_seconds"] for s in finalized.values())),
        "total_cores": total_cores,
        "worker_count": worker_count,
        "cpu_fraction_target": CPU_FRACTION,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": __import__("numba").__version__,
    }

    save_outputs(output_dir, condition_stats, finalized, runtime_report,
                 convergence_rows, seed_rows)
    png_fig, eps_fig = plot_results(finalized, output_dir)

    print(f"总计耗时: {total_elapsed:.2f} s")
    print(f"结果目录: {output_dir.resolve()}")
    print(f"PNG图像文件: {png_fig.resolve()}")
    print(f"EPS矢量图像文件: {eps_fig.resolve()}")
    print("阴影: 均值的逐点95% Student-t区间；误差棒: 快照均值的95% bootstrap percentile区间")
    print(f"收敛判据: 相邻两个 {CONVERGENCE_WINDOW} MCS 窗口均值差的绝对值 <= {CONVERGENCE_TOL}")


if __name__ == "__main__":
    mp.freeze_support()
    prepare_run_directory(__file__, globals())
    main()
