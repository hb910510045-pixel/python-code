import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from numba import njit


# ============================================================
# 1. Model parameters
# ============================================================
K = 0.1
GAMMA = 0.1
EPSILON = 0.01

ALPHA = 0.80
BETA = 0.50

# Finite-size test.
L_VALUES = (50, 100, 200, 300)

# Mutation-rate grid.
U_MIN = 0.000
U_MAX = 0.040
U_STEP = 0.001

# Critical temptation search.
B_LOW_INIT = 1.00
B_HIGH_INIT = 1.70
B_TOLERANCE = 0.001
FC_THRESHOLD = 0.98

# Paper-level statistics.
MCS_MAX = 20000
MCS_STABLE = 3000
NUM_RUNS = 50

# Reproducible independent runs.
BASE_SEED = 20261007

# Do not smooth the data at this stage.
SHOW_PLOTS = False

# Limit worker count to about 70% of logical CPU cores.
CPU_FRACTION = 0.70
MAX_WORKERS = max(1, int((os.cpu_count() or 1) * CPU_FRACTION))

# All output files are saved here, independent of the PyCharm working directory.
OUTPUT_DIR = Path(__file__).resolve().parent / "7a_results"


# ============================================================
# 2. Lattice and synchronous-reputation dynamics
# ============================================================
@njit
def build_neighbor_map(L):
    """Periodic von Neumann neighbors for a one-dimensional lattice array."""
    N = L * L
    neighbors = np.empty((N, 4), dtype=np.int64)

    for i in range(N):
        row = i // L
        col = i % L

        neighbors[i, 0] = ((row - 1 + L) % L) * L + col
        neighbors[i, 1] = ((row + 1) % L) * L + col
        neighbors[i, 2] = row * L + ((col - 1 + L) % L)
        neighbors[i, 3] = row * L + ((col + 1) % L)

    return neighbors


@njit
def payoff(s_i, s_j, b):
    """Weak prisoner's dilemma payoff matrix [[1, 0], [b, 0]]."""
    if s_i == 1:
        return 1.0 if s_j == 1 else 0.0
    return b if s_j == 1 else 0.0


@njit
def run_simulation(
    L,
    u_current,
    b,
    alpha,
    beta,
    gamma,
    epsilon,
    mcs_max,
    mcs_stable,
    neighbors,
    seed,
):
    """
    One independent realization with synchronous reputation updates.

    At the beginning of each MCS, the reputation-weighted global
    endorsements are calculated once and then held fixed during all
    N microscopic strategy updates in that MCS. Strategies may change
    asynchronously during the MCS, but reputations remain unchanged.
    After the N microscopic updates, all reputations are updated
    synchronously using the resulting strategy configuration.
    """
    np.random.seed(seed)

    N = L * L
    strategies = np.random.randint(0, 2, size=N)
    reputations = np.full(N, 0.5, dtype=np.float64)
    fc_history = np.empty(mcs_stable, dtype=np.float64)

    for step in range(mcs_max):
        # --------------------------------------------------------
        # Global endorsement snapshot at the beginning of this MCS.
        # This scan is performed once per MCS, rather than once per
        # microscopic update.
        # --------------------------------------------------------
        global_R_C = 0.0
        global_R_D = 0.0
        for q in range(N):
            value = reputations[q] + epsilon
            if strategies[q] == 1:
                global_R_C += value
            else:
                global_R_D += value
        global_R_total = global_R_C + global_R_D

        # Random selection with replacement: N microscopic events per MCS.
        for _ in range(N):
            i = np.random.randint(0, N)

            # 1. Mutation/reset phase.
            if np.random.random() < u_current:
                # Random reset, rather than a forced strategy flip.
                strategies[i] = np.random.randint(0, 2)

            # 2. Fermi imitation phase.
            else:
                j = neighbors[i, np.random.randint(0, 4)]

                if strategies[i] != strategies[j]:
                    # Fitness of focal player i.
                    payoff_i = 0.0
                    local_num_i = reputations[i] + epsilon
                    local_den_i = reputations[i] + epsilon

                    for n in range(4):
                        nb = neighbors[i, n]
                        payoff_i += payoff(strategies[i], strategies[nb], b)

                        value = reputations[nb] + epsilon
                        local_den_i += value
                        if strategies[nb] == strategies[i]:
                            local_num_i += value

                    local_i = local_num_i / local_den_i
                    if strategies[i] == 1:
                        global_i = global_R_C / global_R_total
                    else:
                        global_i = global_R_D / global_R_total

                    endorsement_i = alpha * local_i + (1.0 - alpha) * global_i
                    fitness_i = payoff_i * (max(endorsement_i, 1e-12) ** beta)

                    # Fitness of target neighbor j.
                    payoff_j = 0.0
                    local_num_j = reputations[j] + epsilon
                    local_den_j = reputations[j] + epsilon

                    for n in range(4):
                        nb = neighbors[j, n]
                        payoff_j += payoff(strategies[j], strategies[nb], b)

                        value = reputations[nb] + epsilon
                        local_den_j += value
                        if strategies[nb] == strategies[j]:
                            local_num_j += value

                    local_j = local_num_j / local_den_j
                    if strategies[j] == 1:
                        global_j = global_R_C / global_R_total
                    else:
                        global_j = global_R_D / global_R_total

                    endorsement_j = alpha * local_j + (1.0 - alpha) * global_j
                    fitness_j = payoff_j * (max(endorsement_j, 1e-12) ** beta)

                    exponent = (fitness_i - fitness_j) / K
                    if exponent > 50.0:
                        imitation_probability = 0.0
                    elif exponent < -50.0:
                        imitation_probability = 1.0
                    else:
                        imitation_probability = 1.0 / (1.0 + np.exp(exponent))

                    if np.random.random() < imitation_probability:
                        strategies[i] = strategies[j]

            # No reputation update is performed here. Reputations remain
            # fixed throughout the current MCS.

        # 3. Synchronous reputation update at the end of the MCS.
        for i in range(N):
            reputations[i] = (
                (1.0 - gamma) * reputations[i]
                + gamma * float(strategies[i])
            )

        if step >= mcs_max - mcs_stable:
            fc_history[step - (mcs_max - mcs_stable)] = np.sum(strategies) / N

    return np.mean(fc_history)


# ============================================================
# 3. Critical-b search and reproducible averaging
# ============================================================
def make_seed(l_index, u_index, run_index):
    """Generate a stable seed for each (L, u, independent-run) combination."""
    modulus = 2**32 - 1
    return int(
        (
            BASE_SEED
            + 1_000_003 * l_index
            + 10_007 * u_index
            + 97 * run_index
        )
        % modulus
    )


_WORKER_L = None
_WORKER_NEIGHBORS = None


def initialize_worker(L):
    global _WORKER_L, _WORKER_NEIGHBORS
    _WORKER_L = L
    _WORKER_NEIGHBORS = build_neighbor_map(L)


def run_one_worker(task):
    l_index, u_index, u_value, b_value, run_index = task
    return run_simulation(
        _WORKER_L,
        u_value,
        b_value,
        ALPHA,
        BETA,
        GAMMA,
        EPSILON,
        MCS_MAX,
        MCS_STABLE,
        _WORKER_NEIGHBORS,
        make_seed(l_index, u_index, run_index),
    )


def average_fc(L, l_index, u_index, u_value, b_value, neighbors, executor=None):
    tasks = [
        (l_index, u_index, u_value, b_value, run_index)
        for run_index in range(NUM_RUNS)
    ]

    if executor is None:
        values = []
        for task in tasks:
            run_index = task[-1]
            values.append(
                run_simulation(
                    L,
                    u_value,
                    b_value,
                    ALPHA,
                    BETA,
                    GAMMA,
                    EPSILON,
                    MCS_MAX,
                    MCS_STABLE,
                    neighbors,
                    make_seed(l_index, u_index, run_index),
                )
            )
    else:
        values = list(executor.map(run_one_worker, tasks))

    return float(np.mean(np.asarray(values, dtype=np.float64)))


def estimate_bcr(L, l_index, u_index, u_value, neighbors, executor=None):
    """Find the largest b for which mean(fc) >= FC_THRESHOLD."""
    b_low = B_LOW_INIT
    b_high = B_HIGH_INIT

    fc_low = average_fc(
        L, l_index, u_index, u_value, b_low, neighbors, executor
    )
    if fc_low < FC_THRESHOLD:
        return b_low

    fc_high = average_fc(
        L, l_index, u_index, u_value, b_high, neighbors, executor
    )
    if fc_high >= FC_THRESHOLD:
        return b_high

    while (b_high - b_low) > B_TOLERANCE:
        b_mid = 0.5 * (b_low + b_high)
        fc_mid = average_fc(
            L, l_index, u_index, u_value, b_mid, neighbors, executor
        )
        if fc_mid >= FC_THRESHOLD:
            b_low = b_mid
        else:
            b_high = b_mid

    return b_low


# ============================================================
# 4. Data and figure output
# ============================================================
def normalize_l_values(values):
    if np.isscalar(values):
        values = (int(values),)
    else:
        values = tuple(int(value) for value in values)

    if len(values) == 0 or any(value < 2 for value in values):
        raise ValueError("L_VALUES must contain positive lattice sizes >= 2.")
    return values


def make_u_values():
    point_count = int(round((U_MAX - U_MIN) / U_STEP))
    return np.linspace(U_MIN, U_MAX, point_count + 1)


def save_data(L, u_values, bcr_results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_name = OUTPUT_DIR / f"{ALPHA:.2f}bcr_vs_u_L{L}_data.txt"
    np.savetxt(
        str(data_name),
        np.column_stack((u_values, bcr_results)),
        fmt="%.6f",
        delimiter="\t",
        header="u\tbcr",
        comments="",
    )
    return data_name


def save_combined_plot(lattice_values, u_values, results_by_L):
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fallback_colors = [
        "#00AEB8", "#2146CB", "#E69F00", "#D62728", "#7A3DB8"
    ]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.labelsize": 23,
        "axes.linewidth": 1.3,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    colors = {
        50: "#7A3DB8",
        100: "#00AEB8",
        200: "#2146CB",
        300: "#E69F00",
        400: "#D62728",
    }
    line_style = dict(
        linestyle="-",
        linewidth=1.5,
        marker="+",
        markersize=6,
        markeredgewidth=1.2,
        markevery=1,
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained")
    for index, L in enumerate(lattice_values):
        u = np.asarray(u_values)
        bcr = np.asarray(results_by_L[L])
        color = colors.get(L, fallback_colors[index % len(fallback_colors)])

        ax.plot(
            u,
            bcr,
            label=rf"$L={L}$",
            solid_capstyle="butt",
            dash_capstyle="butt",
            color=color,
            zorder=2 + index,
            **line_style,
        )

        peak_index = int(np.argmax(bcr))
        ax.scatter(
            u[peak_index],
            bcr[peak_index],
            color=color,
            marker="o",
            s=45,
            linewidths=0,
            zorder=10 + index,
            label="_nolegend_",
        )
        print(
            f"L={L}: peak u={u[peak_index]:.6f}, "
            f"b_cr={bcr[peak_index]:.6f}"
        )

    ax.set_xlabel(r"$u$", labelpad=8)
    ax.set_ylabel(r"$b_{cr}$", labelpad=10)
    ax.set_xlim(0.0, 0.040)
    ax.set_ylim(1.0, 1.70)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:.2f}")
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.2,
    )
    ax.grid(False)
    legend = ax.legend(
        loc="upper right",
        fontsize=13,
        handlelength=2.8,
        frameon=True,
        framealpha=1,
        edgecolor="0.2",
        borderpad=0.4,
        labelspacing=0.3,
    )
    legend.get_frame().set_linewidth(1.0)

    for suffix in ("png", "eps"):
        path = OUTPUT_DIR / f"{ALPHA:.2f}bcr_vs_u_all_L_lines.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return OUTPUT_DIR / f"{ALPHA:.2f}bcr_vs_u_all_L_lines.png"


def main():
    if MCS_STABLE > MCS_MAX:
        raise ValueError("MCS_STABLE must not exceed MCS_MAX.")
    if U_STEP <= 0.0 or U_MAX < U_MIN:
        raise ValueError("Invalid mutation-rate grid.")
    if NUM_RUNS < 1:
        raise ValueError("NUM_RUNS must be at least 1.")

    lattice_values = normalize_l_values(L_VALUES)
    u_values = make_u_values()
    all_summary = []
    results_by_L = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Starting b_cr(u) calculations")
    print(f"alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, K={K}")
    print(f"u grid: {len(u_values)} points, step={U_STEP}")
    print(f"L values: {lattice_values}")
    print(
        f"MCS={MCS_MAX}, stable window={MCS_STABLE}, "
        f"independent runs={NUM_RUNS}"
    )
    print(
        "Reputation rule: global endorsement is fixed within each MCS; "
        "reputation is updated synchronously at the MCS boundary."
    )

    for l_index, L in enumerate(lattice_values):
        neighbors = build_neighbor_map(L)
        bcr_results = []

        print(f"\nCalculating L={L} (N={L * L})")
        print(
            f"Using {MAX_WORKERS}/{os.cpu_count() or 1} logical CPU cores "
            f"(target fraction={CPU_FRACTION:.0%})"
        )

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=initialize_worker,
            initargs=(L,),
        ) as executor:
            for u_index, u_value in enumerate(u_values):
                bcr = estimate_bcr(
                    L,
                    l_index,
                    u_index,
                    float(u_value),
                    neighbors,
                    executor,
                )
                bcr_results.append(bcr)
                print(f"L={L}, u={u_value:.4f} -> b_cr={bcr:.4f}")

        bcr_results = np.asarray(bcr_results, dtype=np.float64)
        results_by_L[L] = bcr_results
        data_name = save_data(L, u_values, bcr_results)

        for u_value, bcr in zip(u_values, bcr_results):
            all_summary.append([L, u_value, bcr])

        optimal_u_index = int(np.argmax(bcr_results))
        print(f"Saved data: {Path(data_name).resolve()}")
        print(
            f"L={L}: raw-grid optimum "
            f"u*={u_values[optimal_u_index]:.4f}, "
            f"b_cr={bcr_results[optimal_u_index]:.4f}"
        )

    summary_name = OUTPUT_DIR / f"{ALPHA:.2f}bcr_vs_u_all_L_summary.txt"
    np.savetxt(
        str(summary_name),
        np.asarray(all_summary),
        fmt=["%d", "%.6f", "%.6f"],
        delimiter="\t",
        header="L\tu\tbcr",
        comments="",
    )
    print(f"Saved finite-size summary: {summary_name.resolve()}")

    combined_figure = save_combined_plot(
        lattice_values,
        u_values,
        results_by_L,
    )
    print(f"Saved combined figure: {Path(combined_figure).resolve()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
