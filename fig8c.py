"""Compare initial cooperator fractions with the supplied spatial-game model.

Run this file directly in PyCharm or with:
    python initial_conditions_comparison.py
Results are saved in initial_condition_results next to this script.

This standalone script scans f_c(0) only. Initial cooperator counts are fixed
and their locations are randomized independently in every realization.
The model dynamics and plotting style follow the supplied source code.
About 60% of logical CPU cores are used as single-thread worker processes;
this is not an exact cap on total CPU utilization.
"""

import os

# Apply before NumPy/Numba imports, including in Windows child processes.
for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from numba import njit
from tqdm import tqdm


# ==========================================================
# 1. 正式模拟参数：只改变初始合作比例
# ==========================================================
INITIAL_FC_VALUES = [0.00, 0.30, 0.50, 0.80, 1.00]
L = 100
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_GAMMA = 0.10
FIXED_U = 0.00  # 随机策略重置概率；由突变引起的有效翻转概率为 u/2
FIXED_K = 0.10
EPSILON = 0.01

B_VALUES = np.linspace(1.0, 1.8, 33)
MCS_TOTAL = 20000
MCS_TRANSIENT = 17000  # 最后 3000 MCS 的合作比例取时间平均
NUM_RUNS = 100

CPU_FRACTION = 0.60
BASE_SEED = 20261007  # 每次独立运行分配不同子种子；固定本值便于复现
PNG_DPI = 600


# ==========================================================
# 2. 模型：收益、背书、随机重置、Fermi 模仿及 EMA 声誉更新
# ==========================================================
@njit(cache=True)
def build_neighbor_map(lattice_size):
    population = lattice_size * lattice_size
    neighbors = np.empty((population, 4), dtype=np.int64)
    for i in range(population):
        row, col = i // lattice_size, i % lattice_size
        neighbors[i, 0] = ((row - 1 + lattice_size) % lattice_size) * lattice_size + col
        neighbors[i, 1] = ((row + 1) % lattice_size) * lattice_size + col
        neighbors[i, 2] = row * lattice_size + ((col - 1 + lattice_size) % lattice_size)
        neighbors[i, 3] = row * lattice_size + ((col + 1) % lattice_size)
    return neighbors


@njit(cache=True)
def calculate_payoff(strategy_i, strategy_j, b):
    # Weak Prisoner's Dilemma: R=1, S=0, T=b, P=0; C=1 and D=0.
    if strategy_i == 1:
        return 1.0 if strategy_j == 1 else 0.0
    return b if strategy_j == 1 else 0.0


@njit(cache=True)
def simulation_step(population, noise_K, b, alpha, beta, gamma, epsilon, u,
                    strategies, reputations, neighbor_map):
    # Global reputation-weighted popularity is a snapshot at the start of MCS.
    total_reputation = np.sum(reputations) + population * epsilon
    cooperator_reputation = 0.0
    for i in range(population):
        if strategies[i] == 1:
            cooperator_reputation += reputations[i] + epsilon

    global_C = cooperator_reputation / total_reputation
    global_D = 1.0 - global_C

    # Preserve the source rule: each player is selected once in a random order per MCS.
    for i in np.random.permutation(population):
        # Mutation is a random reset; it is not a guaranteed strategy flip.
        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)
            continue

        j = neighbor_map[i, np.random.randint(0, 4)]
        if strategies[i] == strategies[j]:
            continue

        payoff_i = 0.0
        numerator_i = reputations[i] + epsilon
        denominator_i = reputations[i] + epsilon
        for neighbor_index in range(4):
            neighbor = neighbor_map[i, neighbor_index]
            payoff_i += calculate_payoff(strategies[i], strategies[neighbor], b)
            weight = reputations[neighbor] + epsilon
            denominator_i += weight
            if strategies[neighbor] == strategies[i]:
                numerator_i += weight
        local_i = numerator_i / denominator_i
        total_i = alpha * local_i + (1.0 - alpha) * (
            global_C if strategies[i] == 1 else global_D
        )
        if total_i < 1e-9:
            total_i = 1e-9
        fitness_i = payoff_i * (total_i ** beta)

        payoff_j = 0.0
        numerator_j = reputations[j] + epsilon
        denominator_j = reputations[j] + epsilon
        for neighbor_index in range(4):
            neighbor = neighbor_map[j, neighbor_index]
            payoff_j += calculate_payoff(strategies[j], strategies[neighbor], b)
            weight = reputations[neighbor] + epsilon
            denominator_j += weight
            if strategies[neighbor] == strategies[j]:
                numerator_j += weight
        local_j = numerator_j / denominator_j
        total_j = alpha * local_j + (1.0 - alpha) * (
            global_C if strategies[j] == 1 else global_D
        )
        if total_j < 1e-9:
            total_j = 1e-9
        fitness_j = payoff_j * (total_j ** beta)

        exponent = (fitness_i - fitness_j) / noise_K
        if exponent > 20.0:
            imitation_probability = 0.0
        elif exponent < -20.0:
            imitation_probability = 1.0
        else:
            imitation_probability = 1.0 / (1.0 + np.exp(exponent))
        if np.random.random() < imitation_probability:
            strategies[i] = strategies[j]

    # Synchronous EMA reputation update at the end of MCS.
    for i in range(population):
        reputations[i] = (1.0 - gamma) * reputations[i] + gamma * float(strategies[i])
    return strategies, reputations


@njit(cache=True)
def run_single_experiment(initial_fraction, noise_K, b, alpha, beta, gamma, u, neighbor_map,
                          seed, mcs_total, mcs_transient):
    np.random.seed(seed)
    population = neighbor_map.shape[0]
    # Fix the initial count; randomize only the positions of cooperators.
    # This implements the specified f_c(0), rather than a Bernoulli expectation.
    number_of_cooperators = int(np.floor(initial_fraction * population + 0.5))
    strategies = np.zeros(population, dtype=np.int64)
    initial_positions = np.random.permutation(population)
    for position_index in range(number_of_cooperators):
        strategies[initial_positions[position_index]] = 1
    reputations = np.full(population, 0.5, dtype=np.float64)
    initial_cooperation = np.mean(strategies)
    cooperation_sum = 0.0
    sample_count = mcs_total - mcs_transient
    absorbed_mcs = 0

    for mcs in range(mcs_total):
        strategies, reputations = simulation_step(
            population, noise_K, b, alpha, beta, gamma, EPSILON, u,
            strategies, reputations, neighbor_map
        )
        cooperation = np.mean(strategies)
        if mcs >= mcs_transient:
            cooperation_sum += cooperation

        # With u=0, all-C and all-D are absorbing states. Complete the average
        # analytically if an absorbing state is reached before MCS_TOTAL.
        if u == 0.0 and (cooperation == 0.0 or cooperation == 1.0):
            if absorbed_mcs == 0:
                absorbed_mcs = mcs + 1
            remaining_samples = mcs_total - max(mcs + 1, mcs_transient)
            cooperation_sum += remaining_samples * cooperation
            return (cooperation_sum / sample_count, initial_cooperation, absorbed_mcs)

    return cooperation_sum / sample_count, initial_cooperation, absorbed_mcs



# ==========================================================
# 3. 多进程：并行独立运行，不并行更新同一张晶格
# ==========================================================
_WORKER_NEIGHBORS = None
_WORKER_TOTAL = None
_WORKER_TRANSIENT = None


def initialize_worker(lattice_size, mcs_total, mcs_transient):
    global _WORKER_NEIGHBORS, _WORKER_TOTAL, _WORKER_TRANSIENT
    _WORKER_NEIGHBORS = build_neighbor_map(lattice_size)
    _WORKER_TOTAL = mcs_total
    _WORKER_TRANSIENT = mcs_transient


def execute_task(task):
    initial_index, b_index, run_index, initial_fraction, b, seed = task
    fc, actual_initial_fc, absorbed_mcs = run_single_experiment(
        initial_fraction, FIXED_K, b, FIXED_ALPHA, FIXED_BETA,
        FIXED_GAMMA, FIXED_U, _WORKER_NEIGHBORS, seed,
        _WORKER_TOTAL, _WORKER_TRANSIENT
    )
    return (initial_index, b_index, run_index, seed, actual_initial_fc,
            fc, absorbed_mcs)


# ==========================================================
# 4. 绘图：保持源代码的尺寸、颜色、标记、刻度及图例位置
# ==========================================================
def save_plot(initial_values, b_values, mean_data, output_dir, file_stem, show):
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C', '#228B22']
    markers = ['*', '+', 'o', 'o', 's']
    figure, axis = plt.subplots(figsize=(5, 4.5))

    for index, initial_fraction in enumerate(initial_values):
        marker_face = 'none' if colors[index] == '#0000CD' else colors[index]
        marker_size = 8 if markers[index] == '*' else 6
        axis.plot(
            b_values, mean_data[index],
            label=rf'$f_c(0)={initial_fraction:.2f}$',
            color=colors[index], marker=markers[index], markersize=marker_size,
            linewidth=1.2, markeredgecolor=colors[index],
            markerfacecolor=marker_face
        )

    axis.set_ylim(0, 1.0)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axis.set_xlim(1.0, 1.8)
    axis.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    axis.set_xticklabels(['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8'])
    axis.margins(0)
    axis.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
    axis.set_xlabel(r'$b$', fontsize=16)
    axis.set_ylabel(r'$f_c$', fontsize=16)

    # 🌟 修改点：将 loc='upper right' 改为 loc='lower left'，将图例移至左下角
    axis.legend(fontsize=12, frameon=True, framealpha=1.0,
                edgecolor='black', loc='lower left', handlelength=1.5)

    plt.tight_layout()

    png_path = output_dir / (file_stem + '.png')
    eps_path = output_dir / (file_stem + '.eps')
    figure.savefig(png_path, dpi=PNG_DPI, bbox_inches='tight')
    figure.savefig(eps_path, dpi=PNG_DPI, bbox_inches='tight', format='eps')
    print('PNG:', png_path)
    print('EPS:', eps_path)
    if show:
        plt.show()
    plt.close(figure)


def write_header(file_handle, metadata, columns):
    for line in metadata.rstrip().splitlines():
        file_handle.write('# ' + line + '\n')
    file_handle.write('# ' + columns + '\n')
    file_handle.flush()


def write_summary_row(file_handle, initial_fraction, b, values):
    number_of_runs = len(values)
    mean = float(np.mean(values))
    std = (float(np.std(values, ddof=1))
           if number_of_runs > 1 else float('nan'))
    sem = std / np.sqrt(number_of_runs)
    file_handle.write(
        '{:.8f} {:.8f} {:.10f} {:.10f} {:.10f} {}\n'.format(
            initial_fraction, b, mean, std, sem, number_of_runs
        )
    )
    file_handle.flush()


def main():
    parser = argparse.ArgumentParser(
        description='Compare initial cooperator fractions with parallel independent runs.'
    )
    parser.add_argument('--no-show', action='store_true',
                        help='Save PNG/EPS without opening a plot window.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Optional worker-count override; default uses about 60%% of logical cores.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'initial_condition_results')
    arguments = parser.parse_args()

    initial_values = np.asarray(INITIAL_FC_VALUES, dtype=np.float64)
    b_values = np.asarray(B_VALUES, dtype=np.float64)
    if (len(initial_values) != 5 or np.any(~np.isfinite(initial_values))
            or np.any(initial_values < 0) or np.any(initial_values > 1)
            or len(np.unique(initial_values)) != len(initial_values)):
        raise ValueError('Use five distinct initial fractions in [0, 1].')
    if L < 3 or int(L) != L:
        raise ValueError('L must be an integer >= 3.')
    if not (0 <= MCS_TRANSIENT < MCS_TOTAL) or NUM_RUNS < 1:
        raise ValueError('Require 0 <= MCS_TRANSIENT < MCS_TOTAL and NUM_RUNS >= 1.')
    if not np.isfinite(FIXED_K) or FIXED_K <= 0:
        raise ValueError('FIXED_K must be positive.')
    if not (0 <= FIXED_GAMMA <= 1) or not (0 <= FIXED_U <= 1):
        raise ValueError('FIXED_GAMMA and FIXED_U must lie in [0, 1].')
    if not (0 < CPU_FRACTION <= 1):
        raise ValueError('CPU_FRACTION must lie in (0, 1].')
    if b_values.ndim != 1 or len(b_values) == 0 or np.any(~np.isfinite(b_values)):
        raise ValueError('B_VALUES must contain finite b values.')

    population = L * L
    total_tasks = len(initial_values) * len(b_values) * NUM_RUNS
    logical_cores = os.cpu_count() or 1
    workers = max(1, int(logical_cores * CPU_FRACTION))
    if arguments.workers is not None:
        workers = arguments.workers
    if not (1 <= workers <= logical_cores):
        raise ValueError('--workers must be between 1 and the logical CPU count.')
    workers = min(workers, total_tasks)

    # 主进程统一分配种子；记录各次种子，结果不依赖并行任务完成顺序。
    seed_sequence = np.random.SeedSequence(BASE_SEED)
    child_sequences = seed_sequence.spawn(total_tasks)
    tasks = []
    used_seeds = set()
    task_number = 0
    for initial_index, initial_fraction in enumerate(initial_values):
        for b_index, b in enumerate(b_values):
            for run_index in range(NUM_RUNS):
                seed = int(child_sequences[task_number].generate_state(1)[0])
                while seed in used_seeds:
                    seed = int(seed_sequence.spawn(1)[0].generate_state(1)[0])
                used_seeds.add(seed)
                tasks.append((initial_index, b_index, run_index,
                              float(initial_fraction), float(b), seed))
                task_number += 1

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        'initial_conditions_alpha{:.2f}_beta{:.2f}_gamma{:.2f}_u{:.2f}_K{:.2f}_L{}'.format(
            FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U, FIXED_K, L
        )
    )
    summary_path = output_dir / (file_stem + '_summary.txt')
    metadata = (
        'alpha={} beta={} gamma={} u_reset={} u_flip={} K={} epsilon={}\n'.format(
            FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U,
            FIXED_U / 2, FIXED_K, EPSILON
        )
        + 'L={} N={} initial_fc_values={}\n'.format(
            L, population, initial_values.tolist()
        )
        + 'MCS_TOTAL={} MCS_TRANSIENT={} averaging_MCS={} NUM_RUNS={}\n'.format(
            MCS_TOTAL, MCS_TRANSIENT, MCS_TOTAL - MCS_TRANSIENT, NUM_RUNS
        )
        + 'logical_cores={} workers={} CPU_core_fraction={} BASE_SEED={} seed_entropy={}\n'.format(
            logical_cores, workers, CPU_FRACTION, BASE_SEED, seed_sequence.entropy
        )
        + 'initialization=fixed_cooperator_count_random_positions; '
          'count=round_half_up(fc_initial_target*N); reputation_initial=0.5\n'
        + 'periodic_boundary=True degree=4; update=random_permutation_per_MCS; '
          'global_endorsement=start_of_MCS_snapshot; '
          'reputation_update=synchronous_end_of_MCS\n'
        + 'fc_run=mean_over_sampling_MCS; fc_mean=mean_of_independent_run_means; '
          'fc_std=sample_std(ddof=1); fc_sem=fc_std/sqrt(NUM_RUNS)\n'
        + 'u=0 absorbing states use exact continuation for the sampling window.\n'
    )

    print('Initial cooperator fractions:', initial_values.tolist())
    print('Fixed alpha={}, beta={}, gamma={}, u={}, K={}, L={}'.format(
        FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U, FIXED_K, L
    ))
    print('Logical cores: {}; worker processes: {} ({:.1f}% of cores).'.format(
        logical_cores, workers, 100 * workers / logical_cores
    ))
    print('CPU usage varies during startup, disk writes and plotting.')
    print('Output directory:', output_dir)

    run_values = np.full((len(initial_values), len(b_values), NUM_RUNS), np.nan)
    completed_counts = np.zeros((len(initial_values), len(b_values)), dtype=np.int64)
    summary_columns = 'fc_initial_target b fc_mean fc_std fc_sem n_runs'

    # 数据只由主进程写入，避免多进程写入造成文件交错。
    with summary_path.open('w', encoding='utf-8') as summary_file:
        write_header(summary_file, metadata, summary_columns)
        context = mp.get_context('spawn')
        with context.Pool(
            processes=workers,
            initializer=initialize_worker,
            initargs=(L, MCS_TOTAL, MCS_TRANSIENT),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(execute_task, tasks, chunksize=1),
                total=total_tasks, desc='Initial-condition simulations', unit='run'
            ):
                (initial_index, b_index, run_index, seed,
                 actual_initial_fc, fc, absorbed_mcs) = result
                run_values[initial_index, b_index, run_index] = fc
                completed_counts[initial_index, b_index] += 1
                initial_fraction = initial_values[initial_index]
                b = b_values[b_index]

                if completed_counts[initial_index, b_index] == NUM_RUNS:
                    write_summary_row(summary_file, initial_fraction, b,
                                      run_values[initial_index, b_index])

        # 正式完成后按初始比例、b 排序汇总数据。
        summary_file.seek(0)
        summary_file.truncate()
        write_header(summary_file, metadata, summary_columns)
        for initial_index, initial_fraction in enumerate(initial_values):
            for b_index, b in enumerate(b_values):
                write_summary_row(summary_file, initial_fraction, b,
                                  run_values[initial_index, b_index])

    print('Summary TXT:', summary_path)
    mean_data = np.mean(run_values, axis=2)
    save_plot(initial_values, b_values, mean_data, output_dir, file_stem,
              show=not arguments.no_show)


if __name__ == '__main__':
    mp.freeze_support()
    main()
