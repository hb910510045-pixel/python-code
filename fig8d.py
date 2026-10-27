"""Noise-level robustness test for the supplied spatial reputation model.

This script only generates the requested simulation code.  It scans K while
keeping alpha=0.80 and beta=0.50, preserves the original figure style, and
saves TXT, PNG, and EPS files.  The default worker count is approximately
60% of the logical CPU cores; each worker is restricted to one numerical
thread so that processes do not oversubscribe the CPU.
"""

import os

# Set these before importing NumPy/Numba in the main process and spawned workers.
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
# 1. Parameters: scan K only
# ==========================================================
L = 100
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_GAMMA = 0.10
FIXED_U = 0.00
EPSILON = 0.01
INITIAL_C_PROBABILITY = 0.5

# Five curves preserve the original plotting format. K=0.10 is the baseline.
K_VALUES = [0.01, 0.05, 0.10, 0.20, 0.30]
B_VALUES = np.linspace(1.0, 1.8, 33)
MCS_TOTAL = 20000
MCS_TRANSIENT = 17000
NUM_RUNS = 100
CPU_FRACTION = 0.60

# None gives a new random batch at each launch. Set an integer to reproduce it.
BASE_SEED = 20261007
PNG_DPI = 600


# ==========================================================
# 2. Model functions
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

    # Random asynchronous updating: every player is selected once on average.
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
def run_single_experiment(noise_K, b, alpha, beta, gamma, u, neighbor_map,
                          seed, mcs_total, mcs_transient):
    np.random.seed(seed)
    population = neighbor_map.shape[0]
    strategies = (
        np.random.random(population) < INITIAL_C_PROBABILITY
    ).astype(np.int64)
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
# 3. Multiprocessing worker functions
# ==========================================================
_WORKER_NEIGHBOR_MAP = None
_WORKER_MCS_TOTAL = None
_WORKER_MCS_TRANSIENT = None


def initialize_worker(lattice_size, mcs_total, mcs_transient):
    global _WORKER_NEIGHBOR_MAP, _WORKER_MCS_TOTAL, _WORKER_MCS_TRANSIENT
    _WORKER_NEIGHBOR_MAP = build_neighbor_map(lattice_size)
    _WORKER_MCS_TOTAL = mcs_total
    _WORKER_MCS_TRANSIENT = mcs_transient


def execute_task(task):
    k_index, b_index, run_index, noise_K, b, seed = task
    cooperation, initial_cooperation, absorbed_mcs = run_single_experiment(
        noise_K, b, FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U,
        _WORKER_NEIGHBOR_MAP, seed, _WORKER_MCS_TOTAL, _WORKER_MCS_TRANSIENT
    )
    return (k_index, b_index, run_index, seed, cooperation,
            initial_cooperation, absorbed_mcs)


# ==========================================================
# 4. Plotting and output
# ==========================================================
def save_plot(k_values, b_values, mean_data, output_dir, file_stem, show):
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C', '#228B22']
    markers = ['*', '+', 'o', 'o', 's']
    figure, axis = plt.subplots(figsize=(5, 4.5))

    for index, noise_K in enumerate(k_values):
        marker_face = 'none' if colors[index] == '#0000CD' else colors[index]
        marker_size = 8 if markers[index] == '*' else 6
        axis.plot(
            b_values, mean_data[index], label=rf'$K={noise_K:.2f}$',
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
    axis.legend(fontsize=14, frameon=True, framealpha=1.0,
                edgecolor='black', loc='lower left', handlelength=1.5)
    plt.tight_layout()

    png_path = output_dir / (file_stem + '.png')
    eps_path = output_dir / (file_stem + '.eps')
    figure.savefig(png_path, dpi=PNG_DPI, bbox_inches='tight')
    figure.savefig(eps_path, dpi=PNG_DPI, bbox_inches='tight', format='eps')
    if show:
        plt.show()
    plt.close(figure)
    print('PNG:', png_path)
    print('EPS:', eps_path)


def write_header(file_handle, metadata, columns):
    for line in metadata.rstrip().splitlines():
        file_handle.write('# ' + line + '\n')
    file_handle.write('# ' + columns + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true',
                        help='Short test configuration; not for manuscript data.')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override the default approximately-60%% worker count.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'noise_level_results')
    arguments = parser.parse_args()

    if arguments.smoke_test:
        lattice_size = 10
        k_values = np.array([0.05, 0.10, 0.20, 0.50, 1.00])
        b_values = np.array([1.0, 1.4, 1.8])
        number_of_runs, mcs_total, mcs_transient = 3, 60, 40
        suffix = '_SMOKE_TEST'
    else:
        lattice_size = L
        k_values = np.asarray(K_VALUES, dtype=np.float64)
        b_values = np.asarray(B_VALUES, dtype=np.float64)
        number_of_runs, mcs_total, mcs_transient = NUM_RUNS, MCS_TOTAL, MCS_TRANSIENT
        suffix = ''

    if len(k_values) != 5 or np.any(k_values <= 0):
        raise ValueError('Exactly five positive K values are required for the original plot style.')
    if arguments.workers is None:
        logical_cores = os.cpu_count() or 1
        workers = max(1, int(logical_cores * CPU_FRACTION))
    else:
        workers = arguments.workers
        logical_cores = os.cpu_count() or 1
    if workers < 1 or workers > logical_cores:
        raise ValueError('workers must be between 1 and the logical CPU count.')

    total_tasks = len(k_values) * len(b_values) * number_of_runs
    seed_sequence = np.random.SeedSequence(BASE_SEED)
    child_sequences = seed_sequence.spawn(total_tasks)
    tasks = []
    task_number = 0
    for k_index, noise_K in enumerate(k_values):
        for b_index, b in enumerate(b_values):
            for run_index in range(number_of_runs):
                seed = int(child_sequences[task_number].generate_state(1)[0])
                tasks.append((k_index, b_index, run_index,
                              float(noise_K), float(b), seed))
                task_number += 1

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        'noise_levels_alpha{:.2f}_beta{:.2f}_gamma{:.2f}_u{:.2f}_L{}{}'.format(
            FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U, lattice_size, suffix
        )
    )
    summary_path = output_dir / (file_stem + '_summary.txt')

    metadata = (
        'alpha={} beta={} gamma={} u_reset={} u_flip={} epsilon={}\n'.format(
            FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U, FIXED_U / 2, EPSILON
        )
        + 'L={} N={} K_values={}\n'.format(lattice_size, lattice_size ** 2, k_values.tolist())
        + 'MCS_TOTAL={} MCS_TRANSIENT={} NUM_RUNS={}\n'.format(
            mcs_total, mcs_transient, number_of_runs
        )
        + 'logical_cores={} workers={} CPU_target={} seed_entropy={}\n'.format(
            logical_cores, workers, CPU_FRACTION, seed_sequence.entropy
        )
        + 'initialization=independent_Bernoulli(p_C={}); reputation_initial=0.5\n'.format(
            INITIAL_C_PROBABILITY
        )
        + 'periodic_boundary=True degree=4; asynchronous_strategy_update=True; '
          'synchronous_reputation_update=True\n'
        + 'fc_mean=mean_of_independent_run_time_means; '
          'fc_std=sample_std(ddof=1); fc_sem=fc_std/sqrt(NUM_RUNS)\n'
    )

    run_values = np.full((len(k_values), len(b_values), number_of_runs), np.nan)

    with summary_path.open('w', encoding='utf-8') as summary_file:
        write_header(summary_file, metadata, 'K b fc_mean fc_std fc_sem n_runs')

        # Only the main process writes files. This prevents interleaved TXT output.
        context = mp.get_context('spawn')
        with context.Pool(
            processes=workers,
            initializer=initialize_worker,
            initargs=(lattice_size, mcs_total, mcs_transient),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(execute_task, tasks, chunksize=1),
                total=total_tasks, desc='Noise-level simulations', unit='run'
            ):
                (k_index, b_index, run_index, seed, cooperation,
                 initial_cooperation, absorbed_mcs) = result
                run_values[k_index, b_index, run_index] = cooperation

        for k_index, noise_K in enumerate(k_values):
            for b_index, b in enumerate(b_values):
                values = run_values[k_index, b_index]
                mean = np.mean(values)
                standard_deviation = np.std(values, ddof=1)
                standard_error = standard_deviation / np.sqrt(number_of_runs)
                summary_file.write(
                    '{:.8f} {:.8f} {:.10f} {:.10f} {:.10f} {}\n'.format(
                        noise_K, b, mean, standard_deviation,
                        standard_error, number_of_runs
                    )
                )

    mean_data = np.mean(run_values, axis=2)
    save_plot(k_values, b_values, mean_data, output_dir, file_stem,
              show=not arguments.no_show)
    print('Summary TXT:', summary_path)


if __name__ == '__main__':
    mp.freeze_support()
    main()

