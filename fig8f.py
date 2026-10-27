# -*- coding: utf-8 -*-
"""Regularization-constant sensitivity analysis for the supplied model.

The benchmark model is kept fixed while epsilon is varied over several
orders of magnitude.  The script only prepares and runs the calculations;
it does not perform any post hoc fitting or smoothing of the curves.

Dependencies:
    python -m pip install numpy numba matplotlib tqdm

Example:
    python epsilon_sensitivity_analysis.py
    python epsilon_sensitivity_analysis.py --no-show

For every completed run the script writes individual-run TXT data, summary
TXT data, plot data TXT, a run report, and the figure in both PNG and EPS
formats.  Worker processes are single-threaded and use approximately 60% of
the logical CPU cores by default.
"""

import os

# Set these before NumPy/Numba imports.  Windows spawn workers execute this
# module again, so the setting also prevents nested BLAS/Numba oversubscription.
for _name in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
):
    os.environ[_name] = '1'

import argparse
import multiprocessing as mp
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
from tqdm import tqdm

# ==========================================================
# 1. Fixed benchmark parameters and epsilon scan
# ==========================================================
L = 100
N = L * L
FIXED_INITIAL_FC = 0.50
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_GAMMA = 0.10
FIXED_U = 0.00  # random strategy reset probability
FIXED_K = 0.10
INITIAL_REPUTATION = 0.50

# Positive values are used because epsilon regularizes zero-reputation
# denominators.  The reference value in the manuscript is epsilon=0.01.
EPSILON_VALUES = np.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=np.float64)

# Keep the b sampling and plot range used by the latest model-comparison code.
B_VALUES = np.round(np.linspace(1.0, 1.8, 33), 10)

MCS_TOTAL = 20000
MCS_TRANSIENT = 17000  # average over the final 3000 MCS
NUM_RUNS = 100

CPU_FRACTION = 0.60
BASE_SEED = 20261007
PNG_DPI = 600
EARLY_STOP_ABSORBING = True
ABSORB_CHECK_INTERVAL = 100


# ==========================================================
# 2. Periodic square lattice
# ==========================================================
def build_square_network():
    """Return a CSR-like periodic four-neighbor lattice."""
    nodes = np.arange(N, dtype=np.int64)
    row, col = nodes // L, nodes % L
    neighbors = np.empty((N, 4), dtype=np.int64)
    neighbors[:, 0] = ((row - 1 + L) % L) * L + col
    neighbors[:, 1] = ((row + 1) % L) * L + col
    neighbors[:, 2] = row * L + ((col - 1 + L) % L)
    neighbors[:, 3] = row * L + ((col + 1) % L)
    offsets = np.arange(N + 1, dtype=np.int64) * 4
    return offsets, neighbors.ravel()


# ==========================================================
# 3. Model dynamics
# ==========================================================
@njit(cache=True, inline='always')
def calculate_payoff(strategy_i, strategy_j, b):
    # Weak PD: R=1, S=0, T=b, P=0; C=1 and D=0.
    if strategy_i == 1:
        return 1.0 if strategy_j == 1 else 0.0
    return b if strategy_j == 1 else 0.0


@njit(cache=True, inline='always')
def calculate_fitness(node, b, alpha, beta, epsilon, global_c, strategies,
                      reputations, offsets, neighbors):
    strategy = strategies[node]
    payoff = 0.0
    numerator = reputations[node] + epsilon
    denominator = reputations[node] + epsilon

    for edge_index in range(offsets[node], offsets[node + 1]):
        neighbor = neighbors[edge_index]
        payoff += calculate_payoff(strategy, strategies[neighbor], b)
        weight = reputations[neighbor] + epsilon
        denominator += weight
        if strategies[neighbor] == strategy:
            numerator += weight

    local_endorsement = numerator / denominator
    global_endorsement = global_c if strategy == 1 else 1.0 - global_c

    endorsement = alpha * local_endorsement + (1.0 - alpha) * global_endorsement

    if endorsement < 1e-9:
        endorsement = 1e-9
    return payoff * (endorsement ** beta)


@njit(cache=True)
def simulation_step(noise_k, b, alpha, beta, gamma, epsilon, u,
                    strategies, reputations, offsets, neighbors):
    population = len(strategies)

    # Global endorsement is a snapshot taken at the beginning of the MCS.
    total_reputation = np.sum(reputations) + population * epsilon
    cooperator_reputation = 0.0
    for i in range(population):
        if strategies[i] == 1:
            cooperator_reputation += reputations[i] + epsilon
    global_c = cooperator_reputation / total_reputation

    # Each player is selected once per MCS in a random order.
    for i in np.random.permutation(population):
        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)
            continue

        degree_i = offsets[i + 1] - offsets[i]
        if degree_i == 0:
            continue
        j = neighbors[offsets[i] + np.random.randint(0, degree_i)]
        if strategies[i] == strategies[j]:
            continue

        fitness_i = calculate_fitness(
            i, b, alpha, beta, epsilon, global_c,
            strategies, reputations, offsets, neighbors,
        )
        fitness_j = calculate_fitness(
            j, b, alpha, beta, epsilon, global_c,
            strategies, reputations, offsets, neighbors,
        )

        exponent = (fitness_i - fitness_j) / noise_k
        if exponent > 20.0:
            imitation_probability = 0.0
        elif exponent < -20.0:
            imitation_probability = 1.0
        else:
            imitation_probability = 1.0 / (1.0 + np.exp(exponent))
        if np.random.random() < imitation_probability:
            strategies[i] = strategies[j]

    # Synchronous EMA reputation update at the end of each MCS.
    for i in range(population):
        reputations[i] = (1.0 - gamma) * reputations[i] + gamma * float(strategies[i])


@njit(cache=True)
def all_edges_agree(strategies, offsets, neighbors):
    for node in range(len(strategies)):
        for edge_index in range(offsets[node], offsets[node + 1]):
            if strategies[node] != strategies[neighbors[edge_index]]:
                return False
    return True


@njit(cache=True)
def run_single_experiment(initial_fc, noise_k, b, alpha, beta, gamma, epsilon,
                          u, initial_reputation, offsets, neighbors, seed,
                          mcs_total, mcs_transient, early_stop, check_interval):
    np.random.seed(seed)
    population = len(offsets) - 1
    initial_count = int(np.floor(initial_fc * population + 0.5))
    strategies = np.zeros(population, dtype=np.int64)
    initial_positions = np.random.permutation(population)
    for index in range(initial_count):
        strategies[initial_positions[index]] = 1

    reputations = np.full(population, initial_reputation, dtype=np.float64)
    actual_initial_fc = initial_count / population
    cooperation_sum = 0.0
    sample_count = mcs_total - mcs_transient

    for mcs in range(mcs_total):
        simulation_step(
            noise_k, b, alpha, beta, gamma, epsilon, u,
            strategies, reputations, offsets, neighbors,
        )
        cooperative_count = np.sum(strategies)
        cooperation = cooperative_count / population
        if mcs >= mcs_transient:
            cooperation_sum += cooperation

        if early_stop and u == 0.0:
            stop_code = 0
            if cooperative_count == 0 or cooperative_count == population:
                stop_code = 1
            elif (mcs + 1) % check_interval == 0 and all_edges_agree(
                    strategies, offsets, neighbors):
                stop_code = 2
            if stop_code:
                remaining_samples = mcs_total - max(mcs + 1, mcs_transient)
                cooperation_sum += remaining_samples * cooperation
                return (cooperation_sum / sample_count, actual_initial_fc,
                        cooperation, mcs + 1, stop_code)

    return (cooperation_sum / sample_count, actual_initial_fc,
            cooperation, mcs_total, 0)


# ==========================================================
# 4. Parallel independent runs
# ==========================================================
_WORKER_OFFSETS = None
_WORKER_NEIGHBORS = None


def initialize_worker():
    global _WORKER_OFFSETS, _WORKER_NEIGHBORS
    _WORKER_OFFSETS, _WORKER_NEIGHBORS = build_square_network()


def execute_task(task):
    epsilon_index, b_index, run_index, epsilon, b, seed = task
    started = perf_counter()
    result = run_single_experiment(
        FIXED_INITIAL_FC, FIXED_K, b, FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA,
        epsilon, FIXED_U, INITIAL_REPUTATION, _WORKER_OFFSETS,
        _WORKER_NEIGHBORS, seed, MCS_TOTAL, MCS_TRANSIENT,
        EARLY_STOP_ABSORBING, ABSORB_CHECK_INTERVAL,
    )
    return epsilon_index, b_index, run_index, epsilon, b, seed, result, (
            perf_counter() - started
    )


def make_tasks(epsilon_values, b_values):
    root = np.random.SeedSequence(BASE_SEED)
    total_tasks = len(epsilon_values) * len(b_values) * NUM_RUNS
    children = iter(root.spawn(total_tasks))
    used_seeds = set()
    tasks = []

    for epsilon_index, epsilon in enumerate(epsilon_values):
        for b_index, b in enumerate(b_values):
            for run_index in range(NUM_RUNS):
                seed = int(next(children).generate_state(1, dtype=np.uint32)[0])
                while seed in used_seeds:
                    seed = int(root.spawn(1)[0].generate_state(
                        1, dtype=np.uint32
                    )[0])
                used_seeds.add(seed)
                tasks.append((epsilon_index, b_index, run_index,
                              float(epsilon), float(b), seed))
    return tasks


# ==========================================================
# 5. Plotting: retain the supplied figure style
# ==========================================================
def epsilon_label(epsilon):
    if epsilon == 1.0:
        return r'$\epsilon=1$'
    exponent = int(np.round(np.log10(epsilon)))
    return rf'$\epsilon=10^{{{exponent}}}$'


def save_plot(epsilon_values, b_values, mean_data, output_dir, file_stem, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C', '#228B22']
    markers = ['*', '+', 'o', 'o', 's']
    figure, axis = plt.subplots(figsize=(5, 4.5))

    for index, epsilon in enumerate(epsilon_values):
        marker_face = 'none' if colors[index] == '#0000CD' else colors[index]
        marker_size = 8 if markers[index] == '*' else 6
        axis.plot(
            b_values, mean_data[index], label=epsilon_label(float(epsilon)),
            color=colors[index], marker=markers[index], markersize=marker_size,
            linewidth=1.2, markeredgecolor=colors[index],
            markerfacecolor=marker_face,
        )

    axis.set_ylim(0, 1.0)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axis.set_xlim(1.0, 1.8)
    axis.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4,
                     1.5, 1.6, 1.7, 1.8])
    axis.set_xticklabels([
        '1', '1.1', '1.2', '1.3', '1.4',
        '1.5', '1.6', '1.7', '1.8',
    ])
    axis.margins(0)
    axis.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
    axis.set_xlabel(r'$b$', fontsize=14)
    axis.set_ylabel(r'$f_c$', fontsize=14)

    # 唯一改动的部分：将 loc='lower left' 修改为 loc='upper right'
    axis.legend(fontsize=13, frameon=True, framealpha=1.0,
                edgecolor='black', loc='upper right', handlelength=1.5)

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


def write_header(handle, metadata, columns):
    for line in metadata.rstrip().splitlines():
        handle.write('# ' + line + '\n')
    handle.write('# ' + columns + '\n')
    handle.flush()


def summary_row(epsilon_index, b, values, early_flags):
    count = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if count > 1 else float('nan')
    sem = std / np.sqrt(count)
    return (
        f'{epsilon_index} {EPSILON_VALUES[epsilon_index]:.10g} {b:.8f} '
        f'{mean:.10f} {std:.10f} {sem:.10f} {count} '
        f'{int(np.sum(early_flags))}\n'
    )


def validate_parameters(epsilon_values, b_values):
    if not isinstance(L, int) or L < 3 or N != L * L:
        raise ValueError('L must be an integer >= 3 and N must equal L*L.')
    if (any(not isinstance(x, int) for x in
            (MCS_TOTAL, MCS_TRANSIENT, NUM_RUNS, ABSORB_CHECK_INTERVAL))
            or not 0 <= MCS_TRANSIENT < MCS_TOTAL or NUM_RUNS < 1):
        raise ValueError('Invalid MCS or repetition settings.')
    for name, value in (
            ('initial_fc', FIXED_INITIAL_FC), ('alpha', FIXED_ALPHA),
            ('beta', FIXED_BETA), ('gamma', FIXED_GAMMA), ('u', FIXED_U),
            ('initial_reputation', INITIAL_REPUTATION),
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + ' must lie in [0, 1].')
    if not np.isfinite(FIXED_K) or FIXED_K <= 0:
        raise ValueError('K must be positive.')
    if not (0 < CPU_FRACTION <= 1):
        raise ValueError('CPU_FRACTION must lie in (0, 1].')
    if (epsilon_values.ndim != 1 or len(epsilon_values) < 1
            or np.any(~np.isfinite(epsilon_values))
            or np.any(epsilon_values <= 0)
            or np.any(np.diff(epsilon_values) <= 0)):
        raise ValueError('EPSILON_VALUES must be positive and strictly increasing.')
    if (b_values.ndim != 1 or len(b_values) < 1
            or np.any(~np.isfinite(b_values))
            or np.any(b_values < 1.0) or np.any(b_values > 1.8)
            or np.any(np.diff(b_values) <= 0)):
        raise ValueError('B_VALUES must be strictly increasing in [1, 1.8].')


# ==========================================================
# 6. Main program and text output
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description='Sensitivity analysis of the reputation regularization epsilon.'
    )
    parser.add_argument('--no-show', action='store_true',
                        help='Save PNG/EPS without opening a plot window.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override worker count; default uses about 60% of cores.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'epsilon_sensitivity_results',
                        help='Output root; each invocation creates a timestamped directory.')
    arguments = parser.parse_args()

    epsilon_values = np.asarray(EPSILON_VALUES, dtype=np.float64)
    b_values = np.asarray(B_VALUES, dtype=np.float64)
    validate_parameters(epsilon_values, b_values)

    logical_cores = os.cpu_count() or 1
    workers = max(1, int(logical_cores * CPU_FRACTION))
    if arguments.workers is not None:
        workers = arguments.workers
    if not 1 <= workers <= logical_cores:
        raise ValueError('--workers must be between 1 and the logical core count.')

    tasks = make_tasks(epsilon_values, b_values)
    total_tasks = len(tasks)
    workers = min(workers, total_tasks)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = arguments.output_dir.resolve() / ('run_' + timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    file_stem = (
        f'epsilon_sensitivity_alpha{FIXED_ALPHA:.2f}_beta{FIXED_BETA:.2f}'
        f'_gamma{FIXED_GAMMA:.2f}_u{FIXED_U:.2f}_K{FIXED_K:.2f}_N{N}'
    )
    raw_path = output_dir / (file_stem + '_individual_runs.txt')
    summary_path = output_dir / (file_stem + '_summary.txt')
    plot_data_path = output_dir / (file_stem + '_plot_data.txt')
    report_path = output_dir / (file_stem + '_run_report.txt')

    library_versions = ' '.join(
        f'{name}={version(name)}'
        for name in ('numpy', 'numba', 'matplotlib', 'tqdm')
    )
    metadata = (
        f'alpha={FIXED_ALPHA} beta={FIXED_BETA} gamma={FIXED_GAMMA} '
        f'u_reset={FIXED_U} u_flip={FIXED_U / 2} K={FIXED_K}\n'
        f'N={N} L={L} network=periodic_square degree=4 '
        f'initial_fc={FIXED_INITIAL_FC} initial_reputation={INITIAL_REPUTATION}\n'
        f'epsilon_values={epsilon_values.tolist()} reference_epsilon=0.01\n'
        f'b_values={b_values.tolist()}\n'
        f'MCS_TOTAL={MCS_TOTAL} MCS_TRANSIENT={MCS_TRANSIENT} '
        f'averaging_MCS={MCS_TOTAL - MCS_TRANSIENT} NUM_RUNS={NUM_RUNS}\n'
        'payoff=weak_PD_R1_S0_Tb_P0; fitness=accumulated_payoff*endorsement**beta; '
        'local_endorsement=neighbors_plus_self; global_endorsement=start_of_MCS_snapshot\n'
        'update=random_permutation_each_MCS; reputation_update=synchronous_end_of_MCS; '
        'mutation=random_strategy_reset_without_reputation_reset\n'
        f'early_stop_absorbing={EARLY_STOP_ABSORBING} '
        f'all_edges_check_interval={ABSORB_CHECK_INTERVAL}; '
        'remaining_sampling_window_completed_exactly\n'
        'fc_run=mean_over_sampling_MCS; fc_mean=mean_of_run_means; '
        'fc_std=sample_std_ddof1; fc_sem=fc_std/sqrt(n_runs)\n'
        f'BASE_SEED={BASE_SEED} logical_cores={logical_cores} workers={workers} '
        f'CPU_FRACTION={CPU_FRACTION} worker_core_fraction={workers / logical_cores:.6f}\n'
        f'python={platform.python_version()} {library_versions}\n'
    )

    print(f'N={N}, alpha={FIXED_ALPHA}, beta={FIXED_BETA}, K={FIXED_K}, '
          f'f_C(0)={FIXED_INITIAL_FC}')
    print('Epsilon values:', epsilon_values.tolist())
    print(f'{len(epsilon_values)} epsilon values x {len(b_values)} b values '
          f'x {NUM_RUNS} runs = {total_tasks} independent runs')
    print(f'{MCS_TOTAL} MCS/run; average over final {MCS_TOTAL - MCS_TRANSIENT} MCS')
    print(f'Logical cores: {logical_cores}; workers: {workers} '
          f'({100 * workers / logical_cores:.1f}% of logical cores)')
    print('Output directory:', output_dir)

    shape = (len(epsilon_values), len(b_values), NUM_RUNS)
    run_values = np.full(shape, np.nan, dtype=np.float64)
    early_flags = np.zeros(shape, dtype=np.bool_)
    completed_counts = np.zeros(shape[:2], dtype=np.int64)
    stop_counts = np.zeros(3, dtype=np.int64)
    completed = 0
    full_runs = 0
    status = 'incomplete'
    started = perf_counter()
    raw_columns = (
        'epsilon_index epsilon b run_index seed fc_initial_actual fc_run '
        'fc_final executed_mcs absorbed_detected_mcs stop_code elapsed_seconds'
    )
    summary_columns = (
        'epsilon_index epsilon b fc_mean fc_std fc_sem n_runs n_early_stop'
    )

    try:
        with raw_path.open('w', encoding='utf-8') as raw_file, \
                summary_path.open('w', encoding='utf-8') as summary_file:
            write_header(raw_file, metadata, raw_columns)
            write_header(summary_file, metadata, summary_columns)
            context = mp.get_context('spawn')
            with context.Pool(processes=workers, initializer=initialize_worker) as pool:
                for returned in tqdm(
                        pool.imap_unordered(execute_task, tasks, chunksize=1),
                        total=total_tasks, desc='Epsilon sensitivity simulations', unit='run',
                ):
                    (epsilon_index, b_index, run_index, epsilon, b, seed,
                     result, run_seconds) = returned
                    fc, initial_fc, final_fc, executed_mcs, stop_code = result
                    if not np.isfinite(fc) or not 0.0 <= fc <= 1.0:
                        raise RuntimeError('Cooperation fraction is invalid.')
                    if not np.isnan(run_values[epsilon_index, b_index, run_index]):
                        raise RuntimeError('Duplicate task result detected.')

                    run_values[epsilon_index, b_index, run_index] = fc
                    early_flags[epsilon_index, b_index, run_index] = (
                            executed_mcs < MCS_TOTAL
                    )
                    completed_counts[epsilon_index, b_index] += 1
                    completed += 1
                    full_runs += int(executed_mcs == MCS_TOTAL)
                    stop_counts[stop_code] += 1
                    detected_mcs = executed_mcs if stop_code else 0
                    raw_file.write(
                        f'{epsilon_index} {epsilon:.10g} {b:.8f} {run_index + 1} '
                        f'{seed} {initial_fc:.10f} {fc:.10f} {final_fc:.10f} '
                        f'{executed_mcs} {detected_mcs} {stop_code} {run_seconds:.6f}\n'
                    )
                    raw_file.flush()
                    if completed_counts[epsilon_index, b_index] == NUM_RUNS:
                        summary_file.write(summary_row(
                            epsilon_index, b, run_values[epsilon_index, b_index],
                            early_flags[epsilon_index, b_index],
                        ))
                        summary_file.flush()

        if completed != total_tasks or np.any(completed_counts != NUM_RUNS):
            raise RuntimeError('The completed task count is insufficient.')
        if np.any(~np.isfinite(run_values)):
            raise RuntimeError('Missing finite run values.')

        # Rewrite the summary in deterministic epsilon,b order.
        temporary_summary = summary_path.with_suffix('.tmp')
        with temporary_summary.open('w', encoding='utf-8') as summary_file:
            write_header(summary_file, metadata, summary_columns)
            for epsilon_index in range(len(epsilon_values)):
                for b_index, b in enumerate(b_values):
                    summary_file.write(summary_row(
                        epsilon_index, b, run_values[epsilon_index, b_index],
                        early_flags[epsilon_index, b_index],
                    ))
        temporary_summary.replace(summary_path)

        mean_data = np.mean(run_values, axis=2)
        plot_matrix = np.column_stack((b_values, mean_data.T))
        np.savetxt(
            plot_data_path, plot_matrix, fmt='%.10f', encoding='utf-8',
            header=metadata + 'b ' + ' '.join(
                f'epsilon_{value:.10g}' for value in epsilon_values
            ),
        )
        status = 'simulation_and_text_complete'
    finally:
        elapsed = perf_counter() - started
        with report_path.open('w', encoding='utf-8') as report:
            write_header(report, metadata, 'run_report')
            report.write(
                f'status={status}\nexpected_runs={total_tasks}\n'
                f'completed_runs={completed}\n'
                f'complete_parameter_points={int(np.sum(completed_counts == NUM_RUNS))}\n'
                f'full_length_runs={full_runs}\n'
                f'early_stopped_runs={completed - full_runs}\n'
                f'stop_code_counts={stop_counts.tolist()}\n'
                f'elapsed_seconds={elapsed:.6f}\n'
                f'raw_data={raw_path.name}\nsummary_data={summary_path.name}\n'
            )

    print(f'Completed {completed}/{total_tasks}; full runs={full_runs}; '
          f'early stops={completed - full_runs}; elapsed={elapsed:.1f}s')
    print('Individual TXT:', raw_path)
    print('Summary TXT:', summary_path)
    print('Plot data TXT:', plot_data_path)
    print('Run report TXT:', report_path)
    save_plot(epsilon_values, b_values, mean_data, output_dir, file_stem,
              show=not arguments.no_show)
    with report_path.open('a', encoding='utf-8') as report:
        report.write('figure_export=complete_png_and_eps\n')


if __name__ == '__main__':
    mp.freeze_support()
    main()
