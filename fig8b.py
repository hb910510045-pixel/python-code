
import os

# 必须在NumPy/Numba导入之前设置，Windows spawn子进程也会执行这些设置。
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

import networkx as nx
import numpy as np
from numba import njit
from tqdm import tqdm


# ==========================================================
# 1. 参数：与原代码相同，只将扫描变量换成网络类型
# ==========================================================
NETWORK_TYPES = ('square', 'rrg', 'ws', 'moore')
NETWORK_LABELS = (
    'Square lattice', 'Random regular', 'WS small-world',
    'Moore lattice',
)

L = 100
N = L * L
TARGET_DEGREE = 4
WS_REWIRE_P = 0.10

FIXED_INITIAL_FC = 0.50
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_GAMMA = 0.10
FIXED_U = 0.00                 # 随机重置概率，有效翻转概率为u/2
FIXED_K = 0.10
EPSILON = 0.01
INITIAL_REPUTATION = 0.50

B_VALUES = np.linspace(1.0, 2.0, 33)
MCS_TOTAL = 20000
MCS_TRANSIENT = 17000
NUM_RUNS = 100

CPU_FRACTION = 0.60
BASE_SEED = 20261007
PNG_DPI = 600
EARLY_STOP_ABSORBING = True
ABSORB_CHECK_INTERVAL = 100    # 每100 MCS检查一次是否所有边两端都同策略


# ==========================================================
# 2. 通用CSR邻接表：适应不同节点的不同度数
# ==========================================================
def build_square_network():
    # 邻居顺序保持原代码的上、下、左、右。
    nodes = np.arange(N, dtype=np.int64)
    row, col = nodes // L, nodes % L
    neighbors = np.empty((N, 4), dtype=np.int64)
    neighbors[:, 0] = ((row - 1 + L) % L) * L + col
    neighbors[:, 1] = ((row + 1) % L) * L + col
    neighbors[:, 2] = row * L + ((col - 1 + L) % L)
    neighbors[:, 3] = row * L + ((col + 1) % L)
    offsets = np.arange(N + 1, dtype=np.int64) * 4
    # N, E, mean_degree, degree_std, min_degree, max_degree,
    # isolated_nodes, connected_components, largest_component_size
    stats = (N, 2 * N, 4.0, 0.0, 4, 4, 0, 1, N)
    return offsets, neighbors.ravel(), stats


def build_moore_network():
    # 周期性Moore邻居：上、下、左、右及四个对角邻居，固定度数为8。
    nodes = np.arange(N, dtype=np.int64)
    row, col = nodes // L, nodes % L
    neighbors = np.empty((N, 8), dtype=np.int64)
    neighbors[:, 0] = ((row - 1 + L) % L) * L + col
    neighbors[:, 1] = ((row + 1) % L) * L + col
    neighbors[:, 2] = row * L + ((col - 1 + L) % L)
    neighbors[:, 3] = row * L + ((col + 1) % L)
    neighbors[:, 4] = ((row - 1 + L) % L) * L + ((col - 1 + L) % L)
    neighbors[:, 5] = ((row - 1 + L) % L) * L + ((col + 1) % L)
    neighbors[:, 6] = ((row + 1) % L) * L + ((col - 1 + L) % L)
    neighbors[:, 7] = ((row + 1) % L) * L + ((col + 1) % L)
    offsets = np.arange(N + 1, dtype=np.int64) * 8
    stats = (N, 4 * N, 8.0, 0.0, 8, 8, 0, 1, N)
    return offsets, neighbors.ravel(), stats


def graph_to_csr(graph):
    if graph.number_of_nodes() != N:
        raise ValueError('生成的网络节点数与N不符。')
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError('必须使用无向、无自环、无重边网络。')

    degrees = np.fromiter((graph.degree(i) for i in range(N)),
                          dtype=np.int64, count=N)
    offsets = np.empty(N + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degrees, out=offsets[1:])
    neighbors = np.empty(int(offsets[-1]), dtype=np.int64)
    for node in range(N):
        # 固定邻居顺序，使种子可复现；不依赖图内部的边插入顺序。
        neighbors[offsets[node]:offsets[node + 1]] = sorted(graph.neighbors(node))

    component_sizes = [len(c) for c in nx.connected_components(graph)]
    stats = (
        N, graph.number_of_edges(), float(np.mean(degrees)),
        float(np.std(degrees)), int(degrees.min()), int(degrees.max()),
        int(np.count_nonzero(degrees == 0)), len(component_sizes),
        max(component_sizes),
    )
    return offsets, neighbors, stats


def build_random_network(network_type, network_seed):
    if network_type == 'rrg':
        graph = nx.random_regular_graph(TARGET_DEGREE, N, seed=network_seed)
    elif network_type == 'ws':
        graph = nx.watts_strogatz_graph(N, TARGET_DEGREE, WS_REWIRE_P,
                                      seed=network_seed)
    else:
        raise ValueError('未知网络类型：' + network_type)
    # 不以“是否连通”为条件重抽样，也不改变边或节点。
    return graph_to_csr(graph)


# ==========================================================
# 3. 动力学：累计收益、含自身的局部背书、Fermi模仿与EMA
# ==========================================================
@njit(cache=True, inline='always')
def calculate_payoff(strategy_i, strategy_j, b):
    # Weak PD: R=1, S=0, T=b, P=0；C=1、D=0。
    if strategy_i == 1:
        return 1.0 if strategy_j == 1 else 0.0
    return b if strategy_j == 1 else 0.0


@njit(cache=True, inline='always')
def calculate_fitness(node, b, alpha, beta, epsilon, global_C, global_D,
                      strategies, reputations, offsets, neighbors):
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
    global_endorsement = global_C if strategy == 1 else global_D
    endorsement = alpha * local_endorsement + (1.0 - alpha) * global_endorsement
    if endorsement < 1e-9:
        endorsement = 1e-9
    # 延续原代码的累计收益，不除以度数。
    return payoff * (endorsement ** beta)


@njit(cache=True)
def simulation_step(population, noise_K, b, alpha, beta, gamma, epsilon, u,
                    strategies, reputations, offsets, neighbors):
    # 全局背书在每个MCS开始时计算一次快照。
    total_reputation = np.sum(reputations) + population * epsilon
    cooperator_reputation = 0.0
    for i in range(population):
        if strategies[i] == 1:
            cooperator_reputation += reputations[i] + epsilon
    global_C = cooperator_reputation / total_reputation
    global_D = 1.0 - global_C

    # 每个MCS将全部节点随机排列，每个节点恰好获得一次更新机会。
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
            i, b, alpha, beta, epsilon, global_C, global_D,
            strategies, reputations, offsets, neighbors,
        )
        fitness_j = calculate_fitness(
            j, b, alpha, beta, epsilon, global_C, global_D,
            strategies, reputations, offsets, neighbors,
        )
        exponent = (fitness_i - fitness_j) / noise_K
        # 保留原代码的指数截断规则。
        if exponent > 20.0:
            imitation_probability = 0.0
        elif exponent < -20.0:
            imitation_probability = 1.0
        else:
            imitation_probability = 1.0 / (1.0 + np.exp(exponent))
        if np.random.random() < imitation_probability:
            strategies[i] = strategies[j]

    # 声誉只在MCS结束时同步更新；突变不直接重置声誉。
    for i in range(population):
        reputations[i] = (1.0 - gamma) * reputations[i] + gamma * float(strategies[i])


@njit(cache=True)
def all_edges_agree(strategies, offsets, neighbors):
    # 断连网络中只要每个连通分量同策略，u=0时策略也永久不变。
    for node in range(len(strategies)):
        for edge_index in range(offsets[node], offsets[node + 1]):
            if strategies[node] != strategies[neighbors[edge_index]]:
                return False
    return True


@njit(cache=True)
def run_single_experiment(initial_fraction, noise_K, b, alpha, beta, gamma, u,
                          epsilon, initial_reputation, offsets, neighbors, seed,
                          mcs_total, mcs_transient, early_stop, check_interval):
    np.random.seed(seed)
    population = len(offsets) - 1
    number_of_cooperators = int(np.floor(initial_fraction * population + 0.5))
    strategies = np.zeros(population, dtype=np.int64)
    initial_positions = np.random.permutation(population)
    for index in range(number_of_cooperators):
        strategies[initial_positions[index]] = 1
    reputations = np.full(population, initial_reputation, dtype=np.float64)
    initial_cooperation = number_of_cooperators / population
    cooperation_sum = 0.0
    sample_count = mcs_total - mcs_transient
    cooperation = initial_cooperation

    for mcs in range(mcs_total):
        simulation_step(population, noise_K, b, alpha, beta, gamma, epsilon, u,
                        strategies, reputations, offsets, neighbors)
        number_of_cooperators = np.sum(strategies)
        cooperation = number_of_cooperators / population
        if mcs >= mcs_transient:
            cooperation_sum += cooperation

        stop_code = 0
        if early_stop and u == 0.0:
            if number_of_cooperators == 0 or number_of_cooperators == population:
                stop_code = 1
            elif (mcs + 1) % check_interval == 0:
                if all_edges_agree(strategies, offsets, neighbors):
                    stop_code = 2
        if stop_code:
            # mcs是从0开始的索引，mcs+1才是已执行MCS数。
            # 当前采样点已在上方累加；这里只补齐尚未执行的采样点。
            remaining_samples = mcs_total - max(mcs + 1, mcs_transient)
            cooperation_sum += remaining_samples * cooperation
            return (cooperation_sum / sample_count, initial_cooperation,
                    cooperation, mcs + 1, stop_code)

    return (cooperation_sum / sample_count, initial_cooperation,
            cooperation, mcs_total, 0)


# ==========================================================
# 4. 并行独立运行：单个网络的异步策略更新不做多线程并行
# ==========================================================
_WORKER_SQUARE = None
_WORKER_MOORE = None


def initialize_worker():
    global _WORKER_SQUARE, _WORKER_MOORE
    _WORKER_SQUARE = build_square_network()
    _WORKER_MOORE = build_moore_network()


def execute_task(task):
    network_index, b_index, run_index, b, network_seed, dynamics_seed = task
    started = perf_counter()
    network_type = NETWORK_TYPES[network_index]
    if network_type == 'square':
        offsets, neighbors, graph_stats = _WORKER_SQUARE
    elif network_type == 'moore':
        offsets, neighbors, graph_stats = _WORKER_MOORE
    else:
        offsets, neighbors, graph_stats = build_random_network(network_type, network_seed)
    graph_seconds = perf_counter() - started
    experiment_started = perf_counter()
    result = run_single_experiment(
        FIXED_INITIAL_FC, FIXED_K, b, FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA,
        FIXED_U, EPSILON, INITIAL_REPUTATION, offsets, neighbors, dynamics_seed,
        MCS_TOTAL, MCS_TRANSIENT, EARLY_STOP_ABSORBING, ABSORB_CHECK_INTERVAL,
    )
    experiment_seconds = perf_counter() - experiment_started
    return (network_index, b_index, run_index, network_seed, dynamics_seed,
            result, graph_stats, graph_seconds, experiment_seconds)


def make_tasks(b_values):
    # 只在主进程分配种子；结果不依赖worker数或任务返回顺序。
    root = np.random.SeedSequence(BASE_SEED)
    task_count = len(NETWORK_TYPES) * len(b_values) * NUM_RUNS
    children = iter(root.spawn(task_count * 2))
    used = set()

    def next_seed():
        seed = int(next(children).generate_state(1, dtype=np.uint32)[0])
        while seed in used:
            seed = int(root.spawn(1)[0].generate_state(1, dtype=np.uint32)[0])
        used.add(seed)
        return seed

    tasks = []
    for network_index in range(len(NETWORK_TYPES)):
        for b_index, b in enumerate(b_values):
            for run_index in range(NUM_RUNS):
                network_seed, dynamics_seed = next_seed(), next_seed()
                tasks.append((network_index, b_index, run_index, float(b),
                              network_seed, dynamics_seed))
    return tasks


# ==========================================================
# 5. 绘图：保持原图尺寸、4种颜色/标记、字体、左下角图例，横轴扩展至2
# ==========================================================
def save_plot(b_values, mean_data, output_dir, file_stem, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C']
    markers = ['*', '+', 'o', 'o']
    figure, axis = plt.subplots(figsize=(5, 4.5))

    for index, label in enumerate(NETWORK_LABELS):
        marker_face = 'none' if colors[index] == '#0000CD' else colors[index]
        marker_size = 8 if markers[index] == '*' else 6
        axis.plot(
            b_values, mean_data[index], label=label,
            color=colors[index], marker=markers[index], markersize=marker_size,
            linewidth=1.2, markeredgecolor=colors[index],
            markerfacecolor=marker_face,
        )

    axis.set_ylim(0, 1.0)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axis.set_xlim(1.0, 2.0)
    axis.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    axis.set_xticklabels(['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2'])
    axis.margins(0)
    axis.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
    axis.set_xlabel(r'$b$', fontsize=16)
    axis.set_ylabel(r'$f_c$', fontsize=16)
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


# ==========================================================
# 6. 数据与检查：写盘全部由主进程完成
# ==========================================================
def write_header(handle, metadata, columns):
    for line in metadata.rstrip().splitlines():
        handle.write('# ' + line + '\n')
    handle.write('# ' + columns + '\n')
    handle.flush()


def summary_row(network_index, b, values, graph_means, early_flags):
    count = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if count > 1 else float('nan')
    sem = std / np.sqrt(count)
    return (
        f'{network_index} {NETWORK_TYPES[network_index]} {b:.8f} '
        f'{mean:.10f} {std:.10f} {sem:.10f} {count} '
        f'{np.mean(graph_means):.10f} {int(np.sum(early_flags))}\n'
    )


def validate_parameters(b_values):
    if len(NETWORK_TYPES) != 4 or len(NETWORK_LABELS) != 4:
        raise ValueError('本绘图配置了4条曲线，请保留4种网络。')
    if tuple(NETWORK_TYPES) != ('square', 'rrg', 'ws', 'moore'):
        raise ValueError('NETWORK_TYPES需与NETWORK_LABELS的顺序一致。')
    if not isinstance(L, int) or L < 3 or N != L * L:
        raise ValueError('L必须为>=3的整数，且N=L*L。')
    if TARGET_DEGREE != 4:
        raise ValueError('本比较固定方格度4、随机网络目标平均度4，Moore网格度8。')
    if not (0 <= WS_REWIRE_P <= 1):
        raise ValueError('WS_REWIRE_P必须在[0,1]内。')
    for name, value in (
        ('initial_fc', FIXED_INITIAL_FC), ('alpha', FIXED_ALPHA),
        ('gamma', FIXED_GAMMA), ('u', FIXED_U),
        ('initial_reputation', INITIAL_REPUTATION),
    ):
        if not np.isfinite(value) or not (0 <= value <= 1):
            raise ValueError(name + '必须在[0,1]内。')
    if not np.isfinite(FIXED_BETA) or FIXED_BETA < 0:
        raise ValueError('beta必须为非负有限值。')
    if not np.isfinite(FIXED_K) or FIXED_K <= 0:
        raise ValueError('K必须为正有限值。')
    if not np.isfinite(EPSILON) or EPSILON <= 0:
        raise ValueError('epsilon必须为正有限值。')
    if any(not isinstance(x, int) for x in
           (MCS_TOTAL, MCS_TRANSIENT, NUM_RUNS, ABSORB_CHECK_INTERVAL)):
        raise ValueError('MCS和重复次数必须为整数。')
    if not (0 <= MCS_TRANSIENT < MCS_TOTAL) or NUM_RUNS < 1:
        raise ValueError('要求0<=MCS_TRANSIENT<MCS_TOTAL，NUM_RUNS>=1。')
    if ABSORB_CHECK_INTERVAL < 1 or not (0 < CPU_FRACTION <= 1):
        raise ValueError('检查间隔必须>=1，CPU_FRACTION必须在(0,1]内。')
    if (b_values.ndim != 1 or len(b_values) == 0
            or np.any(~np.isfinite(b_values)) or np.any(b_values < 1.0)
            or np.any(b_values > 2.0) or np.any(np.diff(b_values) <= 0)):
        raise ValueError('b必须在[1.0,2.0]范围内，有限且严格递增。')


def main():
    parser = argparse.ArgumentParser(description='网络结构稳健性：并行独立运行。')
    parser.add_argument('--no-show', action='store_true', help='保存图片但不弹出窗口。')
    parser.add_argument('--workers', type=int, default=None,
                        help='覆盖进程数；默认使用约60%%逻辑核心。')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'network_structure_results',
                        help='输出根目录；每次运行创建独立时间戳子目录。')
    arguments = parser.parse_args()
    b_values = np.asarray(B_VALUES, dtype=np.float64)
    validate_parameters(b_values)

    logical_cores = os.cpu_count() or 1
    workers = max(1, int(logical_cores * CPU_FRACTION))
    if arguments.workers is not None:
        workers = arguments.workers
    if not (1 <= workers <= logical_cores):
        raise ValueError('--workers必须在1和逻辑核心数之间。')
    tasks = make_tasks(b_values)
    total_tasks = len(tasks)
    workers = min(workers, total_tasks)

    # 每次运行使用独立目录，避免覆盖以前长时间计算得到的数据。
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = arguments.output_dir.resolve() / ('run_' + run_stamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    file_stem = (
        f'network_structures_alpha{FIXED_ALPHA:.2f}_beta{FIXED_BETA:.2f}'
        f'_gamma{FIXED_GAMMA:.2f}_u{FIXED_U:.2f}_K{FIXED_K:.2f}_N{N}'
    )
    raw_path = output_dir / (file_stem + '_individual_runs.txt')
    summary_path = output_dir / (file_stem + '_summary.txt')
    plot_data_path = output_dir / (file_stem + '_plot_data.txt')
    report_path = output_dir / (file_stem + '_run_report.txt')

    library_versions = ' '.join(f'{name}={version(name)}' for name in
                                ('numpy', 'numba', 'networkx', 'matplotlib', 'tqdm'))
    metadata = (
        f'alpha={FIXED_ALPHA} beta={FIXED_BETA} gamma={FIXED_GAMMA} '
        f'u_reset={FIXED_U} u_flip={FIXED_U / 2} K={FIXED_K} epsilon={EPSILON}\n'
        f'N={N} initial_fc={FIXED_INITIAL_FC} initial_reputation={INITIAL_REPUTATION}\n'
        f'MCS_TOTAL={MCS_TOTAL} MCS_TRANSIENT={MCS_TRANSIENT} '
        f'averaging_MCS={MCS_TOTAL - MCS_TRANSIENT} NUM_RUNS={NUM_RUNS}\n'
        f'b_values={b_values.tolist()}\n'
        f'network_mapping={dict(enumerate(NETWORK_TYPES))}\n'
        f'square: L={L} periodic=True degree=4\n'
        f'rrg: degree={TARGET_DEGREE}\n'
        f'ws: ring_degree={TARGET_DEGREE} rewiring_p={WS_REWIRE_P}\n'
        'moore: periodic=True diagonal_connections=True degree=8 '
        'mean_degree=8\n'
        'network_sampling=fresh_random_graph_for_each_b_and_run; '
        'topology_fixed_during_each_run; square_and_moore_network_seeds_are_unused\n'
        'all_N_nodes_retained; no_connectivity_conditioning; '
        'isolates_skip_imitation_only; fc_denominator=N\n'
        'payoff=accumulated_not_degree_normalized; local_endorsement=neighbors_plus_self; '
        'update=random_permutation_each_MCS; global_endorsement=start_of_MCS_snapshot; '
        'reputation_update=synchronous_end_of_MCS\n'
        'initialization=fixed_count_round_half_up_with_random_positions\n'
        f'early_stop_absorbing={EARLY_STOP_ABSORBING} '
        f'all_edges_check_interval={ABSORB_CHECK_INTERVAL}; '
        'early_stop_only_if_u=0; remaining_sampling_window_filled_exactly\n'
        'stop_code: 0=ran_to_MCS_TOTAL, 1=global_consensus, 2=all_edges_same_strategy; '
        'absorbed_detected_mcs=0_if_not_detected\n'
        'fc_run=mean_over_sampling_MCS; fc_std=sample_std_ddof1; '
        'fc_sem=fc_std/sqrt(n_runs); degree_std=population_std_ddof0\n'
        'graph_seconds_for_square_excludes_worker_initialization; '
        'experiment_seconds_includes_first_Numba_compilation_if_needed\n'
        f'logical_cores={logical_cores} workers={workers} CPU_FRACTION={CPU_FRACTION} '
        f'actual_worker_core_fraction={workers / logical_cores:.6f} BASE_SEED={BASE_SEED}\n'
        f'expected_individual_rows={total_tasks} '
        f'expected_summary_rows={len(NETWORK_TYPES) * len(b_values)}\n'
        f'python={platform.python_version()} {library_versions}\n'
        'Reproduce network generation with the recorded seeds and package versions.\n'
    )

    print(f'N={N}, alpha={FIXED_ALPHA}, beta={FIXED_BETA}, K={FIXED_K}, '
          f'f_C(0)={FIXED_INITIAL_FC}')
    print('Networks:', ', '.join(NETWORK_LABELS))
    print(f'{len(NETWORK_TYPES)} networks x {len(b_values)} b values x '
          f'{NUM_RUNS} runs = {total_tasks} independent runs')
    print(f'{MCS_TOTAL} MCS/run; average over final {MCS_TOTAL - MCS_TRANSIENT} MCS')
    print(f'Logical cores: {logical_cores}; workers: {workers} '
          f'({100 * workers / logical_cores:.1f}% of logical cores)')
    print('Numba compilation may delay the first result; CPU utilization varies.')
    print('Output directory:', output_dir)

    shape = (len(NETWORK_TYPES), len(b_values), NUM_RUNS)
    run_values = np.full(shape, np.nan)
    mean_degrees = np.full(shape, np.nan)
    early_flags = np.zeros(shape, dtype=np.bool_)
    completed_counts = np.zeros(shape[:2], dtype=np.int64)
    stop_counts = np.zeros(3, dtype=np.int64)
    completed = 0
    full_runs = 0
    status = 'incomplete'
    started = perf_counter()
    raw_columns = (
        'network_index network b run_index network_seed dynamics_seed '
        'fc_initial_actual fc_run fc_final executed_mcs absorbed_detected_mcs '
        'stop_code N edges mean_degree degree_std min_degree max_degree '
        'isolated_nodes components largest_component_size graph_seconds experiment_seconds'
    )
    summary_columns = (
        'network_index network b fc_mean fc_std fc_sem n_runs mean_degree_mean n_early_stop'
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
                    total=total_tasks, desc='Network simulations', unit='run',
                ):
                    (ni, bi, ri, network_seed, dynamics_seed, result,
                     graph_stats, graph_seconds, experiment_seconds) = returned
                    fc, initial_fc, final_fc, executed_mcs, stop_code = result
                    if not np.isfinite(fc) or not (0.0 <= fc <= 1.0):
                        raise RuntimeError('合作率非有限值或超出[0,1]。')
                    if not np.isnan(run_values[ni, bi, ri]):
                        raise RuntimeError('同一个任务返回了重复结果。')
                    run_values[ni, bi, ri] = fc
                    mean_degrees[ni, bi, ri] = graph_stats[2]
                    early_flags[ni, bi, ri] = executed_mcs < MCS_TOTAL
                    completed_counts[ni, bi] += 1
                    completed += 1
                    full_runs += int(executed_mcs == MCS_TOTAL)
                    stop_counts[stop_code] += 1
                    detected_mcs = executed_mcs if stop_code else 0
                    n, edges, mean_k, std_k, min_k, max_k, isolates, components, giant = graph_stats
                    raw_file.write(
                        f'{ni} {NETWORK_TYPES[ni]} {b_values[bi]:.8f} {ri + 1} '
                        f'{network_seed} {dynamics_seed} {initial_fc:.10f} '
                        f'{fc:.10f} {final_fc:.10f} {executed_mcs} {detected_mcs} '
                        f'{stop_code} {n} {edges} {mean_k:.10f} {std_k:.10f} '
                        f'{min_k} {max_k} {isolates} {components} {giant} '
                        f'{graph_seconds:.6f} {experiment_seconds:.6f}\n'
                    )
                    # 每完成一个任务即保存，异常中断仍保留已完成的逐次结果。
                    raw_file.flush()
                    if completed_counts[ni, bi] == NUM_RUNS:
                        summary_file.write(summary_row(
                            ni, b_values[bi], run_values[ni, bi],
                            mean_degrees[ni, bi], early_flags[ni, bi],
                        ))
                        summary_file.flush()

        if completed != total_tasks or np.any(completed_counts != NUM_RUNS):
            raise RuntimeError('实际任务数不足，不能按100次平均输出正式图片。')
        if np.any(~np.isfinite(run_values)):
            raise RuntimeError('存在缺失数据，不能输出正式图片。')

        # 全部完成后排序summary；使用临时文件替换，保留逐次原始文件。
        temporary_summary = summary_path.with_suffix('.tmp')
        with temporary_summary.open('w', encoding='utf-8') as summary_file:
            write_header(summary_file, metadata, summary_columns)
            for ni in range(len(NETWORK_TYPES)):
                for bi, b in enumerate(b_values):
                    summary_file.write(summary_row(
                        ni, b, run_values[ni, bi], mean_degrees[ni, bi], early_flags[ni, bi],
                    ))
        temporary_summary.replace(summary_path)

        mean_data = np.mean(run_values, axis=2)
        np.savetxt(plot_data_path, np.column_stack((b_values, mean_data.T)),
                   fmt='%.10f', encoding='utf-8',
                   header=metadata + 'b ' + ' '.join(NETWORK_TYPES))
        status = 'simulation_and_text_complete'
    finally:
        # 即使中途异常/中断，也记录已完成任务数和已落盘文件的位置。
        elapsed = perf_counter() - started
        with report_path.open('w', encoding='utf-8') as report:
            write_header(report, metadata, 'run_report')
            report.write(
                f'status={status}\nexpected_runs={total_tasks}\ncompleted_runs={completed}\n'
                f'complete_parameter_points={int(np.sum(completed_counts == NUM_RUNS))}\n'
                f'full_length_runs={full_runs}\nearly_stopped_runs={completed - full_runs}\n'
                f'stop_code_counts={stop_counts.tolist()}\nelapsed_seconds={elapsed:.6f}\n'
                f'raw_data={raw_path.name}\nsummary_data={summary_path.name}\n'
            )

    print(f'Completed {completed}/{total_tasks}; full runs={full_runs}; '
          f'early stops={completed - full_runs}; elapsed={elapsed:.1f}s')
    print('Individual TXT:', raw_path)
    print('Summary TXT:', summary_path)
    print('Plot data TXT:', plot_data_path)
    print('Run report TXT:', report_path)
    save_plot(b_values, mean_data, output_dir, file_stem, show=not arguments.no_show)
    with report_path.open('a', encoding='utf-8') as report:
        report.write('figure_export=complete_png_and_eps\n')


if __name__ == '__main__':
    mp.freeze_support()
    main()


