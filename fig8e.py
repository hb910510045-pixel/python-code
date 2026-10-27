import os

# 在导入数值库前设置；Windows spawn子进程也会执行，防止线程过度叠加。
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
# 1. 参数
# ==========================================================
GAME_TYPES = ('weak_pd', 'pd', 'snowdrift', 'stag_hunt')
GAME_LABELS = ('Weak PD', 'PD', 'Snowdrift', 'Stag hunt')
SUCKER_PAYOFFS = (0.0, -0.50, 0.50, -0.50)
REWARD_R = 1.0
PUNISHMENT_P = 0.0
B_VALUES = np.round(np.linspace(0.5, 2.6, 41), 10)

L = 100
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_GAMMA = 0.10
FIXED_U = 0.00
FIXED_K = 0.10
EPSILON = 0.01
FIXED_INITIAL_FC = 0.50
INITIAL_REPUTATION = 0.50

MCS_TOTAL = 20000
MCS_TRANSIENT = 17000
NUM_RUNS = 100
CPU_FRACTION = 0.60
BASE_SEED = 20261007
PNG_DPI = 600
EARLY_STOP_ABSORBING = True


def valid_game_point(game, reward, sucker, temptation, punishment):
    """按真实收益关系筛选，不把同一标签扩展到其他博弈类别。"""
    if game == 'weak_pd':
        # 保留b=1边界点，另存边界标记。
        return temptation >= reward > punishment and sucker == punishment
    if game == 'pd':
        return temptation > reward > punishment > sucker
    if game == 'snowdrift':
        return temptation > reward > sucker > punishment
    if game == 'stag_hunt':
        return reward > temptation > punishment > sucker
    raise ValueError('未知博弈类型：' + game)


def validate_and_select_points(b_values):
    if GAME_TYPES != ('weak_pd', 'pd', 'snowdrift', 'stag_hunt'):
        raise ValueError('GAME_TYPES的顺序应与图例和SUCKER_PAYOFFS一致。')
    if len(GAME_LABELS) != 4 or len(SUCKER_PAYOFFS) != 4:
        raise ValueError('本图配置4种博弈。')
    if not isinstance(L, int) or L < 3:
        raise ValueError('L必须是>=3的整数。')
    if (any(not isinstance(x, int) for x in (MCS_TOTAL, MCS_TRANSIENT, NUM_RUNS))
            or not 0 <= MCS_TRANSIENT < MCS_TOTAL or NUM_RUNS < 1):
        raise ValueError('MCS和运行次数须为整数，且0<=暂态MCS<总MCS、NUM_RUNS>=1。')
    for name, value in (
            ('alpha', FIXED_ALPHA), ('beta', FIXED_BETA), ('gamma', FIXED_GAMMA),
            ('u', FIXED_U), ('initial_fc', FIXED_INITIAL_FC),
            ('initial_reputation', INITIAL_REPUTATION),
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + '必须在[0,1]内。')
    if not np.isfinite(FIXED_K) or FIXED_K <= 0:
        raise ValueError('K必须是正有限值。')
    if not np.isfinite(EPSILON) or EPSILON <= 0 or not 0 < CPU_FRACTION <= 1:
        raise ValueError('epsilon必须>0，CPU_FRACTION必须在(0,1]内。')
    if not np.all(np.isfinite((REWARD_R, PUNISHMENT_P, *SUCKER_PAYOFFS))):
        raise ValueError('收益参数必须是有限值。')
    if REWARD_R != 1.0 or PUNISHMENT_P != 0.0:
        raise ValueError('本比较固定R=1、P=0。')
    if (b_values.ndim != 1 or len(b_values) == 0
            or np.any(~np.isfinite(b_values)) or np.any(np.diff(b_values) <= 0)
            or np.any(b_values < 0.5) or np.any(b_values > 2.6)):
        raise ValueError('B_VALUES应在[0.5,2.6]内且有限、严格递增。')

    valid = np.zeros((len(GAME_TYPES), len(b_values)), dtype=np.bool_)
    for gi, game in enumerate(GAME_TYPES):
        for bi, temptation in enumerate(b_values):
            valid[gi, bi] = valid_game_point(
                game, REWARD_R, SUCKER_PAYOFFS[gi], temptation, PUNISHMENT_P,
            )
        # 不属于该博弈定义域的b点保留为NaN，不强行当作模拟数据。
    return valid


# ==========================================================
# 2. 方格与模型动力学
# ==========================================================
def build_neighbor_map():
    nodes = np.arange(L * L, dtype=np.int64)
    row, col = nodes // L, nodes % L
    neighbors = np.empty((L * L, 4), dtype=np.int64)
    neighbors[:, 0] = ((row - 1 + L) % L) * L + col
    neighbors[:, 1] = ((row + 1) % L) * L + col
    neighbors[:, 2] = row * L + ((col - 1 + L) % L)
    neighbors[:, 3] = row * L + ((col + 1) % L)
    return neighbors


@njit(cache=True, inline='always')
def calculate_payoff(strategy_i, strategy_j, reward, sucker, temptation, punishment):
    # 策略编码C=1、D=0；避免直接用该编码索引按(C,D)排列的矩阵而颠倒策略。
    if strategy_i == 1:
        return reward if strategy_j == 1 else sucker
    return temptation if strategy_j == 1 else punishment


@njit(cache=True, inline='always')
def calculate_fitness(node, reward, sucker, temptation, punishment, alpha, beta,
                      epsilon, global_C, strategies, reputations, neighbors):
    strategy = strategies[node]
    payoff = 0.0
    numerator = reputations[node] + epsilon
    denominator = reputations[node] + epsilon
    for k in range(4):
        neighbor = neighbors[node, k]
        payoff += calculate_payoff(strategy, strategies[neighbor],
                                   reward, sucker, temptation, punishment)
        weight = reputations[neighbor] + epsilon
        denominator += weight
        if strategies[neighbor] == strategy:
            numerator += weight
    local = numerator / denominator
    global_value = global_C if strategy == 1 else 1.0 - global_C

    endorsement = alpha * local + (1.0 - alpha) * global_value

    if endorsement < 1e-9:
        endorsement = 1e-9
    # payoff可以为负；幂只作用于正的endorsement，原公式直接延拓。
    return payoff * endorsement ** beta


@njit(cache=True)
def simulation_step(reward, sucker, temptation, punishment, noise_K, alpha, beta,
                    gamma, epsilon, u, strategies, reputations, neighbors):
    population = len(strategies)
    total_reputation = np.sum(reputations) + population * epsilon
    cooperator_reputation = 0.0
    for i in range(population):
        if strategies[i] == 1:
            cooperator_reputation += reputations[i] + epsilon
    global_C = cooperator_reputation / total_reputation

    # 延续所给Python的每MCS随机排列更新规则。
    for i in np.random.permutation(population):
        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)
            continue
        j = neighbors[i, np.random.randint(0, 4)]
        if strategies[i] == strategies[j]:
            continue
        fitness_i = calculate_fitness(
            i, reward, sucker, temptation, punishment, alpha, beta, epsilon,
            global_C, strategies, reputations, neighbors,
        )
        fitness_j = calculate_fitness(
            j, reward, sucker, temptation, punishment, alpha, beta, epsilon,
            global_C, strategies, reputations, neighbors,
        )
        exponent = (fitness_i - fitness_j) / noise_K
        # 与所给代码一致的Fermi指数截断。
        if exponent > 20.0:
            probability = 0.0
        elif exponent < -20.0:
            probability = 1.0
        else:
            probability = 1.0 / (1.0 + np.exp(exponent))
        if np.random.random() < probability:
            strategies[i] = strategies[j]

    for i in range(population):
        reputations[i] = (1.0 - gamma) * reputations[i] + gamma * float(strategies[i])


@njit(cache=True)
def run_single_experiment(reward, sucker, temptation, punishment, noise_K, alpha,
                          beta, gamma, epsilon, u, initial_fc, initial_reputation,
                          neighbors, seed, mcs_total, mcs_transient, early_stop):
    np.random.seed(seed)
    population = neighbors.shape[0]
    initial_count = int(np.floor(initial_fc * population + 0.5))
    strategies = np.zeros(population, dtype=np.int64)
    initial_positions = np.random.permutation(population)
    for i in range(initial_count):
        strategies[initial_positions[i]] = 1
    reputations = np.full(population, initial_reputation, dtype=np.float64)
    actual_initial_fc = initial_count / population
    cooperation_sum = 0.0
    sample_count = mcs_total - mcs_transient
    cooperation = actual_initial_fc

    for mcs in range(mcs_total):
        simulation_step(reward, sucker, temptation, punishment, noise_K, alpha,
                        beta, gamma, epsilon, u, strategies, reputations, neighbors)
        cooperative_count = np.sum(strategies)
        cooperation = cooperative_count / population
        if mcs >= mcs_transient:
            cooperation_sum += cooperation

        if (early_stop and u == 0.0
                and (cooperative_count == 0 or cooperative_count == population)):
            remaining_samples = mcs_total - max(mcs + 1, mcs_transient)
            cooperation_sum += remaining_samples * cooperation
            # 平均合作率、初始合作率、最终合作率、已执行MCS、吸收态检测MCS。
            return (cooperation_sum / sample_count, actual_initial_fc,
                    cooperation, mcs + 1, mcs + 1)
    return (cooperation_sum / sample_count, actual_initial_fc,
            cooperation, mcs_total, 0)


# ==========================================================
# 3. 多进程与种子：只并行独立运行
# ==========================================================
_WORKER_NEIGHBORS = None


def initialize_worker():
    global _WORKER_NEIGHBORS
    _WORKER_NEIGHBORS = build_neighbor_map()


def execute_task(task):
    gi, bi, ri, reward, sucker, temptation, punishment, seed = task
    started = perf_counter()
    result = run_single_experiment(
        reward, sucker, temptation, punishment, FIXED_K, FIXED_ALPHA, FIXED_BETA,
        FIXED_GAMMA, EPSILON, FIXED_U, FIXED_INITIAL_FC, INITIAL_REPUTATION,
        _WORKER_NEIGHBORS, seed, MCS_TOTAL, MCS_TRANSIENT, EARLY_STOP_ABSORBING,
    )
    return gi, bi, ri, seed, result, perf_counter() - started


def make_tasks(b_values, valid):
    root = np.random.SeedSequence(BASE_SEED)
    children = iter(root.spawn(int(np.count_nonzero(valid)) * NUM_RUNS))
    used_seeds = set()
    tasks = []
    for gi in range(len(GAME_TYPES)):
        for bi, temptation in enumerate(b_values):
            if not valid[gi, bi]:
                continue
            for ri in range(NUM_RUNS):
                seed = int(next(children).generate_state(1, dtype=np.uint32)[0])
                while seed in used_seeds:
                    seed = int(root.spawn(1)[0].generate_state(1, dtype=np.uint32)[0])
                used_seeds.add(seed)
                tasks.append((gi, bi, ri, REWARD_R, SUCKER_PAYOFFS[gi],
                              float(temptation), PUNISHMENT_P, seed))
    return tasks


# ==========================================================
# 4. 绘图：沿用原样式，横轴为b=0.5--2.6
# ==========================================================
def save_plot(b_values, means, output_dir, file_stem, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C']
    markers = ['*', '+', 'o', 'o']
    figure, axis = plt.subplots(figsize=(5, 4.5))
    for gi, label in enumerate(GAME_LABELS):
        # nan表示该b不属于这一博弈类别，不补零、不外推、不跨越连接。
        axis.plot(
            b_values, means[gi], label=label, color=colors[gi],
            marker=markers[gi], markersize=8 if markers[gi] == '*' else 6,
            linewidth=1.2, markeredgecolor=colors[gi],
            markerfacecolor='none' if colors[gi] == '#0000CD' else colors[gi],
        )
    axis.set_ylim(0, 1.0)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axis.set_xlim(0.45, 2.6)
    axis.set_xticks([0.5, 0.9, 1.3, 1.7, 2.1, 2.6])
    axis.set_xticklabels(['0.5', '0.9', '1.3', '1.7', '2.1', '2.6'])
    axis.margins(0)
    axis.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
    axis.set_xlabel(r'$b$', fontsize=16)
    axis.set_ylabel(r'$f_c$', fontsize=16)

    # 图例缩小后放在坐标轴内部中部偏右，减少对曲线的遮挡
    axis.legend(fontsize=10, frameon=True, framealpha=0.78,
                edgecolor='black', loc='center', bbox_to_anchor=(0.35, 0.55),
                borderaxespad=0.0, handlelength=1.2,
                borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

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


def point_prefix(gi, bi, temptation):
    return (f'{gi} {GAME_TYPES[gi]} {bi} {REWARD_R:.8f} '
            f'{SUCKER_PAYOFFS[gi]:.8f} {temptation:.8f} {PUNISHMENT_P:.8f} '
            f'{int(temptation == REWARD_R)}')


def summary_row(gi, bi, temptation, values, early):
    count = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if count > 1 else float('nan')
    sem = std / np.sqrt(count)
    return (point_prefix(gi, bi, temptation)
            + f' {mean:.10f} {std:.10f} {sem:.10f} {count} {int(np.sum(early))}\n')


# ==========================================================
# 5. 主程序：TXT逐次保存，完整结束后输出正式图片
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description='Compare evolutionary game classes.')
    parser.add_argument('--no-show', action='store_true', help='保存PNG/EPS，不弹出窗口。')
    parser.add_argument('--workers', type=int, default=None,
                        help='覆盖worker数；默认约60%%逻辑核心。')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'game_class_results')
    arguments = parser.parse_args()
    b_values = np.asarray(B_VALUES, dtype=np.float64)
    valid = validate_and_select_points(b_values)
    tasks = make_tasks(b_values, valid)
    total_tasks = len(tasks)
    total_points = int(np.count_nonzero(valid))
    logical_cores = os.cpu_count() or 1
    workers = max(1, int(logical_cores * CPU_FRACTION))
    if arguments.workers is not None:
        workers = arguments.workers
    if not 1 <= workers <= logical_cores:
        raise ValueError('--workers必须在1与逻辑核心数之间。')
    workers = min(workers, total_tasks)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = arguments.output_dir.resolve() / ('run_' + timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    file_stem = (f'game_classes_alpha{FIXED_ALPHA:.2f}_beta{FIXED_BETA:.2f}'
                 f'_gamma{FIXED_GAMMA:.2f}_u{FIXED_U:.2f}_K{FIXED_K:.2f}_L{L}')
    plot_path = output_dir / (file_stem + '_plot_data.txt')
    versions = ' '.join(f'{name}={version(name)}'
                        for name in ('numpy', 'numba', 'matplotlib', 'tqdm'))
    metadata = (
        f'alpha={FIXED_ALPHA} beta={FIXED_BETA} gamma={FIXED_GAMMA} '
        f'u_reset={FIXED_U} u_flip={FIXED_U / 2} K={FIXED_K} epsilon={EPSILON}\n'
        f'L={L} N={L * L} network=periodic_square degree=4 '
        f'initial_fc={FIXED_INITIAL_FC} initial_reputation={INITIAL_REPUTATION}\n'
        f'MCS_TOTAL={MCS_TOTAL} MCS_TRANSIENT={MCS_TRANSIENT} '
        f'averaging_MCS={MCS_TOTAL - MCS_TRANSIENT} NUM_RUNS={NUM_RUNS}\n'
        f'game_mapping={dict(enumerate(GAME_TYPES))}; S_values={SUCKER_PAYOFFS}\n'
        f'R={REWARD_R} P={PUNISHMENT_P}; matrix_row_column_order=C,D; matrix=[[R,S],[b,P]]\n'
        f'b_values={b_values.tolist()}; valid_points_per_game={valid.sum(axis=1).tolist()}\n'
        'weak_pd:b>=R>P=S (b=R is boundary); pd:b>R>P>S; snowdrift:b>R>S>P; '
        'stag_hunt:R>b>P>S\n'
        'b_equals_R_boundary=1_only_for_the_weak_PD_reference_point; '
        'invalid_class_points_are_not_simulated\n'
        'fitness=accumulated_payoff*endorsement**beta; '
        'negative_payoffs_and_fitness_retained; payoff_shift=0; payoff_clipping=False\n'
        'update=random_permutation_per_MCS; global_endorsement=start_of_MCS_snapshot; '
        'local_endorsement=neighbors_plus_self; reputation_update=synchronous_end_of_MCS\n'
        'initialization=fixed_cooperator_count_round_half_up_random_positions; '
        'mutation=random_strategy_reset_without_reputation_reset\n'
        f'early_stop_absorbing={EARLY_STOP_ABSORBING}; '
        'only_u=0_global_consensus; remaining_sampling_window_completed_exactly\n'
        'fc_run=mean_over_sampling_MCS; fc_mean=mean_of_run_means; '
        'fc_std=sample_std_ddof1; fc_sem=fc_std/sqrt(n_runs)\n'
        'absorbed_detected_mcs=0_if_not_detected; elapsed_run_seconds_includes_first_JIT_if_needed\n'
        f'BASE_SEED={BASE_SEED} logical_cores={logical_cores} workers={workers} '
        f'CPU_FRACTION={CPU_FRACTION} worker_core_fraction={workers / logical_cores:.6f}\n'
        f'expected_plot_rows={len(b_values)} expected_game_columns={len(GAME_TYPES)}\n'
        'plot_x=b; xlim=[0.5,2.6]; plot_nan=outside_game_class_domain; '
        'weak_pd_b_equals_original_b\n'
        f'python={platform.python_version()} {versions}\n'
    )
    print(f'alpha={FIXED_ALPHA}, beta={FIXED_BETA}, N={L * L}, K={FIXED_K}')
    print('Games:', ', '.join(GAME_LABELS))
    print('Valid points per game:', valid.sum(axis=1).tolist())
    print(f'{total_points} parameter points x {NUM_RUNS} runs = {total_tasks} runs')
    print(f'{MCS_TOTAL} MCS/run; averaging final {MCS_TOTAL - MCS_TRANSIENT} MCS')
    print(f'Logical cores={logical_cores}; workers={workers} '
          f'({100 * workers / logical_cores:.1f}% of cores)')
    print('First result may wait for Numba compilation. CPU utilization varies.')
    print('Output directory:', output_dir)

    values = np.full((*valid.shape, NUM_RUNS), np.nan)
    early_flags = np.zeros(values.shape, dtype=np.bool_)
    counts = np.zeros(valid.shape, dtype=np.int64)
    completed = 0
    full_runs = 0
    absorbed_runs = 0
    status = 'incomplete'
    started = perf_counter()
    context = mp.get_context('spawn')
    with context.Pool(processes=workers, initializer=initialize_worker) as pool:
        for result in tqdm(
                pool.imap_unordered(execute_task, tasks, chunksize=1),
                total=total_tasks, desc='Game-class simulations', unit='run',
        ):
            gi, bi, ri, seed, observation, run_seconds = result
            fc, initial_fc, final_fc, executed_mcs, absorbed_mcs = observation
            if not np.isfinite(fc) or not 0 <= fc <= 1:
                raise RuntimeError('合作率应为[0,1]内的有限值。')
            values[gi, bi, ri] = fc
            counts[gi, bi] += 1

    if total_tasks > 0:
        if np.any(counts[valid] != NUM_RUNS):
            raise RuntimeError('任务数不足，无法输出完整绘图数据。')
        if np.any(~np.isfinite(values[valid])):
            raise RuntimeError('有效博弈参数点存在缺失结果。')

    means = np.full(valid.shape, np.nan)
    means[valid] = np.mean(values[valid], axis=1)
    np.savetxt(plot_path, np.column_stack((b_values, means.T)), fmt='%.10f',
               encoding='utf-8', header=metadata + 'b ' + ' '.join(GAME_TYPES))
    print('Plot data TXT:', plot_path)
    save_plot(b_values, means, output_dir, file_stem, show=not arguments.no_show)


if __name__ == '__main__':
    mp.freeze_support()
    main()



