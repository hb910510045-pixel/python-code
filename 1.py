# -*- coding: utf-8 -*-
"""复现 Liao et al., Applied Mathematics and Computation 439 (2023) 127635 的 Fig. 2(b)。

论文：Cooperative evolution under the joint influence of local popularity
      and global popularity
DOI：https://doi.org/10.1016/j.amc.2022.127635
依据：论文第2--3页 Model、式(2)--(6)，第4页 Fig. 2(b) 及图注。

原文明确给出的模型和参数：
  100×100 周期方格，von Neumann 四邻居；初始 C、D 各占一半，位置随机。
  弱囚徒困境：行、列策略顺序均为(C,D)，收益矩阵 [[1,0],[b,0]]。
  局部流行度 = (同策略邻居数+1)/5；包含自身。
  全局流行度 = 全系统同策略人数/N；已包含自身，不能再额外加1。
  F_x = P_x * [alpha*local_x + (1-alpha)*global_x]**beta。
  beta=0.50；alpha=0.60、0.65、0.80、1.00；K=0.10。
  每MCS执行N次有放回随机选点，每次选一个随机邻居，按Fermi规则立即更新。
  全局f_C、f_D只在MCS边界更新，在一个MCS内部固定；局部状态即时更新。
  每次20000 MCS，取最后3000 MCS的时间平均，再对100次独立初态求平均。

本复现中需要明确的设置：
  b=1.01,1.02,...,1.60，根据图中标记位置选取；原文未附完整取点表。
  BASE_SEED是本脚本的复现种子；原文没有提供作者的随机种子或源代码。
  不含声誉、EMA、epsilon或突变；不能直接沿用此前声誉模型的更新函数。
  邻居合作人数缓存和概率查表只用于加速，按原文公式及时更新。
  只有全C/全D吸收态可提前结束，剩余采样窗口按不变的合作比例精确补齐。
  不根据暂时平稳提前终止，不拟合、平滑、插值或手动设置任何合作率。

安装依赖：
    python -m pip install numpy numba matplotlib tqdm
运行（在PyCharm中也可直接运行整个文件）：
    python reproduce_liao2023_fig2b.py
仅保存文件，不弹出绘图窗口：
    python reproduce_liao2023_fig2b.py --no-show

默认使用floor(逻辑CPU核心数*0.60)个单线程进程，至少1个。
这控制并行进程数，不保证操作系统显示的总CPU使用率精确等于60%。

结果保存在本脚本旁 fig2b_results/run_时间戳/：
    *_plot_data.txt  第1列b，后4列为对应alpha的平均合作率。
    *_summary.txt    各参数点的均值、独立运行间标准差、标准误和完成次数。
    *.png           600 dpi。
    *.eps           矢量图。
TXT在每个参数点全部完成后更新；中途停止时nan表示尚未完成，并非0。
只保存绘图所需的统计数据，不保存微观策略快照或完整时间序列。

这是按论文公开描述编写的独立实现；未使用作者未公开的源代码或种子，
不能保证随机模拟的结果与原图逐点完全相同。
"""

import os

# 必须在导入NumPy/Numba之前设置，Windows子进程也会执行。
for _thread_variable in (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
):
    os.environ[_thread_variable] = '1'

import argparse
import multiprocessing as mp
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numba import njit
from tqdm import tqdm


# ==========================================================
# 1. Fig. 2(b)参数；与此前声誉模型分别设置
# ==========================================================
L = 100
ALPHA_VALUES = (0.60, 0.65, 0.80, 1.00)
BETA = 0.50
K = 0.10
B_VALUES = np.arange(101, 161, dtype=np.int64) / 100.0
MCS_TOTAL = 20000
MCS_TRANSIENT = 17000
NUM_RUNS = 100

CPU_FRACTION = 0.60
BASE_SEED = 20261007
EARLY_STOP_ABSORBING = True
PNG_DPI = 600


def build_neighbors(lattice_size):
    """上、下、左、右四邻居；周期边界，无自博弈。"""
    nodes = np.arange(lattice_size * lattice_size, dtype=np.int64)
    row, col = nodes // lattice_size, nodes % lattice_size
    neighbors = np.empty((len(nodes), 4), dtype=np.int64)
    neighbors[:, 0] = ((row - 1) % lattice_size) * lattice_size + col
    neighbors[:, 1] = ((row + 1) % lattice_size) * lattice_size + col
    neighbors[:, 2] = row * lattice_size + (col - 1) % lattice_size
    neighbors[:, 3] = row * lattice_size + (col + 1) % lattice_size
    return neighbors


# ==========================================================
# 2. 论文式(5)、式(6)；C=1，D=0
# ==========================================================
@njit(cache=True, inline='always')
def fitness_from_counts(strategy, cooperative_neighbors, global_fc,
                        b, alpha, beta):
    """收益仅来自四邻居；局部流行度另外包含自身。"""
    if strategy == 1:
        payoff = float(cooperative_neighbors)
        local = (cooperative_neighbors + 1.0) / 5.0
        global_popularity = global_fc
    else:
        payoff = b * cooperative_neighbors
        # D的同策略邻居数为4-cooperative_neighbors。
        local = (5.0 - cooperative_neighbors) / 5.0
        global_popularity = 1.0 - global_fc

    popularity = alpha * local + (1.0 - alpha) * global_popularity
    return payoff * popularity ** beta


@njit(cache=True, inline='always')
def fermi_probability(fitness_x, fitness_y, noise):
    """式(6)的稳定数值写法；不使用此前代码的±20硬截断。"""
    exponent = (fitness_x - fitness_y) / noise
    if exponent >= 0.0:
        small = np.exp(-exponent)
        return small / (1.0 + small)
    return 1.0 / (1.0 + np.exp(exponent))


@njit(cache=True)
def fill_probability_table(table, fitness_table, global_fc, b, alpha, beta, noise):
    """每MCS重建查表，避免在N次微更新中重复求幂和指数。

    一个MCS中global_fc固定，适应度只取决于自身策略和当前C邻居数。
    table[strategy_x, nC_x, nC_y]为x模仿异策略邻居y的概率。
    """
    for strategy in range(2):
        for n_c in range(5):
            fitness_table[strategy, n_c] = fitness_from_counts(
                strategy, n_c, global_fc, b, alpha, beta,
            )
    for strategy_x in range(2):
        strategy_y = 1 - strategy_x
        for n_c_x in range(5):
            for n_c_y in range(5):
                table[strategy_x, n_c_x, n_c_y] = fermi_probability(
                    fitness_table[strategy_x, n_c_x],
                    fitness_table[strategy_y, n_c_y], noise,
                )


@njit(cache=True)
def run_single_experiment(neighbors, alpha, beta, b, noise, seed,
                          mcs_total, mcs_transient, early_stop):
    np.random.seed(seed)
    population = neighbors.shape[0]

    # 原文要求人数相等：恰好N/2个C，随机打乱位置。
    strategies = np.zeros(population, dtype=np.int64)
    strategies[:population // 2] = 1
    np.random.shuffle(strategies)
    cooperative_count = population // 2

    # 缓存每个节点的C邻居数。策略改变时，立即更新其四个邻居的缓存。
    # x自身的策略改变不会改变x自己的C邻居数。
    cooperative_neighbors = np.zeros(population, dtype=np.int64)
    for x in range(population):
        for direction in range(4):
            cooperative_neighbors[x] += strategies[neighbors[x, direction]]

    probabilities = np.empty((2, 5, 5), dtype=np.float64)
    fitness_table = np.empty((2, 5), dtype=np.float64)
    sample_sum = 0.0
    sample_count = mcs_total - mcs_transient

    for mcs in range(mcs_total):
        # 此处只在MCS开始取一次全局快照，符合原文的全局信息滞后规则。
        global_fc = cooperative_count / population
        fill_probability_table(
            probabilities, fitness_table, global_fc, b, alpha, beta, noise,
        )

        # 关键：有放回抽样N次，因此每个玩家平均获得一次更新机会。
        # 不使用np.random.permutation(population)安排更新。
        for _ in range(population):
            x = np.random.randint(0, population)
            y = neighbors[x, np.random.randint(0, 4)]
            strategy_x = strategies[x]
            strategy_y = strategies[y]
            if strategy_x == strategy_y:
                # 同策略模仿不会改变状态，仍计为一次微更新。
                continue

            probability = probabilities[
                strategy_x, cooperative_neighbors[x], cooperative_neighbors[y]
            ]
            if np.random.random() < probability:
                delta = strategy_y - strategy_x
                strategies[x] = strategy_y
                cooperative_count += delta
                for direction in range(4):
                    cooperative_neighbors[neighbors[x, direction]] += delta

        current_fc = cooperative_count / population
        if mcs >= mcs_transient:
            # mcs为0基索引；采样已完成的第17001--20000个MCS。
            sample_sum += current_fc

        if (early_stop and
                (cooperative_count == 0 or cooperative_count == population)):
            # 无突变，且只向邻居模仿：全C/全D以后永远不变。
            # 当前MCS若属于采样窗口已累加，只补齐尚未执行的采样点。
            remaining_samples = mcs_total - max(mcs + 1, mcs_transient)
            sample_sum += remaining_samples * current_fc
            return sample_sum / sample_count

    return sample_sum / sample_count


# ==========================================================
# 3. 多进程：仅并行独立模拟，不并行更新同一张晶格
# ==========================================================
_WORKER_NEIGHBORS = None
_WORKER_SETTINGS = None


def initialize_worker(lattice_size, beta, noise, total, transient, early_stop):
    global _WORKER_NEIGHBORS, _WORKER_SETTINGS
    _WORKER_NEIGHBORS = build_neighbors(lattice_size)
    _WORKER_SETTINGS = (beta, noise, total, transient, early_stop)


def execute_task(task):
    alpha_index, b_index, run_index, alpha, b, seed = task
    beta, noise, total, transient, early_stop = _WORKER_SETTINGS
    fc = run_single_experiment(
        _WORKER_NEIGHBORS, alpha, beta, b, noise, seed,
        total, transient, early_stop,
    )
    return alpha_index, b_index, run_index, fc


def make_tasks(alphas, b_values):
    """种子由主进程统一分配，结果不依赖worker数或任务完成顺序。"""
    root = np.random.SeedSequence(BASE_SEED)
    task_count = len(alphas) * len(b_values) * NUM_RUNS
    children = iter(root.spawn(task_count))
    used = set()
    tasks = []
    # b优先，使每个横坐标对应的四条曲线相继完成并保存。
    for bi, b in enumerate(b_values):
        for ai, alpha in enumerate(alphas):
            for ri in range(NUM_RUNS):
                seed = int(next(children).generate_state(1, dtype=np.uint32)[0])
                while seed in used:
                    seed = int(root.spawn(1)[0].generate_state(1, dtype=np.uint32)[0])
                used.add(seed)
                tasks.append((ai, bi, ri, float(alpha), float(b), seed))
    return tasks


# ==========================================================
# 4. 数据和原论文Fig. 2(b)的绘图样式
# ==========================================================
def atomic_savetxt(path, data, header):
    temporary_path = path.with_suffix('.tmp')
    np.savetxt(temporary_path, data, fmt='%.10g', encoding='utf-8', header=header)
    temporary_path.replace(path)


def save_text(output_dir, stem, metadata, alphas, b_values, values, counts):
    means = np.full(counts.shape, np.nan, dtype=np.float64)
    rows = []
    for ai, alpha in enumerate(alphas):
        for bi, b in enumerate(b_values):
            count = int(counts[ai, bi])
            mean = std = sem = float('nan')
            # 只有满NUM_RUNS的参数点才进入正式绘图数据。
            if count == NUM_RUNS:
                observations = values[ai, bi]
                mean = float(np.mean(observations))
                std = (float(np.std(observations, ddof=1))
                       if count > 1 else float('nan'))
                sem = std / np.sqrt(count)
                means[ai, bi] = mean
            rows.append((alpha, b, mean, std, sem, count))

    complete = bool(np.all(counts == NUM_RUNS))
    header = metadata + (
        f'complete={complete} completed_runs={int(np.sum(counts))}\n'
        'nan=parameter_point_not_yet_complete; no_curve_smoothing_or_interpolation\n'
    )
    atomic_savetxt(
        output_dir / (stem + '_plot_data.txt'),
        np.column_stack((b_values, means.T)),
        header + 'b ' + ' '.join(f'fc_alpha_{alpha:.2f}' for alpha in alphas),
    )
    atomic_savetxt(
        output_dir / (stem + '_summary.txt'), np.asarray(rows),
        header + 'alpha b fc_mean fc_std fc_sem n_completed_runs',
    )
    return means


def save_figure(output_dir, stem, alphas, b_values, means, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    colors = ['#FFE000', '#54C5CE', '#0000FF', '#FF0000']
    markers = ['d', '+', 'o', 'o']
    faces = ['none', 'none', 'none', '#FF0000']
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for ai, alpha in enumerate(alphas):
        ax.plot(
            b_values, means[ai], label=rf'$\alpha={alpha:.2f}$',
            color=colors[ai], marker=markers[ai], markerfacecolor=faces[ai],
            markeredgecolor=colors[ai], markeredgewidth=1.0,
            markersize=5, linewidth=1.2, clip_on=False,
        )

    # 原图横轴显示1.0--1.6，右端略留空隙以完整显示b=1.60的标记。
    ax.set_xlim(1.0, 1.62)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    ax.set_xticklabels(['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6'])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
    ax.set_xlabel(r'$b$', fontsize=14)
    ax.set_ylabel(r'$f_c$', fontsize=14)
    ax.text(0.015, 0.98, '(b)', transform=ax.transAxes,
            ha='left', va='top', fontsize=14, fontweight='bold')
    ax.text(0.97, 0.15, r'$\beta=0.50$', transform=ax.transAxes,
            ha='right', va='center', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=1.0,
              edgecolor='black', fancybox=False, handlelength=2.0)
    fig.tight_layout()
    png_path = output_dir / (stem + '.png')
    eps_path = output_dir / (stem + '.eps')
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches='tight')
    fig.savefig(eps_path, format='eps', dpi=PNG_DPI, bbox_inches='tight')
    print('PNG:', png_path)
    print('EPS:', eps_path)
    if show:
        plt.show()
    plt.close(fig)


def validate_parameters(alphas, b_values):
    if not isinstance(L, int) or L < 4 or (L * L) % 2:
        raise ValueError('L必须是>=4的整数，且N=L*L须为偶数以使初始C、D人数相等。')
    if tuple(alphas) != (0.60, 0.65, 0.80, 1.00) or BETA != 0.50:
        raise ValueError('复现Fig. 2(b)须使用alpha=(0.60,0.65,0.80,1.00)、beta=0.50。')
    if K != 0.10:
        raise ValueError('论文固定K=0.10。')
    if (b_values.ndim != 1 or len(b_values) == 0
            or np.any(~np.isfinite(b_values)) or np.any(np.diff(b_values) <= 0)
            or np.any(b_values <= 1.0) or np.any(b_values >= 2.0)):
        raise ValueError('弱PD的b须有限、严格递增且在(1,2)内。')
    if (any(not isinstance(v, int) for v in (MCS_TOTAL, MCS_TRANSIENT, NUM_RUNS))
            or not 0 <= MCS_TRANSIENT < MCS_TOTAL or NUM_RUNS < 1):
        raise ValueError('MCS和重复次数须为整数，且0<=暂态<总MCS、NUM_RUNS>=1。')
    if not np.isfinite(CPU_FRACTION) or not 0 < CPU_FRACTION <= 1:
        raise ValueError('CPU_FRACTION须在(0,1]内。')


def main():
    parser = argparse.ArgumentParser(description='Reproduce Liao et al. (2023), Fig. 2(b).')
    parser.add_argument('--no-show', action='store_true', help='保存PNG/EPS，不弹出窗口。')
    parser.add_argument('--workers', type=int, default=None,
                        help='自选进程数；默认floor(逻辑核心数*0.60)。')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'fig2b_results')
    args = parser.parse_args()
    alphas = np.asarray(ALPHA_VALUES, dtype=np.float64)
    b_values = np.asarray(B_VALUES, dtype=np.float64)
    validate_parameters(alphas, b_values)

    tasks = make_tasks(alphas, b_values)
    logical_cores = os.cpu_count() or 1
    workers = (max(1, int(logical_cores * CPU_FRACTION))
               if args.workers is None else args.workers)
    if not 1 <= workers <= logical_cores:
        raise ValueError('--workers须在1和逻辑CPU核心数之间。')
    workers = min(workers, len(tasks))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = args.output_dir.resolve() / ('run_' + timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    stem = 'liao2023_fig2b_beta0.50'
    versions = ' '.join(f'{name}={version(name)}'
                        for name in ('numpy', 'numba', 'matplotlib', 'tqdm'))
    metadata = (
        'source=Liao_et_al_AMC_439_2023_127635_Fig2b; DOI=10.1016/j.amc.2022.127635\n'
        f'L={L} N={L * L} network=periodic_square_von_Neumann degree=4\n'
        f'alpha_values={alphas.tolist()} beta={BETA} K={K}\n'
        'payoff_matrix_C_D_order=[[1,0],[b,0]]; payoff=accumulated_four_neighbors\n'
        'local_popularity=(same_strategy_neighbors+1)/5; '
        'global_popularity=whole_population_same_strategy_fraction_including_self\n'
        'fitness=payoff*(alpha*local_popularity+(1-alpha)*global_popularity)**beta\n'
        'update=N_uniform_focal_draws_with_replacement_per_MCS; '
        'neighbor=uniform_among_four; adoption=immediate\n'
        'global_fc=snapshot_at_MCS_start; local_counts=immediately_updated; '
        'reputation=False; mutation=False; regularization=False\n'
        'initialization=exact_N/2_cooperators_uniformly_randomized_positions\n'
        f'MCS_TOTAL={MCS_TOTAL} MCS_TRANSIENT={MCS_TRANSIENT} '
        f'averaging_MCS={MCS_TOTAL - MCS_TRANSIENT} NUM_RUNS={NUM_RUNS}\n'
        'fc_mean=mean_of_independent_run_time_means; fc_std=sample_std_ddof1; '
        'fc_sem=fc_std/sqrt(NUM_RUNS); no_error_bars_to_match_original_panel\n'
        f'b_values={b_values.tolist()}; b_grid=inferred_from_visible_markers\n'
        f'BASE_SEED={BASE_SEED}; seed_order=b_index,alpha_index,run_index; '
        'seed_method=SeedSequence_spawn_unique_uint32; author_seed_not_published\n'
        f'early_stop_absorbing={EARLY_STOP_ABSORBING}; '
        'only_exact_all_C_all_D; remaining_sampling_window_filled_exactly\n'
        f'logical_cores={logical_cores} workers={workers} '
        f'worker_core_fraction={workers / logical_cores:.6f}\n'
        f'python={platform.python_version()} {versions}\n'
    )
    print('Fig. 2(b): alpha=', alphas.tolist(), ' beta=', BETA, ' K=', K)
    print(f'{len(alphas)} alpha x {len(b_values)} b x {NUM_RUNS} runs = {len(tasks)} runs')
    print(f'{MCS_TOTAL} MCS/run; average final {MCS_TOTAL - MCS_TRANSIENT} MCS')
    print(f'Workers: {workers}/{logical_cores} logical cores '
          f'({100 * workers / logical_cores:.1f}%)')
    print('Numba首次编译可能使首个结果延迟；CPU总使用率会随运行阶段变化。')
    print('Output directory:', output_dir)

    values = np.full((len(alphas), len(b_values), NUM_RUNS), np.nan)
    counts = np.zeros(values.shape[:2], dtype=np.int64)
    context = mp.get_context('spawn')
    # 即使中断，finally也会记录已完成参数点；未完成点明确保留为nan。
    try:
        with context.Pool(
            processes=workers, initializer=initialize_worker,
            initargs=(L, BETA, K, MCS_TOTAL, MCS_TRANSIENT, EARLY_STOP_ABSORBING),
        ) as pool:
            for ai, bi, ri, fc in tqdm(
                pool.imap_unordered(execute_task, tasks, chunksize=1),
                total=len(tasks), desc='Fig. 2(b)', unit='run',
            ):
                if not np.isfinite(fc) or not 0.0 <= fc <= 1.0:
                    raise RuntimeError('合作比例不是[0,1]内的有限值。')
                if not np.isnan(values[ai, bi, ri]):
                    raise RuntimeError('同一任务重复返回结果。')
                values[ai, bi, ri] = fc
                counts[ai, bi] += 1
                if counts[ai, bi] == NUM_RUNS:
                    save_text(output_dir, stem, metadata, alphas, b_values, values, counts)
    finally:
        means = save_text(output_dir, stem, metadata, alphas, b_values, values, counts)

    if np.any(counts != NUM_RUNS) or np.any(~np.isfinite(means)):
        raise RuntimeError('结果未全部完成，不输出正式图片。')
    print('Plot TXT:', output_dir / (stem + '_plot_data.txt'))
    print('Summary TXT:', output_dir / (stem + '_summary.txt'))
    save_figure(output_dir, stem, alphas, b_values, means, show=not args.no_show)


if __name__ == '__main__':
    mp.freeze_support()
    main()
