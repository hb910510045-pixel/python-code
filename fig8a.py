import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

# ==========================================================
# 1. 正式模拟参数：只扫描晶格尺寸 L，其余参数固定
# ==========================================================
L_VALUES = [50, 100, 150, 200, 250]  # N = L * L
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
FIXED_GAMMA = 0.10
FIXED_U = 0.00  # 沿用源代码；u 为随机策略重置概率，有效翻转概率为 u/2
K = 0.1
EPSILON = 0.01
INITIAL_C_PROBABILITY = 0.5  # 沿用源代码：各节点独立以概率 0.5 初始化为 C

B_VALUES = np.linspace(1.0, 1.8, 33)
MCS_TOTAL = 20000
MCS_TRANSIENT = 17000 # 对最后 3000 MCS 的 f_c 取时间平均
NUM_RUNS = 100
BASE_SEED = 20261007
PNG_DPI = 600


# ==========================================================
# 2. 原模型的更新规则
#    保留周期边界、每 MCS 随机排列节点、全局背书的 MCS 开始快照、
#    MCS 末尾同步更新声誉，以及随机重置而非强制策略翻转。
# ==========================================================
@njit(cache=True)
def build_neighbor_map(L):
    """构建周期边界下的四邻居 von Neumann 邻域。"""
    N = L * L
    neighbors = np.empty((N, 4), dtype=np.int64)
    for i in range(N):
        row, col = i // L, i % L
        neighbors[i, 0] = ((row - 1 + L) % L) * L + col
        neighbors[i, 1] = ((row + 1) % L) * L + col
        neighbors[i, 2] = row * L + ((col - 1 + L) % L)
        neighbors[i, 3] = row * L + ((col + 1) % L)
    return neighbors


@njit(cache=True)
def calculate_payoff(s_i, s_j, b):
    """弱囚徒困境：R=1, S=0, T=b, P=0；C=1，D=0。"""
    if s_i == 1:
        return 1.0 if s_j == 1 else 0.0
    return b if s_j == 1 else 0.0


@njit(cache=True)
def simulation_step_optimized(N, K, b, alpha, beta, gamma, epsilon, u,
                              strategies, reputations, neighbor_map):
    total_rep = np.sum(reputations) + N * epsilon
    sum_c = 0.0
    for i in range(N):
        if strategies[i] == 1:
            sum_c += reputations[i] + epsilon

    gpop_C = sum_c / total_rep
    gpop_D = 1.0 - gpop_C

    indices = np.random.permutation(N)
    for k in range(N):
        i = indices[k]

        # 突变：随机重置为 C 或 D，也可能保留原策略。
        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)
        else:
            j = neighbor_map[i, np.random.randint(0, 4)]
            if strategies[i] != strategies[j]:
                p_i = 0.0
                num_i = reputations[i] + epsilon
                den_i = reputations[i] + epsilon
                for idx in range(4):
                    nb = neighbor_map[i, idx]
                    p_i += calculate_payoff(strategies[i], strategies[nb], b)
                    w = reputations[nb] + epsilon
                    den_i += w
                    if strategies[nb] == strategies[i]:
                        num_i += w

                lpop_i = num_i / den_i
                tpop_i = alpha * lpop_i + (1.0 - alpha) * (
                    gpop_C if strategies[i] == 1 else gpop_D
                )
                if tpop_i < 1e-9:
                    tpop_i = 1e-9
                fit_i = p_i * (tpop_i ** beta)

                p_j = 0.0
                num_j = reputations[j] + epsilon
                den_j = reputations[j] + epsilon
                for idx in range(4):
                    nb = neighbor_map[j, idx]
                    p_j += calculate_payoff(strategies[j], strategies[nb], b)
                    w = reputations[nb] + epsilon
                    den_j += w
                    if strategies[nb] == strategies[j]:
                        num_j += w

                lpop_j = num_j / den_j
                tpop_j = alpha * lpop_j + (1.0 - alpha) * (
                    gpop_C if strategies[j] == 1 else gpop_D
                )
                if tpop_j < 1e-9:
                    tpop_j = 1e-9
                fit_j = p_j * (tpop_j ** beta)

                exp_val = (fit_i - fit_j) / K
                if exp_val > 20:
                    prob = 0.0
                elif exp_val < -20:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + np.exp(exp_val))

                if np.random.random() < prob:
                    strategies[i] = strategies[j]

    for i in range(N):
        reputations[i] = (1.0 - gamma) * reputations[i] + gamma * float(strategies[i])
    return strategies, reputations


@njit(cache=True)
def run_single_experiment(b, alpha, beta, gamma, u, neighbor_map, seed,
                          mcs_total, mcs_transient, early_stop=True):
    """返回稳态时间平均、实际初始合作比例和吸收时刻。

    seed 在 Numba 内设定，使每次独立运行可重复。
    N 从当前邻接表确定，避免扫描 L 时仍使用全局 N=10000。
    仅当 u=0 且策略已全 C/全 D 时提前结束；未模拟的采样步
    用该吸收状态精确补齐，保证仍对应完整的采样窗口。
    """
    np.random.seed(seed)
    N = neighbor_map.shape[0]
    strategies = (np.random.random(N) < INITIAL_C_PROBABILITY).astype(np.int64)
    reputations = np.full(N, 0.5, dtype=np.float64)
    fc_initial = np.mean(strategies)
    fc_sum = 0.0
    num_samples = mcs_total - mcs_transient
    absorbed_mcs = 0  # 0 表示在本次模拟期间没有检测到吸收

    for mcs in range(mcs_total):
        strategies, reputations = simulation_step_optimized(
            N, K, b, alpha, beta, gamma, EPSILON, u,
            strategies, reputations, neighbor_map
        )
        fc = np.mean(strategies)
        if mcs >= mcs_transient:
            fc_sum += fc

        if u == 0.0 and (fc == 0.0 or fc == 1.0):
            if absorbed_mcs == 0:
                absorbed_mcs = mcs + 1
            if early_stop:
                remaining_samples = mcs_total - max(mcs + 1, mcs_transient)
                fc_sum += remaining_samples * fc
                return fc_sum / num_samples, fc_initial, absorbed_mcs

    return fc_sum / num_samples, fc_initial, absorbed_mcs


# ==========================================================
# 3. 输出图形：保持源代码的尺寸、颜色、符号、坐标和图例位置
# ==========================================================
def plot_population_sizes(l_values, b_values, plot_data, output_dir, stem, show=True):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C', '#228B22']
    markers = ['*', '+', 'o', 'o', 's']
    fig, ax = plt.subplots(figsize=(5, 4.5))

    for j, L in enumerate(l_values):
        color = colors[j]
        marker = markers[j]
        mfc = 'none' if color == '#0000CD' else color
        msize = 8 if marker == '*' else 6
        ax.plot(
            b_values, plot_data[L], label=rf'$L={L}$',
            color=color, marker=marker, markersize=msize, linewidth=1.2,
            markeredgecolor=color, markerfacecolor=mfc
        )

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.set_xlim(1.0, 1.8)
    ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    ax.set_xticklabels(['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8'])
    ax.margins(0)
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
    ax.set_xlabel(r'$b$', fontsize=16)
    ax.set_ylabel(r'$f_c$', fontsize=16)
    ax.legend(fontsize=12, frameon=True, framealpha=1.0, edgecolor='black',
              loc='upper right', handlelength=1.5)
    plt.tight_layout()

    png_path = output_dir / (stem + '.png')
    eps_path = output_dir / (stem + '.eps')
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches='tight')
    fig.savefig(eps_path, dpi=PNG_DPI, bbox_inches='tight', format='eps')
    print('PNG:', png_path)
    print('EPS:', eps_path)
    if show:
        plt.show()
    plt.close(fig)
    return png_path, eps_path


def main():
    parser = argparse.ArgumentParser(description='Compare square-lattice population sizes.')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Use a short test configuration, not manuscript results.')
    parser.add_argument('--no-show', action='store_true', help='Save figures without a window.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'population_size_results')
    args = parser.parse_args()

    if args.smoke_test:
        l_values = [10, 15, 20, 25, 30]
        b_values = np.array([1.0, 1.4, 1.8])
        num_runs, mcs_total, mcs_transient = 3, 60, 40
        mode = 'SMOKE_TEST_NOT_FOR_PUBLICATION'
        suffix = '_SMOKE_TEST'
    else:
        l_values = list(L_VALUES)
        b_values = np.asarray(B_VALUES, dtype=np.float64)
        num_runs, mcs_total, mcs_transient = NUM_RUNS, MCS_TOTAL, MCS_TRANSIENT
        mode = 'FULL_SIMULATION'
        suffix = ''

    if len(l_values) != 5 or any(L < 3 for L in l_values):
        raise ValueError('Use five lattice sizes L >= 3 to match the original plotting style.')
    if not (0 <= mcs_transient < mcs_total) or num_runs < 1:
        raise ValueError('Require 0 <= MCS_TRANSIENT < MCS_TOTAL and NUM_RUNS >= 1.')
    if not (0 <= FIXED_U <= 1) or not (0 <= FIXED_GAMMA <= 1):
        raise ValueError('FIXED_U and FIXED_GAMMA must lie in [0, 1].')

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ('population_sizes_alpha{:.2f}_beta{:.2f}_gamma{:.2f}_u{:.2f}{}'.format(
        FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U, suffix
    ))
    summary_path = output_dir / (stem + '_summary.txt')

    metadata = (
            'mode={}\n'.format(mode)
            + 'alpha={} beta={} gamma={} u_reset={} u_flip={} K={} epsilon={}\n'.format(
        FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U, FIXED_U / 2, K, EPSILON
    )
            + 'L_values={} N=L^2 periodic_boundary=True degree=4\n'.format(l_values)
            + 'MCS_TOTAL={} MCS_TRANSIENT={} averaging_MCS={} NUM_RUNS={} BASE_SEED={}\n'.format(
        mcs_total, mcs_transient, mcs_total - mcs_transient, num_runs, BASE_SEED
    )
            + 'initialization=independent_Bernoulli(p_C={}); reputation_initial=0.5\n'.format(
        INITIAL_C_PROBABILITY
    )
            + 'update=random_permutation_per_MCS; global_endorsement=start_of_MCS_snapshot; '
              'reputation_update=synchronous_end_of_MCS\n'
            + 'fc_run=mean_over_sampling_MCS; fc_mean=mean_of_independent_run_means; '
              'fc_std=sample_std_of_run_means(ddof=1); fc_sem=fc_std/sqrt(NUM_RUNS)\n'
            + 'u=0 absorbing states use exact continuation for the sampling window.\n'
    )

    print('Mode:', mode)
    print('Fixed alpha={}, beta={}, gamma={}, u={}, K={}'.format(
        FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA, FIXED_U, K
    ))
    print('L:', l_values, '; N:', [L * L for L in l_values])
    print('Steady-state averaging: final {} MCS; {} independent runs per point.'.format(
        mcs_total - mcs_transient, num_runs
    ))
    print('Output directory:', output_dir)
    print('JIT warmup...')
    warmup_neighbors = build_neighbor_map(4)
    run_single_experiment(1.1, FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA,
                          FIXED_U, warmup_neighbors, 0, 2, 1)

    plot_data = {}
    # 仅保留平均数据的汇总文件写入
    with summary_path.open('w', encoding='utf-8') as summary_file:
        for line in metadata.rstrip().splitlines():
            summary_file.write('# ' + line + '\n')
        summary_file.write('# L N b fc_mean fc_std fc_sem n_runs\n')
        summary_file.flush()

        for L in l_values:
            neighbor_map = build_neighbor_map(L)
            N = L * L
            fc_vs_b = []
            print('\nRunning L={}, N={}'.format(L, N))
            with tqdm(total=len(b_values) * num_runs, desc='L={}'.format(L),
                      unit='run') as progress:
                for b_index, b in enumerate(b_values):
                    runs_fc = np.empty(num_runs, dtype=np.float64)
                    for run_index in range(num_runs):
                        # 每组 (L, b, run) 独立分配种子，不依赖前一次运行的耗时或随机数消耗。
                        seed = int(np.random.SeedSequence(
                            [BASE_SEED, L, b_index, run_index]
                        ).generate_state(1)[0])
                        fc, fc_initial, absorbed_mcs = run_single_experiment(
                            float(b), FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA,
                            FIXED_U, neighbor_map, seed, mcs_total, mcs_transient
                        )
                        runs_fc[run_index] = fc
                        progress.update(1)

                    avg_fc = float(np.mean(runs_fc))
                    std_fc = float(np.std(runs_fc, ddof=1)) if num_runs > 1 else float('nan')
                    sem_fc = std_fc / np.sqrt(num_runs)
                    fc_vs_b.append(avg_fc)
                    summary_file.write('{} {} {:.8f} {:.10f} {:.10f} {:.10f} {}\n'.format(
                        L, N, b, avg_fc, std_fc, sem_fc, num_runs
                    ))
                    summary_file.flush()
            plot_data[L] = np.asarray(fc_vs_b)

    print('\nSummary TXT:', summary_path)
    plot_population_sizes(l_values, b_values, plot_data, output_dir, stem,
                          show=not args.no_show)


if __name__ == '__main__':
    main()
