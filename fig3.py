import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
from tqdm import tqdm
import multiprocessing as mp
import os

# ==========================================
# 1. 参数设定 (严格保留你的核心逻辑)
# ==========================================
L = 100
N = L * L
K = 0.1
FIXED_B = 1.11
MCS_TOTAL = 10000
INIT_COOP_RATE = 0.5
NUM_RUNS = 100

LAMBDA_VAL = 0.1
MUTATION_RATE = 0.05
EPSILON = 0.01

BETA_VALUES = [0.30, 0.50, 0.80, 1.00]
ALPHA_VALUES = [0.00, 0.50, 0.80, 1.00]


# ==========================================
# 2. 核心算法 (JIT 加速)
# ==========================================

@jit(nopython=True)
def build_neighbor_map(L):
    N = L * L
    neighbors = np.empty((N, 4), dtype=np.int64)
    for i in range(N):
        row, col = i // L, i % L
        neighbors[i, 0] = ((row - 1 + L) % L) * L + col
        neighbors[i, 1] = ((row + 1) % L) * L + col
        neighbors[i, 2] = row * L + ((col - 1 + L) % L)
        neighbors[i, 3] = row * L + ((col + 1) % L)
    return neighbors


@jit(nopython=True)
def calculate_payoff(s_i, s_j, b):
    if s_i == 1:
        return 1.0 if s_j == 1 else 0.0
    else:
        return b if s_j == 1 else 0.0


@jit(nopython=True)
def single_mcs_step(N, K, b, alpha, beta, lam, epsilon, u, strategies, reputations, neighbor_map):
    total_rep = np.sum(reputations) + N * epsilon
    sum_c_rep = 0.0
    for i in range(N):
        if strategies[i] == 1: sum_c_rep += reputations[i] + epsilon
    gpop_c = sum_c_rep / total_rep
    gpop_d = 1.0 - gpop_c

    indices = np.random.permutation(N)
    for i in indices:
        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)
        else:
            j = neighbor_map[i, np.random.randint(0, 4)]
            if strategies[i] != strategies[j]:
                si, ri = strategies[i], reputations[i]
                pi, l_num_i, l_den_i = 0.0, ri + epsilon, ri + epsilon
                for idx in range(4):
                    nb = neighbor_map[i, idx]
                    pi += calculate_payoff(si, strategies[nb], b)
                    w = reputations[nb] + epsilon
                    l_den_i += w
                    if strategies[nb] == si: l_num_i += w
                tpop_i = alpha * (l_num_i / l_den_i) + (1.0 - alpha) * (gpop_c if si == 1 else gpop_d)
                Fi = pi * (max(tpop_i, 1e-9) ** beta)

                sj, rj = strategies[j], reputations[j]
                pj, l_num_j, l_den_j = 0.0, rj + epsilon, rj + epsilon
                for idx in range(4):
                    nb = neighbor_map[j, idx]
                    pj += calculate_payoff(sj, strategies[nb], b)
                    w = reputations[nb] + epsilon
                    l_den_j += w
                    if strategies[nb] == sj: l_num_j += w
                tpop_j = alpha * (l_num_j / l_den_j) + (1.0 - alpha) * (gpop_c if sj == 1 else gpop_d)
                Fj = pj * (max(tpop_j, 1e-9) ** beta)

                if np.random.random() < 1.0 / (1.0 + np.exp((Fi - Fj) / K)):
                    strategies[i] = strategies[j]

    for i in range(N):
        reputations[i] = (1.0 - lam) * reputations[i] + lam * float(strategies[i])
    return strategies, reputations


# ==========================================
# 3. 多进程任务封装与主程序
# ==========================================

def run_single_simulation(args):
    """用于多进程的单次模拟任务封装"""
    beta, alpha, mcs_total, init_coop, n, k, fixed_b, lam, eps, mut_rate, l_size = args

    neighbor_map = build_neighbor_map(l_size)
    strategies = (np.random.rand(n) < init_coop).astype(np.int64)
    reputations = np.full(n, 0.5, dtype=np.float64)

    acc_fc = np.zeros(mcs_total + 2)
    acc_fc[0] = init_coop
    acc_fc[1] = init_coop

    for mcs in range(1, mcs_total + 1):
        strategies, reputations = single_mcs_step(
            n, k, fixed_b, alpha, beta, lam, eps, mut_rate,
            strategies, reputations, neighbor_map
        )
        acc_fc[mcs + 1] = np.mean(strategies)

    return acc_fc


def main():
    start_time = time.time()
    results = {beta: {} for beta in BETA_VALUES}

    # 动态获取当前机器的 CPU 核心数，并计算 80% 的占用配额
    total_cores = os.cpu_count() or 1
    target_cores = max(1, int(total_cores * 0.8))

    print(f"演化模拟开始: u={MUTATION_RATE}, Runs={NUM_RUNS}")
    print(f"已开启多进程加速: 目标 CPU 利用率 80% (调用 {target_cores}/{total_cores} 核心)")

    # 开启进程池
    pool = mp.Pool(processes=target_cores)

    for beta in BETA_VALUES:
        for alpha in ALPHA_VALUES:
            print(f"计算: Beta={beta}, Alpha={alpha}")

            # 将独立的模拟运行任务打包为参数列表
            args_list = [(beta, alpha, MCS_TOTAL, INIT_COOP_RATE, N, K, FIXED_B,
                          LAMBDA_VAL, EPSILON, MUTATION_RATE, L)] * NUM_RUNS

            # 核心并行化：将 NUM_RUNS 次循环分发到多个 CPU 核心同时计算
            run_results = pool.map(run_single_simulation, args_list)

            # 归约结果
            acc_fc = np.zeros(MCS_TOTAL + 2)
            for res in run_results:
                acc_fc += res

            results[beta][alpha] = acc_fc / NUM_RUNS

    pool.close()
    pool.join()

    end_time = time.time()
    print(f"总计耗时: {end_time - start_time:.2f}s")

    # 数据保存
    file_name = f"3581Final_Data_u{MUTATION_RATE}.txt"
    with open(file_name, "w") as f:
        f.write("MCS\tBeta\tAlpha\tfc\n")
        for b in BETA_VALUES:
            for a in ALPHA_VALUES:
                for t, val in enumerate(results[b][a]):
                    f.write(f"{t}\t{b}\t{a}\t{val:.6f}\n")
    print(f"数据已完整保存至: {file_name}")

    plot_results(results)


# ==========================================
# 4. 绘图 (完美复刻论文 Fig.6 格式)
# ==========================================

def plot_results(results):
    # 设置全局字体为 Times New Roman 和 STIX 数学字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 颜色配置
    colors = ['#76EE00', '#FFD700', '#4169E1', '#FF0000', 'black']
    sub_labels = ['(a)', '(b)', '(c)', '(d)']

    # 调整画布比例使其紧凑且专业
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, beta in enumerate(BETA_VALUES):
        ax = axes[idx]

        # 避开索引 0 以防止对数坐标系报错
        data_len = len(results[beta][ALPHA_VALUES[0]])
        steps = np.arange(1, data_len)

        for i, alpha in enumerate(ALPHA_VALUES):
            y_data = results[beta][alpha][1:]
            ax.plot(steps, y_data, color=colors[i],
                    linewidth=2.0, label=rf"$\alpha = {alpha:.3f}$")

        # 坐标轴范围与比例调整
        ax.set_xscale('log')
        ax.set_xlim(1, MCS_TOTAL)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # 刻度朝内，显示顶部和右侧刻度
        ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)

        # 坐标轴标签 (使用论文中的斜体格式)
        ax.set_ylabel(r"$f_c$", fontsize=16, fontstyle='italic')
        ax.set_xlabel(r"$MCS$", fontsize=16, fontstyle='italic')

        # 子图序号放置于左上角，标准加粗
        ax.text(0.04, 0.92, sub_labels[idx], transform=ax.transAxes, fontsize=16, fontweight='bold')

        ax.text(0.96, 0.08, rf"$\beta = {beta:.3f}$", transform=ax.transAxes,
                fontsize=14, fontstyle='italic', ha='right', va='bottom')

        # 仅在第一个子图 (idx == 0) 中添加图例
        if idx == 0:
            legend = ax.legend(loc='upper right', frameon=True, fontsize=12, labelspacing=0.2)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.0)

    # 紧凑布局微调
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, wspace=0.2, hspace=0.25)

    save_name = f"3581Final_Evolution_u{MUTATION_RATE}_Fig6Style.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
