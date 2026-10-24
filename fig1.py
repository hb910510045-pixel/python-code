import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import multiprocessing as mp
import time

# ==========================================
# 1. 参数设定与模型常数
# ==========================================
L = 100
N = L * L
K = 0.1  # 费米公式中的噪声 K
LAMBDA_VAL = 0.1  # 声誉衰减因子 gamma
EPSILON = 0.01  # 正则化常数 epsilon
MUTATION_RATE = 0.05  # 突变概率 u

# 严格的长时间演化设定
MCS_TOTAL = 20000  # 保证系统充分收敛
MCS_TRANSIENT = 17000  # 取最后3000步求稳态平均
NUM_RUNS = 100  # 独立运行100次消除偶然性

BETA_VALUES = [0.30, 0.50, 0.80]
ALPHA_VALUES = [0.50, 0.60, 0.80, 1.00]
B_VALUES = np.linspace(1.0, 1.8, 33)


# ==========================================
# 2. 核心逻辑函数
# ==========================================

@jit(nopython=True)
def build_neighbor_map(L):
    """ 构建 von Neumann 邻域 (上下左右) """
    N = L * L
    neighbors = np.empty((N, 4), dtype=np.int64)
    for i in range(N):
        row, col = i // L, i % L
        neighbors[i, 0] = ((row - 1 + L) % L) * L + col  # North
        neighbors[i, 1] = ((row + 1) % L) * L + col  # South
        neighbors[i, 2] = row * L + ((col - 1 + L) % L)  # West
        neighbors[i, 3] = row * L + ((col + 1) % L)  # East
    return neighbors


@jit(nopython=True)
def calculate_payoff(s_i, s_j, b):
    if s_i == 1:  # Si = C
        return 1.0 if s_j == 1 else 0.0
    else:  # Si = D
        return b if s_j == 1 else 0.0


@jit(nopython=True)
def simulation_step_optimized(N, K, b, alpha, beta, lam, epsilon, u, strategies, reputations, neighbor_map):
    # --- [GPop 计算] 全局流行度 ---
    total_rep = np.sum(reputations) + N * epsilon
    sum_c = 0.0
    for i in range(N):
        if strategies[i] == 1:
            sum_c += reputations[i] + epsilon

    gpop_C = sum_c / total_rep
    gpop_D = 1.0 - gpop_C

    # --- 策略更新 ---
    indices = np.random.permutation(N)
    for k in range(N):
        i = indices[k]

        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)
        else:
            j = neighbor_map[i, np.random.randint(0, 4)]
            if strategies[i] != strategies[j]:
                # 计算 F_i
                p_i = 0.0
                num_i, den_i = reputations[i] + epsilon, reputations[i] + epsilon
                for idx in range(4):
                    nb = neighbor_map[i, idx]
                    p_i += calculate_payoff(strategies[i], strategies[nb], b)
                    w = reputations[nb] + epsilon
                    den_i += w
                    if strategies[nb] == strategies[i]:
                        num_i += w

                lpop_i = num_i / den_i
                tpop_i = alpha * lpop_i + (1.0 - alpha) * (gpop_C if strategies[i] == 1 else gpop_D)
                if tpop_i < 1e-9: tpop_i = 1e-9
                fit_i = p_i * (tpop_i ** beta)

                # 计算 F_j
                p_j = 0.0
                num_j, den_j = reputations[j] + epsilon, reputations[j] + epsilon
                for idx in range(4):
                    nb = neighbor_map[j, idx]
                    p_j += calculate_payoff(strategies[j], strategies[nb], b)
                    w = reputations[nb] + epsilon
                    den_j += w
                    if strategies[nb] == strategies[j]:
                        num_j += w

                lpop_j = num_j / den_j
                tpop_j = alpha * lpop_j + (1.0 - alpha) * (gpop_C if strategies[j] == 1 else gpop_D)
                if tpop_j < 1e-9: tpop_j = 1e-9
                fit_j = p_j * (tpop_j ** beta)

                # 费米规则
                exp_val = (fit_i - fit_j) / K
                if exp_val > 20:
                    prob = 0.0
                elif exp_val < -20:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + np.exp(exp_val))

                if np.random.random() < prob:
                    strategies[i] = strategies[j]

    # --- 声誉更新 ---
    for i in range(N):
        reputations[i] = (1.0 - lam) * reputations[i] + lam * float(strategies[i])

    return strategies, reputations


# ==========================================
# 3. 随机数隔离与多进程包裹函数
# ==========================================

@jit(nopython=True)
def set_numba_random_seed(seed):
    """强制重置 Numba 底层 C 随机数生成器的种子"""
    np.random.seed(seed)


def worker_task(args):
    """多进程单次任务的执行体"""
    b, alpha, beta, run_idx = args

    # 【核心防御】生成绝对独一无二的随机种子，防止进程间状态拷贝导致克隆模拟
    # 结合了时间戳的微秒级变动 + 传入参数的哈希值
    unique_seed = hash((time.time(), b, alpha, beta, run_idx)) % (2 ** 32 - 1)

    # 彻底重置 Python 环境和 Numba 编译环境的随机流
    np.random.seed(unique_seed)
    set_numba_random_seed(unique_seed)

    neighbor_map = build_neighbor_map(L)
    strategies = (np.random.rand(N) < 0.5).astype(np.int64)
    reputations = np.full(N, 0.5, dtype=np.float64)
    fc_list = []

    for mcs in range(MCS_TOTAL):
        strategies, reputations = simulation_step_optimized(
            N, K, b, alpha, beta, LAMBDA_VAL, EPSILON, MUTATION_RATE,
            strategies, reputations, neighbor_map
        )
        if mcs >= MCS_TRANSIENT:
            fc_list.append(np.mean(strategies))

    avg_fc = np.mean(fc_list)
    return (beta, alpha, b, avg_fc)


# ==========================================
# 4. 主程序 (多进程调度与绘图)
# ==========================================

if __name__ == "__main__":
    # --- 准备任务池 ---
    tasks = []
    for beta in BETA_VALUES:
        for alpha in ALPHA_VALUES:
            for b in B_VALUES:
                for run_idx in range(NUM_RUNS):
                    tasks.append((b, alpha, beta, run_idx))

    total_tasks = len(tasks)

    # 根据机器硬件自动决定使用的 CPU 核心数量 (留 1 个核保障系统不卡死)
    available_cores = mp.cpu_count()
    use_cores = max(1, available_cores - 1)
    print(f"准备开始蒙特卡洛模拟...")
    print(f"任务总数: {total_tasks} | 使用 CPU 核心数: {use_cores}")

    # 数据归档字典
    results_accum = {beta: {alpha: {b: [] for b in B_VALUES} for alpha in ALPHA_VALUES} for beta in BETA_VALUES}

    # --- 启动多进程池 ---
    # 使用 imap_unordered 结合 tqdm 可以实时看到完成进度
    with mp.Pool(processes=use_cores) as pool:
        for res in tqdm(pool.imap_unordered(worker_task, tasks), total=total_tasks, desc="Simulation Progress"):
            res_beta, res_alpha, res_b, res_fc = res
            results_accum[res_beta][res_alpha][res_b].append(res_fc)

    print("模拟完成，正在整理数据与绘图...")

    # --- 汇总结果 ---
    export_list = []
    plot_data = {beta: {alpha: [] for alpha in ALPHA_VALUES} for beta in BETA_VALUES}

    for beta in BETA_VALUES:
        for alpha in ALPHA_VALUES:
            for b in B_VALUES:
                # 把 100 次独立实验的结果平均化
                avg_fc = np.mean(results_accum[beta][alpha][b])
                plot_data[beta][alpha].append(avg_fc)
                export_list.append([beta, alpha, b, avg_fc])

    # 保存数据
    save_data = np.array(export_list)
    np.savetxt('5681358final_simulation_data_MP.txt', save_data, fmt='%.6f', header='beta alpha b fc', comments='')
    print("数据已保存至 '3481final_simulation_data_MP.txt'")

    # ==========================================
    # --- 绘图部分 (纯格式严格按照上一版的参数保留) ---
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    # 模仿图二的鲜明颜色与标记（alpha=0.3 更亮的黄色星号等）
    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C']
    markers = ['*', '+', 'o', 'o']
    sub_labels = ['(a)', '(b)', '(c)']

    # 1行3列布局
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for i, beta in enumerate(BETA_VALUES):
        ax = axes[i]
        for j, alpha in enumerate(ALPHA_VALUES):
            # 蓝色为空心圈，红色为实心圈
            mfc = 'none' if colors[j] == '#0000CD' else colors[j]

            ax.plot(B_VALUES, plot_data[beta][alpha],
                    label=rf'$\alpha={alpha:.2f}$',
                    color=colors[j],
                    marker=markers[j], markersize=6, linewidth=1.2,
                    markeredgecolor=colors[j], markerfacecolor=mfc)

        # ----------------------------------------------------
        # 严格顶格与自定义标签替换
        # ----------------------------------------------------
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

        ax.set_xlim(1.0, 1.8)
        ticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        ax.set_xticks(ticks)
        ax.set_xticklabels(['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8'])

        ax.margins(0)  # 取消任何自动补边

        # 边框刻度朝内
        ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)

        # 标签说明
        ax.set_xlabel(r"$b$", fontsize=14)
        ax.set_ylabel(r"$f_c$", fontsize=14)

        # 序号与 Beta 值
        ax.text(0.04, 0.95, sub_labels[i], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')
        ax.text(0.96, 0.05, rf"$\beta = {beta:.2f}$", transform=ax.transAxes,
                fontsize=13, ha='right', va='bottom')

        # 图例统一放置右上角
        ax.legend(fontsize=10, frameon=True, edgecolor='black', loc='upper right', handlelength=1.5)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    plt.savefig('5681358Final_Optimized_MP_Plot.png', dpi=300, bbox_inches='tight')
    plt.show()
