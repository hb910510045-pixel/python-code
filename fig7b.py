import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

# ==========================================
# 1. 参数设定与模型常数
# ==========================================
L = 100
N = L * L
K = 0.1  # 费米公式中的噪声 K
LAMBDA_VAL = 0.1  # 声誉衰减因子 gamma
EPSILON = 0.01  # 正则化常数 epsilon

MCS_TOTAL = 20000  # 模拟总步数
MCS_TRANSIENT = 17000  # 暂态步数
NUM_RUNS = 100  # 独立运行次数

# [修改点]：固定 alpha 和 beta，横坐标保留为 b，图例的多条线改为不同的 u
FIXED_ALPHA = 0.80
FIXED_BETA = 0.50
# [增加0.004] 插入在0.00和0.007之间
U_VALUES = [0.00, 0.006, 0.01, 0.05, 0.10]  # 5个不同的突变概率 u
B_VALUES = np.linspace(1.0, 1.8, 33)  # b 区间保留不变 [1.0, 1.8]


# ==========================================
# 2. 核心逻辑函数
# ==========================================

@jit(nopython=True, fastmath=True)  # [优化点] 开启 fastmath 榨干单核算力
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


@jit(nopython=True, fastmath=True)  # [优化点] 开启 fastmath
def calculate_payoff(s_i, s_j, b):
    """
    收益矩阵 M:
    C(1) D(0)
    C 1    0
    D b    0
    """
    if s_i == 1:  # Si = C
        return 1.0 if s_j == 1 else 0.0
    else:  # Si = D
        return b if s_j == 1 else 0.0


@jit(nopython=True, fastmath=True)  # [优化点] 开启 fastmath
def simulation_step_optimized(N, K, b, alpha, beta, lam, epsilon, u, strategies, reputations, neighbor_map, indices):
    # --- [GPop 计算] 全局流行度 (在 MCS 开始时计算一次快照) ---
    total_rep = np.sum(reputations) + N * epsilon
    sum_c = 0.0
    for i in range(N):
        if strategies[i] == 1:
            sum_c += reputations[i] + epsilon

    gpop_C = sum_c / total_rep
    gpop_D = 1.0 - gpop_C

    # --- 策略更新 (MCS 循环) ---
    # [优化点] 改用外部传入的 indices 原地打乱，避免每步分配内存
    np.random.shuffle(indices)

    for k in range(N):
        i = indices[k]

        # 1. 突变
        if np.random.random() < u:
            strategies[i] = np.random.randint(0, 2)

        # 2. 选择与模仿
        else:
            j = neighbor_map[i, np.random.randint(0, 4)]

            if strategies[i] != strategies[j]:
                # --- 计算 F_i ---
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

                # --- 计算 F_j ---
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

                # --- 费米规则 ---
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


def run_single_experiment(u, b, alpha, beta, neighbor_map):
    strategies = (np.random.rand(N) < 0.5).astype(np.int64)
    reputations = np.full(N, 0.5, dtype=np.float64)
    # [优化点] 提前分配好用于 shuffle 的数组，避免在每次循环中重新分配
    indices = np.arange(N, dtype=np.int64)
    fc_list = []

    for mcs in range(MCS_TOTAL):
        strategies, reputations = simulation_step_optimized(
            N, K, b, alpha, beta, LAMBDA_VAL, EPSILON, u,
            strategies, reputations, neighbor_map, indices
        )
        if mcs >= MCS_TRANSIENT:
            fc_list.append(np.mean(strategies))

    return np.mean(fc_list)


# [新增包装函数] 专用于多进程工作节点
def worker_task(args):
    u, b, alpha, beta, neighbor_map, num_runs = args
    # 初始化进程专属的随机种子，保证并行下的随机性纯度
    np.random.seed()
    runs_fc = [run_single_experiment(u, b, alpha, beta, neighbor_map) for _ in range(num_runs)]
    return u, b, np.mean(runs_fc)


# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    neighbor_map = build_neighbor_map(L)
    export_list = []
    plot_data = {u: [] for u in U_VALUES}

    print("JIT Warmup (预热)...")
    dummy_s, dummy_r = np.ones(N, dtype=np.int64), np.ones(N, dtype=np.float64)
    dummy_indices = np.arange(N, dtype=np.int64)
    simulation_step_optimized(N, K, 1.0, 0.5, 1.0, 0.1, 0.01, 0.05, dummy_s, dummy_r, neighbor_map, dummy_indices)

    print(f"Fixed Alpha={FIXED_ALPHA}, Fixed Beta={FIXED_BETA}")

    # [优化点] 构建计算任务队列
    tasks = []
    for u in U_VALUES:
        for b in B_VALUES:
            tasks.append((u, b, FIXED_ALPHA, FIXED_BETA, neighbor_map, NUM_RUNS))

    total_tasks = len(tasks)
    max_cores = os.cpu_count()  # 获取当前机器的物理/逻辑核心总数
    print(f"准备榨干 CPU！检测到 {max_cores} 个核心，共有 {total_tasks} 个参数组合任务需要计算。")

    # [优化点] 启用多进程池，核心火力全开
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        # executor.map 保证了返回结果的顺序与 tasks 输入顺序完全一致
        results = list(tqdm(executor.map(worker_task, tasks), total=total_tasks, desc="并行计算进度"))

    # 将结果按照原先的逻辑填入字典和导出列表
    for res in results:
        u, b, avg_fc = res
        export_list.append([u, b, avg_fc])
        plot_data[u].append(avg_fc)

    # 保存数据 (表头更新为 u b fc)
    save_data = np.array(export_list)
    np.savetxt('α=0.8u=0.01modified_simulation_data_u_lines.txt', save_data, fmt='%.6f', header='u b fc', comments='')
    print("数据已保存。")

    # ==========================================
    # --- 绘图部分 ---
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    colors = ['#e4d00a', '#228B22', '#00ced1', '#0000CD', '#DC143C']
    markers = ['*', 's', '+', 'o', 'o']

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for j, u in enumerate(U_VALUES):
        mfc = 'none' if colors[j] == '#0000CD' else colors[j]
        msize = 8 if markers[j] == '*' else 6

        ax.plot(B_VALUES, plot_data[u],
                label=rf'$u={u:.3f}$',
                color=colors[j],
                marker=markers[j], markersize=msize, linewidth=1.2,
                markeredgecolor=colors[j], markerfacecolor=mfc)

    # ----------------------------------------------------
    # 精准控制坐标轴范围和强制自定义字符串刻度
    # ----------------------------------------------------
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    ax.xlim = (1.0, 1.8)
    ax.set_xlim(1.0, 1.8)
    ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    ax.set_xticklabels(['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8'])

    ax.margins(0)
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)

    ax.set_xlabel(r"$b$", fontsize=14)
    ax.set_ylabel(r"$f_c$", fontsize=14)

    ax.legend(fontsize=10, frameon=True, edgecolor='black', loc='upper right', handlelength=1.5)

    plt.tight_layout()
    plt.savefig('α=0.8u=0.01Modified_Paper_Fig_U_Lines.png', dpi=300, bbox_inches='tight')
    plt.show()
