import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

# ==========================================
# 1. 参数设定与模型常数 (严格保留您原本的设定)
# ==========================================
L = 100
N = L * L
K = 0.1  # 费米公式中的噪声 K
LAMBDA_VAL = 0.1  # 声誉衰减因子 gamma
EPSILON = 0.01  # 正则化常数 epsilon
MUTATION_RATE = 0  # 突变概率 u

# [修正] 恢复为正常模拟步数，确保数据收敛
MCS_TOTAL = 20000  # 原为 100 (过小)
MCS_TRANSIENT = 17000 # 原为 5 (过小)
NUM_RUNS = 100  # 独立运行次数

BETA_VALUES = [0.50]
ALPHA_VALUES = [0.60, 0.65, 0.80, 1.00]  # 根据您的文件保留了这组Alpha
B_VALUES = np.linspace(1, 1.8, 33)  # b 区间 [1.0, 1.8]


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


@jit(nopython=True)
def simulation_step_optimized(N, K, b, alpha, beta, lam, epsilon, u, strategies, reputations, neighbor_map):
    # --- [GPop 计算] 全局流行度 (在 MCS 开始时计算一次快照) ---
    total_rep = np.sum(reputations) + N * epsilon
    sum_c = 0.0
    for i in range(N):
        if strategies[i] == 1:
            sum_c += reputations[i] + epsilon

    gpop_C = sum_c / total_rep
    gpop_D = 1.0 - gpop_C

    # --- 策略更新 (MCS 循环) ---
    indices = np.random.permutation(N)
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


def run_single_experiment(b, alpha, beta, neighbor_map):
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

    return np.mean(fc_list)


# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    neighbor_map = build_neighbor_map(L)
    export_list = []
    plot_data = {}

    print("JIT Warmup...")
    dummy_s, dummy_r = np.ones(N, dtype=np.int64), np.ones(N, dtype=np.float64)
    simulation_step_optimized(N, K, 1.0, 0.5, 1.0, 0.1, 0.01, 0.05, dummy_s, dummy_r, neighbor_map)

    # 数据生成循环
    for beta in BETA_VALUES:
        plot_data[beta] = {}
        for alpha in ALPHA_VALUES:
            fc_vs_b = []
            print(f"Running: Beta={beta}, Alpha={alpha}")
            for b in tqdm(B_VALUES):
                runs_fc = [run_single_experiment(b, alpha, beta, neighbor_map) for _ in range(NUM_RUNS)]
                avg_fc = np.mean(runs_fc)
                fc_vs_b.append(avg_fc)
                export_list.append([beta, alpha, b, avg_fc])
            plot_data[beta][alpha] = fc_vs_b

    # 保存数据
    save_data = np.array(export_list)
    np.savetxt('6581u=0final_simulation_data_corrected.txt', save_data, fmt='%.6f', header='beta alpha b fc', comments='')
    print("数据已保存。")

    # ==========================================
    # --- 绘图部分 (根据 BETA_VALUES 数量自适应) ---
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C']
    markers = ['+', '+', 'o', 'o']
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']  # 多备用几个序号防越界

    num_betas = len(BETA_VALUES)

    # 动态调整画布大小，每个子图宽度分配5，高度4.5
    fig, axes = plt.subplots(1, num_betas, figsize=(5 * num_betas, 4.5))

    # 如果只有一个子图，axes 是一个 Axes 对象，将其转换为列表以便统一遍历
    if num_betas == 1:
        axes = [axes]

    for i, beta in enumerate(BETA_VALUES):
        ax = axes[i]
        for j, alpha in enumerate(ALPHA_VALUES):
            # 判断蓝色为空心，红色为实心
            mfc = 'none' if colors[j] == '#0000CD' else colors[j]

            # 绘图：图例中的 alpha 保留 '.2f' 格式
            ax.plot(B_VALUES, plot_data[beta][alpha],
                    label=rf'$\alpha={alpha:.2f}$',
                    color=colors[j],
                    marker=markers[j], markersize=6, linewidth=1.2,
                    markeredgecolor=colors[j], markerfacecolor=mfc)

        # ----------------------------------------------------
        # 精准控制坐标轴范围和强制自定义字符串刻度
        # ----------------------------------------------------

        # Y 轴严格控制 0 到 1
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # 替换刻度标签：0.0 改为 '0', 1.0 改为 '1'
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

        # X 轴根据您的 B_VALUES 严格控制 1.0 到 1.8
        ax.set_xlim(1.0, 1.8)
        ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
        # 替换刻度标签：起点的 1.0 强制改为 '1'
        ax.set_xticklabels(['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8'])

        ax.margins(0)  # 强制去除任何系统默认留白，实现顶格

        # 坐标轴刻度朝内 (in)，且四周均有边框
        ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)

        # 坐标轴标签 (斜体)
        ax.set_xlabel(r"$b$", fontsize=14)
        ax.set_ylabel(r"$f_c$", fontsize=14)

        # 只有在有多个 Beta 值（多张子图）时，才添加左上角的图序号 (a), (b), (c)
        if num_betas > 1:
            ax.text(0.04, 0.95, sub_labels[i], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')

        # 右下角的 Beta 标注 (保留 '.2f' 格式例如 1.00)
        ax.text(0.96, 0.05, rf"$\beta = {beta:.2f}$", transform=ax.transAxes,
                fontsize=13, ha='right', va='bottom')

        # 图例统一放置在右上角
        ax.legend(fontsize=10, frameon=True, edgecolor='black', loc='upper right', handlelength=1.5)

    plt.tight_layout()
    # 只有在多图的情况下，调整子图间距才有意义
    if num_betas > 1:
        plt.subplots_adjust(wspace=0.18)

    plt.savefig('6581u=0Final_Paper_Fig2_Exact_Ticks.png', dpi=300, bbox_inches='tight')
    plt.show()
