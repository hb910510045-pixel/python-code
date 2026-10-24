import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange  # 引入 prange 用于多线程并行
import time
import pandas as pd
from tqdm import tqdm  # 引入 tqdm 进度条库

# ==========================================
# 1. 参数设定
# ==========================================
L = 100  # 线性尺寸 (L x L 网格) - 论文中为了最终出图使用的是 L=100
N = L * L  # 总人数
K = 0.1  # 费米噪声 K
B_VAL = 1.11  # 背叛诱惑 b (固定为 1.11)
MCS_TOTAL = 20000  # 每个点的模拟总步数 - 论文最终出图建议使用 10000+
AVG_LAST = 3000  # 取最后多少步求稳态平均
LAMBDA_VAL = 0.1  # 声誉更新参数 lambda
MUTATION_RATE = 0.05  # 突变率 u
EPSILON = 0.01  # 防止分母为0的微小量

NUM_RUNS = 100  # 每个参数点取100次实验的平均值

# ================= 关键修改区 =================
# 将 BETA_RANGE 的下限改回 0.0
# 为了保证测试速度，这里步长仍设为 0.1
BETA_RANGE = np.round(np.arange(0.0, 1.01, 0.025), 3)
ALPHA_RANGE = np.round(np.arange(0.0, 1.01, 0.025), 3)


# ==============================================


# ==========================================
# 2. 核心算法 (Numba JIT 加速)
# ==========================================
@jit(nopython=True)
def build_neighbor_map(L):
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
def run_simulation_core(L, N, K, b, alpha, beta, lam, epsilon, u, mcs_steps, avg_steps, neighbor_map):
    strategies = (np.random.random(N) < 0.5).astype(np.int64)
    reputations = np.full(N, 0.5, dtype=np.float64)

    fc_sum = 0.0
    count_avg = 0
    avg_start_step = mcs_steps - avg_steps
    inv_K = 1.0 / K  # 【优化2】提前计算乘法常数，替代除法

    for t in range(mcs_steps):
        total_rep = np.sum(reputations) + N * epsilon
        sum_c_rep = 0.0
        for i in range(N):
            if strategies[i] == 1:
                sum_c_rep += reputations[i] + epsilon

        gpop_c = sum_c_rep / total_rep
        gpop_d = 1.0 - gpop_c

        for _ in range(N):
            i = np.random.randint(0, N)
            if np.random.random() < u:
                strategies[i] = np.random.randint(0, 2)
            else:
                j = neighbor_map[i, np.random.randint(0, 4)]
                if strategies[i] != strategies[j]:
                    si, ri = strategies[i], reputations[i]
                    pi = 0.0
                    l_num_i, l_den_i = ri + epsilon, ri + epsilon
                    for idx in range(4):
                        nb = neighbor_map[i, idx]
                        pi += calculate_payoff(si, strategies[nb], b)
                        w = reputations[nb] + epsilon
                        l_den_i += w
                        if strategies[nb] == si:
                            l_num_i += w

                    lpop_i = l_num_i / l_den_i
                    tpop_i = alpha * lpop_i + (1.0 - alpha) * (gpop_c if si == 1 else gpop_d)
                    Fi = pi * (max(tpop_i, 1e-9) ** beta)

                    sj, rj = strategies[j], reputations[j]
                    pj = 0.0
                    l_num_j, l_den_j = rj + epsilon, rj + epsilon
                    for idx in range(4):
                        nb = neighbor_map[j, idx]
                        pj += calculate_payoff(sj, strategies[nb], b)
                        w = reputations[nb] + epsilon
                        l_den_j += w
                        if strategies[nb] == sj:
                            l_num_j += w

                    lpop_j = l_num_j / l_den_j
                    tpop_j = alpha * lpop_j + (1.0 - alpha) * (gpop_c if sj == 1 else gpop_d)
                    Fj = pj * (max(tpop_j, 1e-9) ** beta)

                    # 【优化2】使用乘法 inv_K 替代原来的除法 / K
                    fermi_prob = 1.0 / (1.0 + np.exp((Fi - Fj) * inv_K))
                    if np.random.random() < fermi_prob:
                        strategies[i] = strategies[j]

        for i in range(N):
            reputations[i] = (1.0 - lam) * reputations[i] + lam * float(strategies[i])

        if t >= avg_start_step:
            fc_sum += np.mean(strategies)
            count_avg += 1

    if count_avg > 0:
        return fc_sum / count_avg
    else:
        return 0.0


# ==========================================
# 【优化1】新增多线程并行函数
# ==========================================
@jit(nopython=True, parallel=True)
def run_multiple_simulations(L, N, K, b, alpha, beta, lam, epsilon, u, mcs_steps, avg_steps, neighbor_map, num_runs):
    fc_accum = 0.0
    # prange 会自动将循环分配到所有 CPU 核心上
    for i in prange(num_runs):
        fc_accum += run_simulation_core(L, N, K, b, alpha, beta, lam, epsilon, u, mcs_steps, avg_steps, neighbor_map)
    return fc_accum / num_runs


# ==========================================
# 3. 主控制程序 (引入 tqdm)
# ==========================================
def main():
    start_time = time.time()
    neighbor_map = build_neighbor_map(L)
    results_matrix = np.zeros((len(ALPHA_RANGE), len(BETA_RANGE)))

    print(f"开始模拟: b={B_VAL}, L={L}, Runs={NUM_RUNS}")
    print(f"Alpha数量: {len(ALPHA_RANGE)}, Beta数量: {len(BETA_RANGE)}")
    print("-" * 60)

    total_points = len(ALPHA_RANGE) * len(BETA_RANGE)

    # 使用 tqdm 包装循环，提供进度条和预估时间
    with tqdm(total=total_points, desc="模拟进度") as pbar:
        for i, alpha in enumerate(ALPHA_RANGE):
            for j, beta in enumerate(BETA_RANGE):
                # 【优化1】用并行函数替代原本的单核循环
                fc_final = run_multiple_simulations(
                    L, N, K, B_VAL, alpha, beta,
                    LAMBDA_VAL, EPSILON, MUTATION_RATE,
                    MCS_TOTAL, AVG_LAST, neighbor_map, NUM_RUNS
                )

                results_matrix[i, j] = fc_final

                # 在进度条右侧动态显示当前的参数和结果
                pbar.set_postfix({
                    'Alpha': f"{alpha:.3f}",
                    'Beta': f"{beta:.3f}",
                    'Fc': f"{fc_final:.4f}"
                })
                pbar.update(1)  # 手动更新进度条 1 步

    print("\n" + "-" * 60)
    print(f"所有模拟结束，总耗时: {time.time() - start_time:.2f}s")

    df = pd.DataFrame(results_matrix, index=ALPHA_RANGE, columns=BETA_RANGE)
    df.to_csv("0.025heatmap_data_positive.csv")
    print("数据已保存至 0.025heatmap_data_positive.csv")

    plot_heatmap(results_matrix)


# ==========================================
# 4. 绘图函数 (高度还原参考图格式，修复 fc 字体)
# ==========================================
def plot_heatmap(data_matrix):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'  # 开启类似 Times 的数学公式字体支持

    X, Y = np.meshgrid(BETA_RANGE, ALPHA_RANGE)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    cf = ax.contourf(X, Y, data_matrix, levels=100, cmap='jet', vmin=0, vmax=1)

    ax.set_xlabel(r'$\beta$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\alpha$', fontsize=24, fontweight='bold')

    # =============== X轴范围 (0 到 1) ===============
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # =============== X 轴格式 (0 到 1, 间隔 0.1) ===============
    xticks = np.linspace(0, 1, 11)
    xticklabels = [f"{t:.1f}" for t in xticks]
    xticklabels[0] = "0"
    xticklabels[-1] = "1"
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # =============== Y 轴格式 (0 到 1, 间隔 0.1) ===============
    yticks = np.linspace(0, 1, 11)
    yticklabels = [f"{t:.1f}" for t in yticks]
    yticklabels[0] = "0"
    yticklabels[-1] = "1"
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # 刻度设置：向内(in)，并且四条边全部显示刻度
    ax.tick_params(axis='both', which='major', labelsize=16, direction='in',
                   top=True, right=True, bottom=True, left=True)

    # =============== Colorbar ===============
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, ticks=yticks)
    cbar.ax.set_yticklabels(yticklabels)
    cbar.ax.tick_params(labelsize=16, direction='in')

    # 【关键修改】将原本的 r'$\mathit{\mathbf{fc}}$' 改为 r'$fc$'
    # 这样就会调用 stix 字体库渲染出与论文图5完全一致的普通数学斜体 fc
    cbar.ax.set_title(r'$fc$', fontsize=24, pad=15)

    plt.tight_layout()
    plt.savefig('0.025heatmap_figure_5_positive.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图表已生成: 0.025heatmap_figure_5_positive.png")


if __name__ == "__main__":
    main()
