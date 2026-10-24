import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from numba import njit
import os


# ==========================================
# 核心演化动力学 (使用 Numba 加速)
# ==========================================

@njit
def get_neighbors(x, y, L):
    """获取冯·诺伊曼邻居的坐标 (周期性边界条件)"""
    return [
        ((x - 1) % L, y),
        ((x + 1) % L, y),
        (x, (y - 1) % L),
        (x, (y + 1) % L)
    ]


@njit
def calc_payoff(S, cx, cy, b, L):
    """计算个体当前的物质收益"""
    my_s = S[cx, cy]
    payoff = 0.0
    for nx, ny in get_neighbors(cx, cy, L):
        neighbor_s = S[nx, ny]
        if my_s == 1 and neighbor_s == 1:
            payoff += 1.0
        elif my_s == 0 and neighbor_s == 1:
            payoff += b
    return payoff


@njit
def calc_fitness(S, R, cx, cy, b, L, α, β, epsilon, sum_R_eps, sum_RC_eps):
    """计算个体适应度：收益与综合流行度的耦合"""
    my_s = S[cx, cy]

    local_sum_all = R[cx, cy] + epsilon
    local_sum_same = R[cx, cy] + epsilon

    for nx, ny in get_neighbors(cx, cy, L):
        val = R[nx, ny] + epsilon
        local_sum_all += val
        if S[nx, ny] == my_s:
            local_sum_same += val

    lpop = local_sum_same / local_sum_all if local_sum_all > 0 else 0.0

    if my_s == 1:
        gpop = sum_RC_eps / sum_R_eps if sum_R_eps > 0 else 0.0
    else:
        sum_RD_eps = sum_R_eps - sum_RC_eps
        gpop = sum_RD_eps / sum_R_eps if sum_R_eps > 0 else 0.0

    tpop = α * lpop + (1.0 - α) * gpop
    payoff = calc_payoff(S, cx, cy, b, L)

    return payoff * (tpop ** β)


@njit
def run_mcs(S, R, b, L, α, β, epsilon, γ, u, K):
    """执行一个完整的蒙特卡洛步 (MCS)"""
    N = L * L
    sum_R_eps = np.sum(R + epsilon)
    sum_RC_eps = np.sum((R + epsilon) * S)

    for _ in range(N):
        idx = np.random.randint(N)
        cx = idx // L
        cy = idx % L

        if np.random.rand() < u:
            new_s = np.random.randint(2)
            if new_s != S[cx, cy]:
                diff = R[cx, cy] + epsilon
                sum_RC_eps += diff if new_s == 1 else -diff
            S[cx, cy] = new_s
            continue

        neighbors = get_neighbors(cx, cy, L)
        nx, ny = neighbors[np.random.randint(4)]

        Fx = calc_fitness(S, R, cx, cy, b, L, α, β, epsilon, sum_R_eps, sum_RC_eps)
        Fy = calc_fitness(S, R, nx, ny, b, L, α, β, epsilon, sum_R_eps, sum_RC_eps)

        diff_F = (Fx - Fy) / K
        if diff_F > 100:
            diff_F = 100
        elif diff_F < -100:
            diff_F = -100

        prob = 1.0 / (1.0 + np.exp(diff_F))

        if np.random.rand() < prob:
            new_s = S[nx, ny]
            if new_s != S[cx, cy]:
                diff = R[cx, cy] + epsilon
                sum_RC_eps += diff if new_s == 1 else -diff
            S[cx, cy] = new_s

    for i in range(L):
        for j in range(L):
            R[i, j] = (1.0 - γ) * R[i, j] + γ * S[i, j]

    return S, R


# ==========================================
# 模拟与双轨快照图绘制主程序
# ==========================================

def simulate_and_plot():
    # --- 系统的固定环境参数 ---
    L = 100  # 晶格大小
    b = 1.11  # 背叛诱惑
    epsilon = 0.01  # 正则化常数
    K = 0.1  # 噪声强度/选择压力

    # 记录快照的时间步 (参考图7)
    snapshots_mcs = [1, 10, 100, 1000, 20000]

    # --- 核心机制参数组 (α, β, γ, u) ---
    param_sets = [
        (1.0, 0.3, 0.1, 0.05),  # 第1组 -> 对应 A(策略) 和 B(声誉)
        (0.5, 0.5, 0.1, 0.05),  # 第2组 -> 对应 C(策略) 和 D(声誉)
        (0.6, 0.5, 0.1, 0.05)   # 第3组 -> 对应 E(策略) 和 F(声誉)
    ]

    num_sets = len(param_sets)
    num_snaps = len(snapshots_mcs)

    # 自动创建保存数据的文件夹
    save_dir = 'simulation_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 动态创建画布：压低总高度以配合紧凑型排版
    fig_width = 2 * num_snaps
    fig_height = 2.1 * num_sets * 2
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 外层 GridSpec：控制大组 (AB, CD, EF) 之间的距离。
    outer_gs = gridspec.GridSpec(num_sets, 1, figure=fig, hspace=0.1)

    # 定义策略图的颜色映射：背叛者 D(0)=蓝色, 合作者 C(1)=红色
    cmap_strategy = ListedColormap(['blue', 'red'])

    for set_idx, (α, β, γ, u) in enumerate(param_sets):
        print(f"\n开始演化模拟: 第 {set_idx + 1} 组参数 (α={α}, β={β}, γ={γ})...")

        S = np.random.randint(2, size=(L, L)).astype(np.float64)
        R = np.full((L, L), 0.5, dtype=np.float64)

        strategy_history = []
        reputation_history = []

        max_mcs = max(snapshots_mcs)
        for mcs in range(1, max_mcs + 1):
            S, R = run_mcs(S, R, b, L, α, β, epsilon, γ, u, K)
            if mcs in snapshots_mcs:
                print(f" 记录 MCS = {mcs} 的快照")
                strategy_history.append(S.copy())
                reputation_history.append(R.copy())

        # 保存数据
        for i, mcs in enumerate(snapshots_mcs):
            strat_filename = os.path.join(save_dir, f'Strategy_Set{set_idx+1}_MCS{mcs}.csv')
            rep_filename = os.path.join(save_dir, f'Reputation_Set{set_idx+1}_MCS{mcs}.csv')
            np.savetxt(strat_filename, strategy_history[i], delimiter=',', fmt='%d')
            np.savetxt(rep_filename, reputation_history[i], delimiter=',', fmt='%.4f')

        # 内层 GridSpec
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, num_snaps, subplot_spec=outer_gs[set_idx], hspace=0.28,
                                                    wspace=0.01)

        # 绘制虚线括号
        bg_ax = fig.add_subplot(inner_gs[:, :])
        bg_ax.axis('off')
        bracket_x = [-0.01, -0.03, -0.03, -0.01]
        bracket_y = [1.0, 1.0, 0.0, 0.0]
        bg_ax.plot(bracket_x, bracket_y, transform=bg_ax.transAxes, color='black', linestyle='--', linewidth=1.5,
                   clip_on=False)

        row_strat = set_idx * 2
        row_rep = set_idx * 2 + 1
        letter_strat = chr(65 + row_strat)
        letter_rep = chr(65 + row_rep)

        for i, mcs in enumerate(snapshots_mcs):
            # 1. 绘制策略分布图
            ax_strat = fig.add_subplot(inner_gs[0, i])
            ax_strat.imshow(strategy_history[i], cmap=cmap_strategy, vmin=0, vmax=1, interpolation='nearest')
            ax_strat.set_xticks([])
            ax_strat.set_yticks([])
            ax_strat.text(0.03, 0.97, f'({letter_strat}-{i + 1})', color='white',
                          transform=ax_strat.transAxes, ha='left', va='top', fontweight='bold', fontsize=12)

            if i == 0:
                ax_strat.text(0, 1.05, f'({letter_strat}) Strategy Distribution',
                              transform=ax_strat.transAxes, ha='left', va='bottom', fontsize=12, fontweight='bold')

            # 2. 绘制声誉分布图
            ax_rep = fig.add_subplot(inner_gs[1, i])
            im_rep = ax_rep.imshow(reputation_history[i], cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
            ax_rep.set_xticks([])
            ax_rep.set_yticks([])
            ax_rep.text(0.03, 0.97, f'({letter_rep}-{i + 1})', color='white',
                        bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1),
                        transform=ax_rep.transAxes, ha='left', va='top', fontweight='bold', fontsize=12)

            if i == 0:
                ax_rep.text(0, 1.05, f'({letter_rep}) Reputation Distribution',
                            transform=ax_rep.transAxes, ha='left', va='bottom', fontsize=12, fontweight='bold')

            # ==========================================
            # 核心修改：仅展示 α, β, γ，位置右移 (x=0.9)，字号放大 (15)
            # ==========================================
            if i == num_snaps // 2:
                ax_rep.text(0.5, 1.08,
                            f'$\\alpha={α:.2f} \\quad \\beta={β:.2f} \\quad \\gamma={γ:.2f}$',
                            transform=ax_rep.transAxes, ha='center', va='bottom', fontsize=15)

            # Colorbar
            if i == num_snaps - 1:
                divider = make_axes_locatable(ax_rep)
                cax = divider.append_axes("right", size="5%", pad=0.08)
                cbar = fig.colorbar(im_rep, cax=cax)
                cbar.set_ticks([0, 0.5, 1.0])
                cbar.ax.tick_params(labelsize=10)

            # MCS 标签
            if set_idx == num_sets - 1:
                ax_rep.text(0.5, -0.12, f'MCS={mcs}', transform=ax_rep.transAxes, ha='center', va='top', fontsize=12)

    plt.subplots_adjust(left=0.06, right=0.92, bottom=0.05, top=0.96)
    # 仅修改此处：后缀改为 .eps，并增加 format='eps'
    plt.savefig('0.3Dual_Track_Snapshot.eps', dpi=300, format='eps', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    simulate_and_plot()
