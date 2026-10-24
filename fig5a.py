import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time

# ================= 1. 参数设置 =================
L = 100  # 网络尺寸 L x L
N = L * L  # 总人口
K = 0.1  # 噪声/理性参数
beta = 0.50  # 流行度强度指数 (对应图2b)
MCS = 10000  # 总蒙特卡洛步数
eq_MCS = 5000  # 取最后3000步取平均
runs = 10  # 独立运行次数 (原论文为100次，为节约时间设为10次，可根据算力修改)

alphas = [0.60, 0.65, 0.80, 1.00]  # 局部流行度的权重系数
b_values= np.linspace(1, 1.8, 33)  # b 区间 [1.0, 1.8]


# ================= 2. 核心蒙特卡洛模拟函数 =================
@njit
def simulate_mc(L, b, alpha, beta, K, MCS, eq_MCS):
    N = L * L
    # 初始化网络，策略：1 代表合作(C)，0 代表背叛(D)。初始各占50%
    grid = np.random.randint(0, 2, size=(L, L))
    fc_history = np.zeros(MCS)

    for step in range(MCS):
        # 每一个MCS只更新一次全局流行度
        fc = np.sum(grid) / N
        fd = 1.0 - fc

        # 一个MCS包含 N 次子演化步骤（异步更新）
        for _ in range(N):
            # 随机选择焦点玩家 x
            rx = np.random.randint(L)
            ry = np.random.randint(L)
            sx = grid[rx, ry]

            # 计算 x 的邻居合作者数量 (周期性边界条件)
            nx_c = grid[(rx - 1) % L, ry] + grid[(rx + 1) % L, ry] + grid[rx, (ry - 1) % L] + grid[rx, (ry + 1) % L]

            # 计算 x 的收益和流行度 (对应公式 4 和 5)
            if sx == 1:
                px = nx_c * 1.0  # 合作者收益
                pop_loc_x = (nx_c + 1) / 5.0
                pop_glob_x = fc
            else:
                px = nx_c * b  # 背叛者收益
                nx_d = 4 - nx_c
                pop_loc_x = (nx_d + 1) / 5.0
                pop_glob_x = fd

            # 适应度公式
            fit_x = ((alpha * pop_loc_x + (1.0 - alpha) * pop_glob_x) ** beta) * px

            # 随机选择一个邻居 y
            rand_n = np.random.randint(4)
            if rand_n == 0:
                ry_x, ry_y = (rx - 1) % L, ry
            elif rand_n == 1:
                ry_x, ry_y = (rx + 1) % L, ry
            elif rand_n == 2:
                ry_x, ry_y = rx, (ry - 1) % L
            else:
                ry_x, ry_y = rx, (ry + 1) % L

            sy = grid[ry_x, ry_y]

            # 邻居 y 的参数计算
            ny_c = grid[(ry_x - 1) % L, ry_y] + grid[(ry_x + 1) % L, ry_y] + grid[ry_x, (ry_y - 1) % L] + grid[
                ry_x, (ry_y + 1) % L]

            if sy == 1:
                py = ny_c * 1.0
                pop_loc_y = (ny_c + 1) / 5.0
                pop_glob_y = fc
            else:
                py = ny_c * b
                ny_d = 4 - ny_c
                pop_loc_y = (ny_d + 1) / 5.0
                pop_glob_y = fd

            fit_y = ((alpha * pop_loc_y + (1.0 - alpha) * pop_glob_y) ** beta) * py

            # Fermi 规则更新策略 (公式 6)
            prob = 1.0 / (1.0 + np.exp((fit_x - fit_y) / K))
            if np.random.random() < prob:
                grid[rx, ry] = sy

        # 记录本MCS稳态时的合作者比例
        fc_history[step] = np.sum(grid) / N

    # 返回最后 3000 步的平均合作者比例
    return np.mean(fc_history[-eq_MCS:])


# 并行运行实验的主函数
@njit(parallel=True)
def run_experiment(L, b_values, alphas, beta, K, MCS, eq_MCS, runs):
    results = np.zeros((len(alphas), len(b_values)))
    for i in prange(len(alphas)):
        alpha = alphas[i]
        for j in prange(len(b_values)):
            b = b_values[j]
            run_avg = 0.0
            for r in range(runs):
                run_avg += simulate_mc(L, b, alpha, beta, K, MCS, eq_MCS)
            results[i, j] = run_avg / runs
    return results


if __name__ == "__main__":
    print("开始并行仿真，请耐心等待...")
    start_time = time.time()

    # 运行模拟
    results = run_experiment(L, b_values, alphas, beta, K, MCS, eq_MCS, runs)

    print(f"仿真完成！耗时: {time.time() - start_time:.2f} 秒")

    # ================= 3. 保存 Text 格式数据 =================
    # 构造保存数据的矩阵
    save_data = np.column_stack((b_values, results.T))
    header = "b\talpha=0.60\talpha=0.65\talpha=0.80\talpha=1.00"
    np.savetxt("fig2b_data.txt", save_data, delimiter='\t', header=header, fmt='%.4f', comments='')
    print("数据已成功保存至 'fig2b_data.txt' 文件中。")

    # ================= 4. 绘图 (参考己图格式) =================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    colors = ['#e4d00a', '#00ced1', '#0000CD', '#DC143C']
    markers = ['+', '+', 'o', 'o']

    # 生成单张图，尺寸与己图单图一致
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    for i, alpha in enumerate(alphas):
        # 判断蓝色为空心，红色为实心
        mfc = 'none' if colors[i] == '#0000CD' else colors[i]

        # 绘图：图例中的 alpha 保留 '.2f' 格式
        ax.plot(b_values, results[i],
                label=rf'$\alpha={alpha:.2f}$',
                color=colors[i],
                marker=markers[i], markersize=6, linewidth=1.2,
                markeredgecolor=colors[i], markerfacecolor=mfc)

    # ----------------------------------------------------
    # 精准控制坐标轴范围和强制自定义字符串刻度
    # ----------------------------------------------------

    # Y 轴严格控制 0 到 1
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # 替换刻度标签：0.0 改为 '0', 1.0 改为 '1'
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    # X 轴根据 B_VALUES 严格控制 1.0 到 1.8
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

    # 右下角的 Beta 标注 (保留 '.2f' 格式)
    ax.text(0.96, 0.05, rf"$\beta = {beta:.2f}$", transform=ax.transAxes,
            fontsize=13, ha='right', va='bottom')

    # 图例统一放置在右上角
    ax.legend(fontsize=10, frameon=True, edgecolor='black', loc='upper right', handlelength=1.5)

    plt.tight_layout()
    plt.savefig("fig2b_reproduction_formatted.png", dpi=300, bbox_inches='tight')
    plt.show()
