from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv


# ============================================================
# 1. 输出路径
# ============================================================

OUTPUT_PATH = Path(r"E:\single_pixel\仿真\bessel_functions_J0_J1_J2.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 横坐标：PEM峰值相位延迟 delta_0
# ============================================================

delta = np.linspace(0, 8, 2000)

J0 = jv(0, delta)
J1 = jv(1, delta)
J2 = jv(2, delta)


# 两个需要重点比较的工作点
delta_code = 2.4048       # 当前仿真代码使用的J0零点附近
delta_half_wave = np.pi   # 实机界面0.5000 lambda对应的半波峰值延迟


# ============================================================
# 3. 绘图
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(delta, J0, linewidth=2.2, label=r"$J_0(\delta_0)$")
ax.plot(delta, J1, linewidth=2.2, label=r"$J_1(\delta_0)$")
ax.plot(delta, J2, linewidth=2.2, label=r"$J_2(\delta_0)$")

# 零值参考线
ax.axhline(0, color="black", linewidth=1, alpha=0.6)

# 标出仿真工作点
ax.axvline(
    delta_code,
    color="red",
    linestyle="--",
    linewidth=1.8,
    label=r"仿真工作点：$\delta_0=2.4048$ rad"
)

# 标出实机半波工作点
ax.axvline(
    delta_half_wave,
    color="purple",
    linestyle="-.",
    linewidth=1.8,
    label=r"实机半波工作点：$\delta_0=\pi$ rad"
)

# 在两个工作点上标出J0、J1、J2数值
colors = ["C0", "C1", "C2"]

for order, color in zip([0, 1, 2], colors):
    value_code = jv(order, delta_code)
    value_half = jv(order, delta_half_wave)

    ax.scatter(
        delta_code,
        value_code,
        color=color,
        edgecolor="black",
        s=55,
        zorder=5
    )

    ax.scatter(
        delta_half_wave,
        value_half,
        color=color,
        marker="s",
        edgecolor="black",
        s=55,
        zorder=5
    )

# 坐标轴和标题
ax.set_xlabel(r"PEM峰值相位延迟 $\delta_0$ / rad", fontsize=13)
ax.set_ylabel(r"贝塞尔函数值 $J_n(\delta_0)$", fontsize=13)
ax.set_title(
    "PEM调制深度对应的第一类贝塞尔函数",
    fontsize=16
)

ax.set_xlim(0, 8)
ax.set_ylim(-0.55, 1.05)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(fontsize=10, loc="best")


# ============================================================
# 4. 添加上方横坐标：以波长lambda为单位的延迟量
#
# delta_0 = 2*pi*r
# r = delta_0/(2*pi)
# ============================================================

def rad_to_wave(x):
    return x / (2 * np.pi)


def wave_to_rad(x):
    return x * 2 * np.pi


secondary_axis = ax.secondary_xaxis(
    "top",
    functions=(rad_to_wave, wave_to_rad)
)

secondary_axis.set_xlabel(
    r"峰值延迟量 $\delta_0/(2\pi)$ / $\lambda$",
    fontsize=12
)


# ============================================================
# 5. 输出两个工作点的具体数值
# ============================================================

print("当前仿真工作点：")
print(f"delta_0 = {delta_code:.6f} rad")
print(f"延迟波数 = {delta_code / (2*np.pi):.6f} lambda")
print(f"J0 = {jv(0, delta_code):.8f}")
print(f"J1 = {jv(1, delta_code):.8f}")
print(f"J2 = {jv(2, delta_code):.8f}")

print("\n实机半波工作点：")
print(f"delta_0 = pi = {delta_half_wave:.6f} rad")
print("延迟波数 = 0.500000 lambda")
print(f"J0 = {jv(0, delta_half_wave):.8f}")
print(f"J1 = {jv(1, delta_half_wave):.8f}")
print(f"J2 = {jv(2, delta_half_wave):.8f}")


# ============================================================
# 6. 保存和显示
# ============================================================

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

print(f"\n图像已保存至：{OUTPUT_PATH}")

plt.show()