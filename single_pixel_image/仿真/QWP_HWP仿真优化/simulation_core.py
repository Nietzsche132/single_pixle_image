"""P1(-45) -> QWP(alpha) -> HWP(beta) -> PEM(0) -> P2(+45).

All internal voltages are in mV. S0 is normalized to 1 after P1.
Library used by qwp_hwp_gui.py; not a separate application.
Dependencies: numpy, scipy, matplotlib. See README.md for conventions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.special import jn_zeros, jv


BLUE, TEAL, ORANGE, GREY = "#205493", "#008C95", "#D47720", "#82909C"
SPECTRUM_MAX_KHZ = 250.0


def retarder(theta_deg: float, delay_rad: float) -> np.ndarray:
    """Mueller retarder: S3=2 Im(Ex Ey*), J(0)=diag(1, exp(i*delay))."""
    c, s = np.cos(np.deg2rad(2 * theta_deg)), np.sin(np.deg2rad(2 * theta_deg))
    C, D = np.cos(delay_rad), np.sin(delay_rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c*c+s*s*C, c*s*(1-C), -s*D],
        [0, c*s*(1-C), s*s+c*c*C, c*D],
        [0, s*D, -c*D, C],
    ])


def polarizer(theta_deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(2 * theta_deg)), np.sin(np.deg2rad(2 * theta_deg))
    return 0.5 * np.array([[1, c, s, 0], [c, c*c, c*s, 0],
                           [s, c*s, s*s, 0], [0, 0, 0, 0]])


def choose_angles(dc: float, vpp: float, delta0: float) -> tuple[float, float]:
    """Choose one physical state with s2=s3=-lambda; not a unique inversion.

    Over delta in [-delta0,delta0], h=cos(delta)+sin(delta).
    DC=C*(1-J0*lambda); Vpp=C*lambda*(max(h)-min(h)).
    Solve the ratio for lambda and realize the state using QWP and HWP.
    """
    candidates = [-delta0, delta0]
    for k in range(-int(delta0 / np.pi)-3, int(delta0 / np.pi)+4):
        d = np.pi / 4 + k * np.pi
        if -delta0 <= d <= delta0:
            candidates.append(d)
    h = np.cos(candidates) + np.sin(candidates)
    ratio = vpp / dc
    lam = ratio / (np.ptp(h) + ratio * jv(0, delta0))
    if not 0 < lam < 1 / np.sqrt(2):
        raise ValueError("Targets cannot give three nonzero Stokes components in the chosen s2=s3 family.")
    s1_abs = np.sqrt(1 - 2 * lam**2)
    alpha = 0.5 * np.arccos(lam)
    beta = (np.arctan2(lam, s1_abs) + 2 * alpha) / 4
    return float(np.rad2deg(alpha)), float(np.rad2deg(beta))


def jones_retarder(theta_deg: float, delta: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return r @ np.diag([1, np.exp(1j * delta)]) @ r.T


def configure_style() -> None:
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"], "font.size": 13, "axes.labelsize": 14,
        "axes.unicode_minus": False, "mathtext.fontset": "stix",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9DAAB5", "xtick.color": "#394957",
        "ytick.color": "#394957", "text.color": "#203547",
        "axes.labelcolor": "#203547", "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def figure(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.16, top=0.75)
    fig.text(0.11, 0.93, title, fontsize=23, weight="bold")
    fig.text(0.11, 0.865, subtitle, fontsize=12, color="#556776")
    ax.grid(axis="y", color="#E5EBF0", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=240)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def plot_results(out, t, voltage, freq, amps, rows, meta):
    configure_style()
    fp = meta["f_pem_hz"]
    mean, vpp = meta["dc_fft_mv"], meta["vpp_sampled_mv"]
    fig, ax = figure("探测器输出：原始时域波形", "完整 PEM 非线性调制模型  |  QWP → HWP → PEM → P2  |  理想无噪声仿真")
    keep = t <= 4 / fp
    ax.plot(t[keep] * 1e6, voltage[keep], color=BLUE, lw=2.0)
    ax.axhline(mean, color=ORANGE, linestyle="--", lw=1.4)
    ax.text(0.025, 0.93, f"直流均值 = {mean:.3f} mV", transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    x_arrow = 3.7 / fp * 1e6
    ax.annotate("", xy=(x_arrow, voltage.max()), xytext=(x_arrow, voltage.min()),
                arrowprops={"arrowstyle": "<->", "color": TEAL, "lw": 1.8})
    ax.text(x_arrow-1, mean+12, f"Vpp = {vpp:.2f} mV", ha="right", color=TEAL,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    ax.set(xlabel="时间 / μs", ylabel="探测器电压 / mV", xlim=(0, 4 / fp * 1e6),
           ylim=(0, max(210, voltage.max()*1.15)))
    fig.text(0.11, 0.045, f"α = {meta['alpha_deg']:.4f}°   β = {meta['beta_deg']:.4f}°   δ₀ = {meta['delta0_rad']:.6f} rad   f = {fp/1e3:g} kHz", fontsize=11)
    save(fig, out, "01_original_waveform")

    fig, ax = figure("FFT 单边幅度谱", "频率范围 0–250 kHz  |  DC 保留均值；非零频率按正弦峰值幅度标定")
    keep = freq <= SPECTRUM_MAX_KHZ * 1000
    ax.plot(freq[keep]/1e3, amps[keep], color="#B6C2CC", lw=0.8)
    for row in rows[:31]:
        n, a = row["harmonic"], row["fft_amplitude_mv"]
        if n * fp > SPECTRUM_MAX_KHZ * 1000:
            continue
        color = [BLUE, TEAL, ORANGE][n] if n < 3 else GREY
        ax.vlines(n*fp/1e3, 0, a, color=color, linewidth=2)
        ax.scatter(n*fp/1e3, a, color=color, s=40, zorder=3)
        label = "DC" if n == 0 else f"{n}f"
        if n <= 4:
            ax.annotate(label, (n*fp/1e3, a), xytext=(0, 10),
                        textcoords="offset points", ha="left" if n == 0 else "center", fontsize=11, color=color)
    ax.set(xlabel="频率 / kHz", ylabel="单边幅度 / mV（DC：均值；AC：峰值）",
           xlim=(0, SPECTRUM_MAX_KHZ), ylim=(0, max(130, amps[keep].max()*1.3)))
    ax.set_xticks(np.arange(0, SPECTRUM_MAX_KHZ + 1, 50))
    fig.text(0.11, 0.045, f"整数 {meta['cycles']} 个周期采样；不去均值、不加额外窗；频率分辨率 Δf = {meta['frequency_resolution_hz']:.1f} Hz。", fontsize=11)
    save(fig, out, "02_full_fft_spectrum")

    titles = ["直流分量 DC", "一倍频分量 1f", "二倍频分量 2f"]
    names = ["03_dc_component", "04_1f_component", "05_2f_component"]
    for n, (title, name, color) in enumerate(zip(titles, names, [BLUE, TEAL, ORANGE])):
        row = rows[n]
        a = row["fft_amplitude_mv"]
        subtitle = (f"频率 = {n*fp/1e3:g} kHz  |  FFT 均值 = {a:.3f} mV" if n == 0 else
                    f"频率 = {n*fp/1e3:g} kHz  |  FFT 峰值幅度 = {a:.3f} mV  |  RMS = {row['rms_mv']:.3f} mV")
        fig, ax = figure(title, subtitle)
        low, high = (0, 0.2*fp) if n == 0 else ((n-0.1)*fp, (n+0.1)*fp)
        keep = (freq >= low) & (freq <= high)
        ax.vlines(freq[keep]/1e3, 0, amps[keep], color=color, lw=2)
        ax.scatter([n*fp/1e3], [a], color=color, s=65, zorder=3)
        ax.annotate(f"{a:.3f} mV", (n*fp/1e3, a), xytext=(20, 12),
                    textcoords="offset points", color=color, fontsize=17, weight="bold")
        ax.set(xlabel="频率 / kHz", ylabel="直流电压 / mV" if n == 0 else "单边峰值幅度 / mV",
               xlim=((low-0.03*fp)/1e3, high/1e3), ylim=(0, max(125, a*1.3)))
        formula = [r"$V_{DC}=C[1+J_0(\delta_0)s_2]$",
                   r"$A_{1f}=|2C J_1(\delta_0)s_3|$",
                   r"$A_{2f}=|2C J_2(\delta_0)s_2|$"][n]
        ax.text(0.36, 0.84, formula, transform=ax.transAxes, fontsize=20)
        ax.text(0.36, 0.70, f"贝塞尔理论值 = {row['theory_amplitude_mv']:.6f} mV", transform=ax.transAxes, fontsize=12)
        footer = "DC 是时间平均值，FFT 的零频点不乘 2，也不除以 √2。" if n == 0 else f"带符号的{'正弦' if n == 1 else '余弦'}系数 = {row['signed_theory_mv']:+.6f} mV；此图显示幅值，符号保存在数据表中。"
        fig.text(0.11, 0.045, footer, fontsize=11)
        save(fig, out, name)

