"""Editable QWP/HWP controls and five simultaneous live Matplotlib charts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.special import jv

from qwp_hwp_model import Parameters, simulate, export_result
from simulation_core import configure_style, BLUE, TEAL, ORANGE, GREY, SPECTRUM_MAX_KHZ

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "输出结果"
CONFIG = BASE / "上次参数.json"
BG = "#F2F5F9"


class SimulatorApp:
    def __init__(self, root: tk.Tk, initial: Parameters | None = None):
        self.root = root
        self.pending = None
        self.suspended = False
        self.result = None
        self.update_count = 0
        self.last_export = None
        self.root.title("QWP / HWP 偏振仿真 · 实时波形与频谱")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        width, height = min(1720,sw-60), min(970,sh-90)
        root.geometry(f"{width}x{height}+{max(0,(sw-width)//2)}+{max(0,(sh-height)//2-20)}")
        root.minsize(1180,760)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.report_callback_exception = self.callback_error
        style=ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame",background=BG)
        style.configure("TLabel",background=BG,font=("Microsoft YaHei UI",10),foreground="#283F53")
        style.configure("Title.TLabel",font=("Microsoft YaHei UI",19,"bold"))
        style.configure("Section.TLabel",font=("Microsoft YaHei UI",12,"bold"))
        style.configure("Hint.TLabel",font=("Microsoft YaHei UI",9),foreground="#65778A")
        style.configure("TButton",font=("Microsoft YaHei UI",10),padding=(8,7))
        style.configure("Accent.TButton",background=BLUE,foreground="white")
        style.map("Accent.TButton",background=[("active","#286DAA")])
        style.configure("TSpinbox",padding=4,font=("Microsoft YaHei UI",11))
        style.configure("TEntry",padding=4,font=("Microsoft YaHei UI",10))

        header=ttk.Frame(root,padding=(22,14,22,8)); header.pack(fill="x")
        ttk.Label(header,text="QWP / HWP 偏振仿真",style="Title.TLabel").pack(side="left")
        ttk.Label(header,text="P1 (−45°)  →  QWP (α)  →  HWP (β)  →  PEM (0°)  →  P2 (+45°)",
                  style="Hint.TLabel").pack(side="right",padx=10)
        body=ttk.Frame(root,padding=(18,4,12,0)); body.pack(fill="both",expand=True)
        controls=ttk.Frame(body,width=290,padding=(5,4,18,4)); controls.pack(side="left",fill="y")
        controls.pack_propagate(False)
        right=ttk.Frame(body); right.pack(side="left",fill="both",expand=True)
        p = initial or self.load_config()
        self.vars={k:tk.StringVar(value=f"{v:.12g}") for k,v in asdict(p).items()}
        self.angle_entries={}
        self.slider_vars={}
        ttk.Label(controls,text="波片角度",style="Section.TLabel").pack(anchor="w",pady=(0,8))
        for key,label in [("alpha_deg","α · 四分之一波片 QWP"),("beta_deg","β · 半波片 HWP")]:
            ttk.Label(controls,text=label).pack(anchor="w")
            row=ttk.Frame(controls); row.pack(fill="x",pady=(5,2))
            entry=ttk.Spinbox(row,textvariable=self.vars[key],from_=-180,to=180,increment=0.1,width=16)
            entry.pack(side="left",fill="x",expand=True)
            entry.bind("<Return>",lambda event:self.update_now())
            ttk.Label(row,text=" °").pack(side="left")
            self.angle_entries[key]=entry
            variable=tk.DoubleVar(value=getattr(p,key)); self.slider_vars[key]=variable
            scale=ttk.Scale(controls,from_=-90,to=90,variable=variable,
                            command=lambda value,k=key:self.slider_changed(k,value))
            scale.pack(fill="x",pady=(2,1))
            ttk.Label(controls,text="滑块 −90°～90°；输入支持 ±180°",style="Hint.TLabel").pack(anchor="w",pady=(0,10))
        ttk.Label(controls,text="数值修改后自动刷新，无需确认。",style="Hint.TLabel").pack(anchor="w")
        ttk.Button(controls,text="恢复初始参数",command=self.reset).pack(fill="x",pady=(8,10))
        ttk.Separator(controls).pack(fill="x",pady=(0,10))
        ttk.Label(controls,text="PEM 与电压尺度",style="Section.TLabel").pack(anchor="w",pady=(0,6))
        for key,label in [("f_pem_khz","频率 f / kHz"),("delta0_rad","延迟 δ0 / rad"),("c_mv","C = GI0/2 / mV")]:
            row=ttk.Frame(controls); row.pack(fill="x",pady=3)
            ttk.Label(row,text=label).pack(side="left")
            entry=ttk.Entry(row,textvariable=self.vars[key],width=13)
            entry.pack(side="right")
            entry.bind("<Return>",lambda event:self.update_now())
        ttk.Label(controls,text="固定 C；波形峰峰值随偏振态改变。",style="Hint.TLabel").pack(anchor="w",pady=(4,10))
        ttk.Separator(controls).pack(fill="x",pady=(0,10))
        ttk.Label(controls,text="当前结果",style="Section.TLabel").pack(anchor="w")
        self.readout=tk.StringVar()
        ttk.Label(controls,textvariable=self.readout,justify="left",font=("Consolas",10),
                  wraplength=265).pack(anchor="w",pady=(6,5))
        self.note=tk.StringVar()
        ttk.Label(controls,textvariable=self.note,wraplength=258,style="Hint.TLabel").pack(anchor="w",pady=(0,9))
        self.save_button=ttk.Button(controls,text="保存当前结果 · 五张图片 + 数据",style="Accent.TButton",command=self.save)
        self.save_button.pack(fill="x",pady=(3,4))
        ttk.Button(controls,text="打开结果文件夹",command=self.open_results).pack(fill="x",pady=3)
        ttk.Label(controls,text="每次保存生成独立文件夹，便于比较。",style="Hint.TLabel").pack(anchor="w",pady=(4,0))

        configure_style()
        self.figure=Figure(figsize=(12.7,8.0),dpi=100,facecolor="white",layout="constrained")
        grid=self.figure.add_gridspec(2,6,height_ratios=[1.05,1],hspace=0.10,wspace=0.08)
        self.axes=[self.figure.add_subplot(grid[0,:3]), self.figure.add_subplot(grid[0,3:]),
                   self.figure.add_subplot(grid[1,:2]), self.figure.add_subplot(grid[1,2:4]),
                   self.figure.add_subplot(grid[1,4:])]
        self.canvas=FigureCanvasTkAgg(self.figure,master=right)
        self.canvas.get_tk_widget().pack(fill="both",expand=True)
        ttk.Label(right,text="单边谱：DC 为均值，交流为峰值；RMS = 峰值 / √2。总频谱仅显示 0–250 kHz。",
                  style="Hint.TLabel").pack(anchor="w",pady=(5,0))
        self.status=tk.StringVar(value="准备就绪")
        ttk.Label(root,textvariable=self.status,style="Hint.TLabel",padding=(23,7)).pack(fill="x")
        for variable in self.vars.values():
            variable.trace_add("write",self.schedule_update)
        self.update_now()

    def load_config(self):
        try:
            p=Parameters(**json.loads(CONFIG.read_text(encoding="utf-8")))
            p.validate()
            return p
        except (OSError,ValueError,TypeError):
            return Parameters()

    def slider_changed(self,key,value):
        if not self.suspended:
            self.vars[key].set(f"{float(value):.4f}")

    def schedule_update(self,*unused):
        if self.suspended:
            return
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending=self.root.after(220,self.update_now)

    def read_parameters(self):
        try:
            p=Parameters(**{k:float(v.get()) for k,v in self.vars.items()})
        except ValueError:
            raise ValueError("正在编辑：请为所有参数输入有效数字。") from None
        p.validate()
        return p

    def update_now(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending=None
        try:
            p=self.read_parameters()
        except ValueError as error:
            self.status.set(str(error)+" 当前仍显示上一次有效结果。")
            self.save_button.state(["disabled"])
            return False
        self.suspended=True
        for key,var in self.slider_vars.items():
            var.set(np.clip(getattr(p,key),-90,90))
        self.suspended=False
        self.result=simulate(p)
        self.draw_charts()
        self.update_count+=1
        s=self.result.stokes
        rows=self.result.harmonics
        m=self.result.metadata
        self.readout.set(f"s1  {s[1]:+9.5f}\ns2  {s[2]:+9.5f}\ns3  {s[3]:+9.5f}\n\nDC  {m['dc_fft_mv']:9.3f} mV\nVpp {m['vpp_mv']:9.3f} mV\n1f  {rows[1]['fft_amplitude_mv']:9.3f} mV 峰值\n2f  {rows[2]['fft_amplitude_mv']:9.3f} mV 峰值")
        self.note.set("J0(δ0) ≈ 0：DC 随角度保持约 C；1f 和 2f 随偏振态改变。" if abs(jv(0,p.delta0_rad))<1e-5
                      else "J0(δ0) 非零：DC 也会随 s2 改变。")
        self.save_button.state(["!disabled"])
        self.status.set(f"已更新  |  α={p.alpha_deg:.4f}°  β={p.beta_deg:.4f}°  |  数据保存到：{OUTPUT}")
        return True

    def draw_charts(self):
        r=self.result; p=r.parameters; m=r.metadata
        fp=p.f_pem_khz
        upper=max(1,2*p.c_mv*1.06)
        for ax in self.axes:
            ax.clear()
            ax.tick_params(labelsize=9)
            ax.grid(axis="y",color="#E5EBF0",lw=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
        ax=self.axes[0]
        keep=r.time_s<=4/(fp*1000)
        ax.plot(r.time_s[keep]*1e6,r.voltage_mv[keep],color=BLUE,lw=1.7)
        ax.axhline(m['dc_fft_mv'],color=ORANGE,lw=1.1,ls="--")
        ax.set_title("原始信号",fontsize=13,weight="bold",loc="left",pad=34)
        ax.text(0,1.04,f"DC {m['dc_fft_mv']:.3f} mV   |   Vpp {m['vpp_mv']:.3f} mV",transform=ax.transAxes,fontsize=10)
        ax.set(xlim=(0,4/(fp*1000)*1e6),ylim=(0,upper))
        ax.set_xlabel("时间 / μs",fontsize=10); ax.set_ylabel("电压 / mV",fontsize=10)
        ax=self.axes[1]
        keep=r.frequency_hz<=SPECTRUM_MAX_KHZ*1000
        frequencies=r.frequency_hz[keep]/1000
        amplitudes=r.fft_amplitude_mv[keep]
        # Plot ALL FFT bins in the displayed band, not only expected harmonic locations.
        ax.plot(frequencies,amplitudes,color=GREY,lw=0.8)
        for n,color in enumerate([BLUE,TEAL,ORANGE]):
            if n*fp > SPECTRUM_MAX_KHZ:
                continue
            a=r.harmonics[n]['fft_amplitude_mv']
            ax.vlines(n*fp,0,a,color=color,lw=2)
            ax.scatter([n*fp],[a],s=24,color=color,zorder=3)
        ax.set_title("所有信号的频谱",fontsize=13,weight="bold",loc="left",pad=34)
        ax.text(0,1.04,"DC、1f、2f 及高次谐波 · 0–250 kHz",transform=ax.transAxes,fontsize=10)
        ax.set(xlim=(0,SPECTRUM_MAX_KHZ),ylim=(0,max(1,amplitudes.max()*1.18)))
        ax.set_xticks(np.arange(0,SPECTRUM_MAX_KHZ+1,50))
        ax.set_xlabel("频率 / kHz",fontsize=10); ax.set_ylabel("幅度 / mV",fontsize=10)
        for n,ax,color in zip(range(3),self.axes[2:],[BLUE,TEAL,ORANGE]):
            a=r.harmonics[n]['fft_amplitude_mv']
            rms=r.harmonics[n]['rms_mv']
            low,high=(0,fp*0.2) if n==0 else ((n-0.12)*fp,(n+0.12)*fp)
            keep=(r.frequency_hz>=low*1000)&(r.frequency_hz<=high*1000)
            display=np.where(r.fft_amplitude_mv[keep]<1e-10,0,r.fft_amplitude_mv[keep])
            ax.vlines(r.frequency_hz[keep]/1000,0,display,color=color,lw=1.8)
            ax.scatter([n*fp],[a if a>1e-10 else 0],s=33,color=color,zorder=3)
            title=["直流分量 DC","一倍频分量 1f","二倍频分量 2f"][n]
            ax.set_title(title,fontsize=12,weight="bold",loc="left",pad=47)
            ax.text(0,1.10,f"{a:.3f} mV"+(" 均值" if n==0 else " 峰值"),transform=ax.transAxes,fontsize=12,color=color,weight="bold")
            ax.text(0,1.025,f"频率 {n*fp:g} kHz"+("" if n==0 else f"  |  RMS {rms:.3f} mV"),transform=ax.transAxes,fontsize=8.5)
            ax.set(xlim=(low-0.015*fp,high),ylim=(0,max(1,1.25*p.c_mv,1.2*a)))
            ax.set_xlabel("频率 / kHz",fontsize=10)
            ax.set_ylabel("均值 / mV" if n==0 else "峰值 / mV",fontsize=10)
        self.canvas.draw_idle()

    def set_parameters(self,p):
        self.suspended=True
        for key,value in asdict(p).items():
            self.vars[key].set(f"{value:.12g}")
        self.suspended=False
        self.update_now()

    def reset(self):
        self.set_parameters(Parameters())

    def save(self):
        if not self.update_now():
            return
        self.save_button.state(["disabled"])
        self.status.set("正在保存五张独立图片、界面图表总览和数值数据…")
        self.root.update_idletasks()
        self.root.after(20,self.finish_save)

    def finish_save(self):
        try:
            # Read once again so the exported data always agrees with valid inputs.
            if not self.update_now():
                return
            p=self.result.parameters
            folder=OUTPUT/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_alpha{p.alpha_deg:.4f}_beta{p.beta_deg:.4f}")
            export_result(self.result,folder)
            self.canvas.draw()
            self.figure.savefig(folder/"00_五图总览.png",dpi=180)
            CONFIG.write_text(json.dumps(asdict(p),ensure_ascii=False,indent=2),encoding="utf-8")
            self.last_export=folder
            self.status.set(f"已保存：{folder.name}（输出结果文件夹内）")
        except Exception as error:
            logging.exception("Export failed")
            messagebox.showerror("保存失败",str(error),parent=self.root)
            self.status.set("保存失败；可查看运行日志。")
        finally:
            self.save_button.state(["!disabled"])

    def open_results(self):
        OUTPUT.mkdir(parents=True,exist_ok=True)
        os.startfile(str(OUTPUT))

    def close(self):
        try:
            p=self.read_parameters()
            CONFIG.write_text(json.dumps(asdict(p),ensure_ascii=False,indent=2),encoding="utf-8")
        except (ValueError,OSError):
            pass
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.root.destroy()

    def callback_error(self,kind,error,tb):
        logging.error("UI callback failed",exc_info=(kind,error,tb))
        self.status.set(f"操作未完成：{error}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-default",action="store_true",help="Export default figures without opening GUI")
    parser.add_argument("--default",action="store_true",help="Ignore last saved configuration")
    args=parser.parse_args()
    logging.basicConfig(filename=BASE/"运行日志.log",level=logging.WARNING,encoding="utf-8")
    if args.export_default:
        destination=export_result(simulate(Parameters()),OUTPUT/"默认参数")
        print(destination)
        return
    root=tk.Tk()
    SimulatorApp(root,Parameters() if args.default else None)
    root.mainloop()


if __name__=="__main__":
    try:
        main()
    except Exception:
        logging.exception("Application failed to start")
        raise
