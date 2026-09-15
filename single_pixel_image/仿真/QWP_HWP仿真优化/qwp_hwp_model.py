"""Interactive simulation model. Angles change polarization; optical gain stays fixed."""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np
from scipy.special import jv
from simulation_core import retarder, plot_results


@dataclass(frozen=True)
class Parameters:
    alpha_deg: float = 27.775049006023256
    beta_deg: float = 24.715988667582394
    f_pem_khz: float = 50.0
    delta0_rad: float = 2.4048255576957724
    c_mv: float = 100.0  # C = G*I0/2, NOT a DC renormalization target.

    def validate(self):
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("请输入有限数值，不能使用 NaN 或无穷大。")
        if not all(-180 <= x <= 180 for x in (self.alpha_deg, self.beta_deg)):
            raise ValueError("α 和 β 的输入范围为 -180°～180°。")
        if not 0.01 <= self.f_pem_khz <= 1000:
            raise ValueError("PEM 频率范围为 0.01～1000 kHz。")
        if not 0 <= self.delta0_rad <= 10:
            raise ValueError("延迟幅值 δ0 的范围为 0～10 rad。")
        if not 0 < self.c_mv <= 10000:
            raise ValueError("电压尺度 C 的范围为 0～10000 mV（不含 0）。")


@dataclass
class Result:
    parameters: Parameters
    stokes: np.ndarray
    time_s: np.ndarray
    voltage_mv: np.ndarray
    frequency_hz: np.ndarray
    fft_complex_mv: np.ndarray
    fft_amplitude_mv: np.ndarray
    harmonics: list
    metadata: dict


def simulate(p: Parameters) -> Result:
    p.validate()
    stokes = retarder(p.beta_deg, np.pi) @ retarder(p.alpha_deg, np.pi/2) @ np.array([1.,0.,-1.,0.])
    cycles, points_per_cycle = 200, 1024
    fp = p.f_pem_khz*1000
    fs, count = fp*points_per_cycle, cycles*points_per_cycle
    time_s = np.arange(count)/fs
    phase = 2*np.pi*fp*time_s
    delay = p.delta0_rad*np.sin(phase)
    # Do NOT force Vpp back to 160 mV or divide by an angle-dependent DC.
    voltage = p.c_mv*(stokes[0] + stokes[2]*np.cos(delay) + stokes[3]*np.sin(delay))
    fft = np.fft.rfft(voltage)/count
    frequency = np.fft.rfftfreq(count,1/fs)
    amplitude = np.abs(fft)
    amplitude[1:-1] *= 2
    rows = []
    for n in range(31):
        k = n*cycles
        signed = (p.c_mv*(stokes[0]+jv(0,p.delta0_rad)*stokes[2]) if n == 0
                  else 2*p.c_mv*jv(n,p.delta0_rad)*stokes[3 if n%2 else 2])
        rows.append(dict(
            harmonic=n, frequency_hz=n*fp, fft_amplitude_mv=float(amplitude[k]),
            theory_amplitude_mv=float(abs(signed)), signed_theory_mv=float(signed),
            cosine_coefficient_mv=float(fft[k].real*(1 if n==0 else 2)),
            sine_coefficient_mv=float(-2*fft[k].imag) if n else 0.,
            rms_mv=float(amplitude[k]/(np.sqrt(2) if n else 1)),
            cosine_phase_deg=float(np.rad2deg(np.angle(fft[k]))) if n and amplitude[k]>1e-9 else None,
        ))
    # Exact extrema: optimize over the actual PEM delay interval, not merely samples.
    phi = np.arctan2(stokes[3],stokes[2])
    delays = [-p.delta0_rad, p.delta0_rad]
    delays += [phi+k*np.pi for k in range(-6,7) if -p.delta0_rad <= phi+k*np.pi <= p.delta0_rad]
    extrema = p.c_mv*(1+stokes[2]*np.cos(delays)+stokes[3]*np.sin(delays))
    metadata = dict(
        **asdict(p), model="P1(-45) -> QWP(alpha) -> HWP(beta) -> PEM(0) -> P2(+45)",
        normalized_stokes_after_hwp=stokes.tolist(), f_pem_hz=fp,
        C_GI0_over_2_mv=p.c_mv, GI0_mv=2*p.c_mv,
        cycles=cycles, sample_rate_hz=fs, samples=count, frequency_resolution_hz=fs/count,
        dc_fft_mv=float(fft[0].real), min_mv=float(np.min(extrema)), max_mv=float(np.max(extrema)),
        vpp_mv=float(np.ptp(extrema)), vpp_sampled_mv=float(np.ptp(voltage)),
        fft_bessel_max_error_mv=float(max(abs(r['fft_amplitude_mv']-r['theory_amplitude_mv']) for r in rows)),
        assumptions="Ideal waveplates and P1; fixed GI0; flat detector response; no noise, offset or chopper.",
        amplitude_convention="DC = mean; nonzero FFT = peak; AC RMS = peak/sqrt(2); signed coefficients preserved in CSV.",
    )
    return Result(p,stokes,time_s,voltage,frequency,fft,amplitude,rows,metadata)


def export_result(result: Result, directory: Path):
    """Write five standalone PNG/SVG figures plus the full numeric result."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    r = result
    plot_results(directory,r.time_s,r.voltage_mv,r.frequency_hz,r.fft_amplitude_mv,r.harmonics,r.metadata)
    (directory/"parameters.json").write_text(json.dumps(r.metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    with (directory/"harmonic_amplitudes.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(r.harmonics[0]))
        writer.writeheader()
        writer.writerows(r.harmonics)
    np.savez_compressed(directory/"simulation_data.npz",time_s=r.time_s,voltage_mv=r.voltage_mv,
                        frequency_hz=r.frequency_hz,fft_amplitude_mv=r.fft_amplitude_mv,
                        fft_complex_mv=r.fft_complex_mv)
    np.savetxt(directory/"waveform_first_5_cycles.csv",np.column_stack([r.time_s[:5120],r.voltage_mv[:5120]]),
               delimiter=",",header="time_s,voltage_mv",comments="",fmt="%.12g")
    keep=r.frequency_hz<=30*r.metadata['f_pem_hz']
    np.savetxt(directory/"spectrum_0_to_30f.csv",np.column_stack([r.frequency_hz[keep],r.fft_amplitude_mv[keep]]),
               delimiter=",",header="frequency_hz,peak_amplitude_mv",comments="",fmt="%.12g")
    return directory
