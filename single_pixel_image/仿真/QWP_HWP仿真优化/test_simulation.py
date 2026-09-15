"""Physical limits, gain invariance and FFT validation; no hardware required."""
import json
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from scipy.special import jv

from qwp_hwp_model import Parameters, simulate
from simulation_core import jones_retarder


class SimulationTests(unittest.TestCase):
    def test_default_target(self):
        r=simulate(Parameters())
        self.assertAlmostEqual(r.metadata['dc_fft_mv'],100,places=9)
        self.assertAlmostEqual(r.metadata['vpp_mv'],160,places=9)
        np.testing.assert_allclose(r.stokes,[1,-0.6,-np.sqrt(0.32),-np.sqrt(0.32)],atol=1e-12)

    def test_known_polarization_limits(self):
        # Circular: even harmonics vanish. Linear -45: odd harmonics vanish.
        circular=simulate(Parameters(alpha_deg=0,beta_deg=0))
        linear=simulate(Parameters(alpha_deg=45,beta_deg=45))
        s1_only=simulate(Parameters(alpha_deg=45,beta_deg=22.5))
        self.assertLess(circular.harmonics[2]['fft_amplitude_mv'],1e-10)
        self.assertGreater(circular.harmonics[1]['fft_amplitude_mv'],100)
        self.assertLess(linear.harmonics[1]['fft_amplitude_mv'],1e-10)
        self.assertGreater(linear.harmonics[2]['fft_amplitude_mv'],80)
        self.assertLess(np.ptp(s1_only.voltage_mv),1e-10)
        self.assertLess(s1_only.harmonics[1]['fft_amplitude_mv'],1e-10)
        self.assertLess(s1_only.harmonics[2]['fft_amplitude_mv'],1e-10)

    def test_gain_is_not_renormalized_with_angle(self):
        a=simulate(Parameters(alpha_deg=45,beta_deg=45,delta0_rad=1.0))
        b=simulate(Parameters(alpha_deg=45,beta_deg=0,delta0_rad=1.0))
        self.assertEqual(a.metadata['GI0_mv'],b.metadata['GI0_mv'])
        self.assertAlmostEqual(a.metadata['dc_fft_mv'],100*(1-jv(0,1)),places=10)
        self.assertAlmostEqual(b.metadata['dc_fft_mv'],100*(1+jv(0,1)),places=10)

    def test_independent_jones_and_fft(self):
        for alpha,beta,delta0 in [(22.5,0,1.8),(-31.2,62.3,2.4048255577),(80,-25,8),(45,45,0)]:
            p=Parameters(alpha_deg=alpha,beta_deg=beta,delta0_rad=delta0)
            r=simulate(p)
            e=jones_retarder(beta,np.pi) @ jones_retarder(alpha,np.pi/2) @ (np.array([1.,-1.])/np.sqrt(2))
            indices=np.linspace(0,len(r.time_s)-1,23,dtype=int)
            d=p.delta0_rad*np.sin(2*np.pi*p.f_pem_khz*1000*r.time_s[indices])
            expected=np.array([2*p.c_mv*abs(np.array([1.,1.])/np.sqrt(2) @ jones_retarder(0,x) @ e)**2 for x in d])
            np.testing.assert_allclose(r.voltage_mv[indices],expected,atol=1e-9)
            self.assertLess(r.metadata['fft_bessel_max_error_mv'],1e-8)
            self.assertGreaterEqual(r.voltage_mv.min(),-1e-9)

    def test_angle_periodicity(self):
        p=Parameters(alpha_deg=-40,beta_deg=10)
        a=simulate(p)
        b=simulate(replace(p,alpha_deg=140,beta_deg=100))
        np.testing.assert_allclose(a.voltage_mv,b.voltage_mv,atol=1e-9)

    def test_invalid_inputs(self):
        for invalid in [dict(alpha_deg=float('nan')),dict(beta_deg=181),dict(delta0_rad=-1),
                        dict(c_mv=0),dict(f_pem_khz=0),dict(c_mv=float('inf'))]:
            with self.assertRaises(ValueError):
                simulate(Parameters(**invalid))


if __name__=='__main__':
    unittest.main(verbosity=2)
