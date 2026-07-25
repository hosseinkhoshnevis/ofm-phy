import numpy as np
import pytest

from ofmphy import SimConfig, run_link
from ofmphy.metrics import theory_ber_16qam


def awgn_only(**kw):
    return SimConfig(fibre_km=0, dgd_ps=0, pol_theta1=0, pol_theta2=0,
                     linewidth_hz=0, freq_offset_hz=0, pdl_db=0, adc_bits=0,
                     tx_iq_amp_db=0, tx_iq_phase_deg=0, tx_iq_skew_ps=0,
                     rx_iq_amp_db=0, rx_iq_phase_deg=0, rx_iq_skew_ps=0, **kw)


def test_awgn_ber_matches_theory():
    """With every impairment off, the measured pre-FEC BER has to sit on the
    textbook 16QAM curve - this pins the OSNR/Es-N0 calibration end to end."""
    cfg = awgn_only(n_data_rows=40, osnr_db=24.0, decoder="hd")
    r = run_link(cfg)
    th = theory_ber_16qam(r["esn0_db"])
    assert 0.7 < r["pre_fec_ber"] / th < 1.4


def test_full_chain_high_osnr_is_clean():
    r = run_link(SimConfig(n_data_rows=30, osnr_db=30.0, decoder="both"))
    assert r["pre_fec_ber"] < 1e-3
    assert r["post_fec"]["sd"]["errors"] == 0
    assert r["post_fec"]["hd"]["errors"] == 0
    assert abs(r["freq_offset_err_hz"]) < 20e6


def test_full_chain_near_threshold():
    """OSNR 23.5 dB puts the blind chain - front-end imperfections and all -
    just under the oFEC 2e-2 operating point; soft decoding must clean it
    up completely."""
    r = run_link(SimConfig(n_data_rows=40, osnr_db=23.5, decoder="sd"))
    assert 1.0e-2 < r["pre_fec_ber"] < 2.2e-2
    assert r["post_fec"]["sd"]["errors"] == 0


def test_decoder_both_reports_two_results():
    r = run_link(SimConfig(n_data_rows=30, osnr_db=26.0, decoder="both"))
    assert set(r["post_fec"]) == {"sd", "hd"}
