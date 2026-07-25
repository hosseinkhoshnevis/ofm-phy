"""Simulation configuration.

Defaults follow the OIF 800ZR / OpenROADM 800G interoperable line format:
DP-16QAM at 118.203350603 GBd with the OpenROADM oFEC (rate 111/128, 15.3%
overhead, pre-FEC threshold 2.0e-2). Everything is a plain dataclass field,
so override whatever you like per run.
"""

from dataclasses import dataclass


@dataclass
class SimConfig:
    # ---- line format ----------------------------------------------------
    symbol_rate: float = 118.203350603e9  # Bd, 800ZR nominal (+-20 ppm in real life)
    sps: int = 2                          # simulation oversampling per polarisation
    rolloff: float = 0.05                 # RRC roll-off, ZR-class tight shaping

    # ---- frame / FEC ----------------------------------------------------
    n_data_rows: int = 100    # oFEC block rows of payload, 1776 info bits each
    decoder: str = "sd"       # "sd", "hd" or "both"
    sd_iters: int = 4         # Chase-Pyndiah SISO iterations
    chase_lrb: int = 6        # least-reliable bits per codeword -> 2^6 test patterns
    hd_iters: int = 8         # sweeps of the plain iBDD decoder ("hd" mode)
    hd_cleanup: int = 2       # HIHO sweeps after the SD pass (G.709.3 App. III style)
    alphas: tuple = (0.20, 0.40, 0.65, 0.90)  # extrinsic scaling per SD iteration
    betas: tuple = (0.30, 0.55, 0.85, 1.20)   # no-competitor boost per iteration

    # ---- noise ----------------------------------------------------------
    osnr_db: float = 23.0     # in 0.1 nm (12.5 GHz), both polarisations
    ref_bw: float = 12.5e9

    # ---- fibre / impairments -------------------------------------------
    fibre_km: float = 80.0        # span length
    n_spans: int = 1              # identical amplified spans (DCI default: 1)
    dispersion: float = 17.0      # ps/(nm km), G.652 at 1550 nm
    wavelength: float = 1550e-9
    # Kerr nonlinearity: enabled by nl_steps > 0, otherwise the fibre is
    # the linear CD operator and the power/attenuation knobs are unused
    nl_steps: int = 0             # SSFM steps over the span
    launch_power_dbm: float = 2.0 # total launch power, both polarisations
    gamma_w_km: float = 1.3       # Kerr coefficient [1/(W km)]
    alpha_db_km: float = 0.2      # fibre attenuation
    amp_nf_db: float = 5.0        # EDFA noise figure, for the OSNR budget
    pol_theta1: float = 0.35      # rad, SOP rotation ahead of the DGD element
    pol_theta2: float = 0.61      # rad, SOP rotation after it
    dgd_ps: float = 3.0
    linewidth_hz: float = 100e3   # per laser; TX and LO both count
    freq_offset_hz: float = 1.0e9 # TX-LO detuning (ZR budget allows +-1.8 GHz)
    pdl_db: float = 1.0           # polarisation dependent loss

    # ---- electrical front ends -----------------------------------------
    # residual (post-factory-calibration) modulator / hybrid imperfections
    tx_iq_amp_db: float = 0.5     # Q-rail gain error, per polarisation
    tx_iq_phase_deg: float = 2.0  # quadrature angle error
    tx_iq_skew_ps: float = 1.0    # I-Q differential delay
    rx_iq_amp_db: float = 0.5
    rx_iq_phase_deg: float = 2.0
    rx_iq_skew_ps: float = 1.0
    adc_bits: int = 6             # ENOB-style uniform quantiser, 0 = ideal ADC
    adc_clip_sigma: float = 3.2   # full scale in units of rail rms

    # ---- receiver DSP ---------------------------------------------------
    eq_taps: int = 21
    eq_block: int = 64
    eq_widely_linear: bool = True  # conjugate branches handle the TX IQ image
    cma_sweeps: int = 3
    rde_sweeps: int = 3
    dd_sweeps: int = 3        # decision-directed LMS after carrier recovery
    mu_cma: float = 2e-2
    mu_rde: float = 5e-3
    mu_dd: float = 4e-3
    bps_phases: int = 32      # test phases over pi/2, the usual choice for 16QAM
    bps_window: int = 96      # optimum grows as the linewidth shrinks; tuned here
    ml_window: int = 0        # decision-aided ML refinement after BPS (Zhou style,
    ml_iters: int = 2         # 0 disables). Measured no gain at 100 kHz linewidth
                              # because BPS plus DD-LMS already reach the noise
                              # floor. Enable it for MHz-class lasers.
    frontend_comp: bool = True  # calibrated I/Q deskew + blind Gram-Schmidt
    wl_taps: int = 5            # widely-linear TX-IQ post filter (0 disables)
    wl_block: int = 2048        # symbols per WL least-squares fit

    seed: int = 2026

    @property
    def sample_rate(self) -> float:
        return self.symbol_rate * self.sps

    @property
    def n_info_bits(self) -> int:
        return self.n_data_rows * 16 * 111
