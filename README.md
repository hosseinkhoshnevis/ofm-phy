# ofmphy

**Hossein Khoshnevis** · MIT licence

Baseband PHY simulation of an 800G-class coherent optical fibre modem, in
plain vectorised numpy: DP-16QAM at the 800ZR symbol rate (118.203350603 GBd),
the standard OpenROADM/OpenZR+ oFEC, a full blind receiver DSP chain, and
pre- and post-FEC BER measurement.

A note on the name: "800G" is the aggregate bit rate, not the baud.
Interoperable 800G coherent interfaces run near 118 GBd so that 16QAM's
8 bit/symbol, the 15.3% oFEC overhead and the framing overhead multiply out
to the client rate. This package simulates the equivalent complex baseband
of one such line interface. Client mapping, DSP framing/pilots and the AES
encryption block of the application standards are out of scope.

## What is inside

* **oFEC to the letter of the spec** (OpenROADM MSA / OpenZR+ sec. 7.1):
  extended BCH(256,239) component code with generator
  `g(y) = y^16+y^14+y^13+y^11+y^10+y^9+y^8+y^6+y^5+y+1` over GF(2^8)/0x11d,
  the braided 128-column square-block interleaver with its XOR twist, the
  exact input/output serialisation rectangles, and the 20-row startup rule.
  Rate 111/128, 15.3% overhead.
* **Two decoders**, since the standard fixes only the encoder:
  Chase-Pyndiah soft-decision turbo product decoding (4 SISO sweeps plus 2
  hard cleanup sweeps, 64 test patterns) and plain iterated
  bounded-distance decoding. The measured soft waterfall reaches the 1e-6
  output decade at input BER 1.95e-2, within about 0.1 dB of the reference
  architecture's qualified 2.0e-2 point; the hard decoder's knee sits near
  1.0e-2.
* **Channel**: chromatic dispersion, SOP rotations around a DGD element,
  PDL, combined TX+LO Wiener phase noise, TX-LO frequency offset, and ASE
  calibrated through OSNR in the 0.1 nm convention. Kerr nonlinearity is
  available as a split-step Manakov fibre with per-span EDFAs
  (`nl_steps > 0`, `n_spans` for long links), and the budget helper
  `osnr_from_launch` ties ASE to launch power for nonlinear studies.
* **Electrical front ends**: per-polarisation I/Q amplitude and
  quadrature-angle imbalance plus I/Q rail skew at both the transmitter and
  the receiver, and ENOB-style ADC quantisation.
* **Receiver DSP**: front-end correction (calibrated I/Q deskew plus blind
  Gram-Schmidt orthogonalisation), frequency-domain CD compensation,
  matched filter, 4th-power frequency offset estimation, and a T/2-spaced
  2x2 widely-linear butterfly equalizer running the standard CMA, RDE,
  DD-LMS cascade. The conjugate branches remove the transmit-side IQ image
  that no strictly linear equalizer can touch, and the decision-directed
  pass runs against the recovered carrier to pick up the residual ISI the
  modulus criteria cannot see. Adaptation uses block-averaged gradients,
  one numpy call per block instead of a Python loop per symbol. Carrier
  recovery is joint-polarisation blind phase search; alignment against the
  reference sequence resolves polarisation pairing, delay and quadrant;
  a short widely-linear least-squares post filter removes residual
  conjugate mixing; and the LLRs are calibrated per polarisation, because
  with PDL in the link the two pols run at different SNRs.
* **Measurement discipline**: pre-FEC BER counted on the line, post-FEC BER
  on the payload, zero-error results reported against the counting floor,
  and an AWGN-only mode that is unit-tested against the textbook 16QAM
  curve to pin the OSNR calibration.

Everything is numpy. A frame at 100 payload rows (177,600 info bits) runs
through the whole chain, both decoders included, in a couple of seconds on
a laptop-class core.

## Install

```
pip install -e .          # numpy only
pip install -e .[dev]     # + matplotlib and pytest
```

## Quick start

```python
from ofmphy import SimConfig, run_link

r = run_link(SimConfig(osnr_db=23.5, decoder="both"))
print(r["pre_fec_ber"])        # ~1.6e-2, just under the oFEC threshold
print(r["post_fec"]["sd"])     # {'errors': 0, ...}
```

Command line:

```
python examples/run_single.py --osnr 23 --rows 60 --decoder both --plots
python examples/ber_sweep.py --start 21.5 --stop 27 --frames 3
python examples/launch_power_sweep.py --spans 10   # Q vs launch power
python examples/eye_diagram.py --osnr 26           # eye before/after the EQ
```

Or use the FEC on its own:

```python
import numpy as np
from ofmphy.ofec import OfecCodec

codec = OfecCodec(n_data_rows=100)
payload = np.random.default_rng(1).integers(0, 2, codec.frame.n_info, dtype=np.uint8)
stream, _ = codec.encode(payload)
decoded = codec.decode_soft(llrs_from_your_channel)
```

## Measured waterfall

Every impairment on (fibre, laser, IQ front ends, ADC) with the full
compensation stack, 3 frames of 100 payload rows per point:

| OSNR (dB) | pre-FEC BER | post-FEC (soft) | post-FEC (hard) |
|-----------|-------------|-----------------|-----------------|
| 22.5      | 2.7e-2      | 3.4e-2          | 2.9e-2          |
| 22.8      | 2.3e-2      | 1.1e-2          | 2.4e-2          |
| 22.95     | 2.1e-2      | 1.1e-3          | 2.2e-2          |
| 23.05     | 2.0e-2      | 3.9e-5          | 2.0e-2          |
| 23.1      | 1.95e-2     | 1.1e-6          | 1.9e-2          |
| 23.5      | 1.6e-2      | 0 (< 2e-6)      | 1.3e-2          |
| 24.5      | 8.7e-3      | 0 (< 2e-6)      | 0 (< 2e-6)      |
| 27.0      | 1.1e-3      | 0 (< 2e-6)      | 0 (< 2e-6)      |

The transition points carry up to 17.8M payload bits each, so the soft
curve is measured through the 1e-3, 1e-4, 1e-5 and 1e-6 decades rather
than extrapolated: a 0.3e-2 slice of input BER maps onto four output
decades, within about 0.1 dB of the reference decoder's qualified 2.0e-2
point. Hard decoding costs about 1.5 dB. At the operating point the full
chain needs OSNR 23.1 dB against a 22.5 dB theory bound, a 0.61 dB
implementation penalty that is mostly the 6-bit ADC floor and the TX-skew
residual, and 3.9 dB inside the 27 dB 800ZR receiver compliance limit.
Both gaps sit where the literature says they should: published
simulation-only penalties run 0.4 to 0.8 dB, while real 118-120 GBd
hardware measures 24 to 26 dB back-to-back and the cleanest published
120 GBd component demonstration needed 25 dB. `docs/standards_gap.py`
regenerates the breakdown, which the paper renders as a table alongside
the literature comparison.

Switching the compensation off (`frontend_comp=False,
eq_widely_linear=False, wl_taps=0`) collapses the chain, which is how the
tests keep those blocks meaningful. A decision-aided ML CPE stage (Zhou
style second stage) ships in `rxdsp.py` but is off by default: measured
against BPS plus DD-LMS at 100 kHz-class linewidth it adds nothing.
Enable `ml_window` for MHz-class lasers.

With the nonlinear fibre on over 10 x 80 km and OSNR tied to launch power,
the classic bell curve comes out: Q peaks at 11.3 dB at +4 dBm total
launch, the oFEC returns error-free payloads from -2 to +10 dBm, and Kerr
collapses the link by +12 dBm. The plots, block diagrams, per-block DSP
equations, eye diagrams and methodology are in the paper (see the Paper
section below).

## Documentation

`ofmphy.pdf` is a draft manuscript. It is the single write-up of the project: 
the block-by-block transmitter, channel and receiver DSP with their equations, 
the system and oFEC diagrams, the measured waterfall, eye and launch-power results, the
distance-to-standard table, the literature comparison, and a contribution
roadmap.

## Layout

```
ofmphy/
  config.py        SimConfig, every knob, 800ZR defaults
  sim.py           run_link: bits -> ... -> BER, plus the line scrambler
  ofec/
    galois.py      GF(256) tables
    ebch.py        eBCH(256,239): encoder, syndromes, vectorised BDD
    structure.py   the braided matrix: index maps, spec equations
    codec.py       encoder + iBDD and Chase-Pyndiah decoders
  mapping.py       Gray DP-16QAM, hard demap, max-log LLRs
  shaping.py       frequency-domain RRC (circular frame convention)
  channel.py       CD, SOP/DGD/PDL, SSFM Manakov fibre, laser, ASE/OSNR,
                   IQ front ends, ADC
  rxdsp.py         deskew+GSOP, CDC, FOE, WL equalizer, BPS, ML CPE,
                   alignment, WL fit
  metrics.py       Q factor, EVM, exact Gray-16QAM theory BER
examples/          run_single.py, ber_sweep.py, launch_power_sweep.py,
                   eye_diagram.py
tests/             49 tests: spec structure checks, decoder knees, theory
                   match, front-end compensation, split-step sanity
docs/              measured data (CSV), standards-gap utility, and the
                   paper with its figure source (docs/paper/make_figs.py)
```

## Simulation conventions worth knowing

The frame is processed circularly, which removes filter transients and
makes dispersion exactly invertible. To keep the circle seamless, the
laser walk is pinned into a Brownian bridge and the frequency offset
snaps to the FFT grid. Frame sync, polarisation pairing, quadrant
rotation, the LLR noise variance and the widely-linear post fit use the
known transmit sequence, standing in for the pilot machinery and
decision-directed loops of real modems. The I/Q deskew uses the
configured skew values, following the factory-calibration practice of
real modules, while the imbalance corrections are blind. Finite oFEC
frames add a fixed 20-row leader that absorbs the spec startup transient
and 21 termination rows covering the coupling span. Both are excluded
from payload BER but counted as line bits; set `leader_rows=0` on
`OfecCodec` to study the cold-start transient itself. The nonlinear fibre
is a symmetric split-step over the Manakov equation with lumped per-span
EDFAs, and receive CD compensation stays strictly linear, so the Kerr
residual is exactly the nonlinear penalty. Not modelled:
clock/sampling-rate offset and timing recovery (the frame is
symbol-synchronous), laser RIN, and DAC quantisation. The natural
extension points are noted in the module docstrings.

## References

* OpenZR+ MSA Technical Specification (openzrplus.org), oFEC definition, sec. 7.1
* OpenROADM MSA W-Port Digital Specification (openroadm.org), the original oFEC text
* OIF-800ZR Implementation Agreement (oiforum.com), line parameters
* ITU-T G.709.3 App. III, the 3 SISO + 2 HIHO reference decoder shape
* Pyndiah, "Near-optimum decoding of product codes", IEEE Trans. Commun. 1998
* Zhou, "An improved feed-forward carrier recovery algorithm for coherent
  receivers with M-QAM modulation format", IEEE PTL 2010
* Fatadin, Savory, Ives, "Compensation of quadrature imbalance in an
  optical QPSK coherent receiver", IEEE PTL 2008

MIT licence. If you spot a deviation from the spec, open an issue; the
structure tests in `tests/test_structure.py` are the place to encode it.
