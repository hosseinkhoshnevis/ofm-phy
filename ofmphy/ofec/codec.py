"""oFEC encoder and decoders.

The encoder follows the OpenROADM/OpenZR+ definition to the bit, including
the startup rule for the first 20 block rows. The standard leaves the
decoder open, so two are provided:

* decode_hard: iterated bounded-distance decoding (iBDD). Fast, a couple
  of dB short of the soft threshold. Good for quick sweeps.
* decode_soft: Chase-Pyndiah turbo product decoding over the whole frame,
  the usual choice in the oFEC literature (G.709.3 App. III style: SISO
  iterations followed by HIHO cleanup sweeps).

Both exploit the fact that codewords of 5 consecutive block rows are
bit-disjoint, so every numpy call decodes an 80-codeword batch (times 2^p
Chase patterns for the soft one).
"""

import numpy as np

from . import ebch
from .structure import B, NCOLS, WARMUP_ROWS, OfecFrame

BIG_LLR = 64.0  # "known bit" magnitude, on top of unit mean |llr| normalisation
_LEADER_SEED = 0x0FEC  # fixed pattern, same at both ends


class OfecCodec:
    def __init__(self, n_data_rows, leader_rows=WARMUP_ROWS):
        self.frame = OfecFrame(n_data_rows, leader_rows)
        fr = self.frame
        # Leader rows depend only on their own info bits (their codewords use
        # the all-zero front), so their full content is fixed and both ends
        # can precompute it.
        rng = np.random.default_rng(_LEADER_SEED)
        self._leader_info = rng.integers(0, 2, fr.leader_rows * 16 * 111, dtype=np.uint8)
        self._leader_bits = self._encode_rows(np.zeros(fr.n_bits, np.uint8),
                                              range(fr.leader_rows),
                                              seed_leader=True)[fr.leader_bit_idx]

    # ------------------------------------------------------------------ TX
    def _encode_rows(self, M, rows, seed_leader=False):
        fr = self.frame
        if seed_leader and fr.leader_rows:
            leader_pos = fr.back_idx[:fr.leader_rows, :, :111].ravel()
            M[leader_pos] = self._leader_info
        for R in rows:
            back = fr.back_idx[R]
            if R >= WARMUP_ROWS:
                front = M[fr.front_idx[R]]
            else:
                # startup rule: parity as if the front half were all zero
                front = np.zeros((B, NCOLS), dtype=np.uint8)
            x = np.concatenate([front, M[back[:, :111]]], axis=1)  # bits 0..238
            par = ebch.parity_bits(x)
            ext = np.bitwise_xor.reduce(x, axis=1) ^ np.bitwise_xor.reduce(par, axis=1)
            M[back[:, 111:127]] = par
            M[back[:, 127]] = ext
        return M

    def encode(self, payload):
        """payload: (n_info,) bits -> (coded_stream, matrix), both uint8."""
        fr = self.frame
        payload = np.asarray(payload, dtype=np.uint8)
        if payload.size != fr.n_info:
            raise ValueError(f"payload must be {fr.n_info} bits")
        M = np.zeros(fr.n_bits, dtype=np.uint8)
        M[fr.info_matrix_idx] = payload
        M = self._encode_rows(M, range(fr.n_rows), seed_leader=True)
        return M[fr.tx_perm], M

    # ------------------------------------------------------------ helpers
    def _gather(self, rows):
        """Index arrays for a chunk of block rows -> (fi, bi, virt)."""
        fr = self.frame
        fi = fr.front_idx[rows].reshape(-1, NCOLS)
        bi = fr.back_idx[rows].reshape(-1, NCOLS)
        virt = np.repeat(np.asarray(rows) < WARMUP_ROWS, B)  # all-zero front rows
        return fi, bi, virt

    def _clamp_known(self, M):
        fr = self.frame
        M[fr.leader_bit_idx] = self._leader_bits
        M[fr.term_info_idx] = 0

    # ------------------------------------------------------------------ HD
    def decode_hard(self, coded_stream=None, n_iter=8, matrix=None):
        """Iterated BDD. Takes the received bit stream (transmit order) or a
        matrix already in structure order. Returns the payload bits."""
        fr = self.frame
        if matrix is None:
            M = np.zeros(fr.n_bits, dtype=np.uint8)
            M[fr.tx_perm] = np.asarray(coded_stream, dtype=np.uint8)
        else:
            M = matrix
        self._clamp_known(M)
        for it in range(n_iter):
            n_flips = 0
            for rows in fr.row_chunks(reverse=bool(it % 2)):
                fi, bi, virt = self._gather(rows)
                front = M[fi]
                front[virt] = 0
                cw = np.concatenate([front, M[bi]], axis=1)
                dec, fail = ebch.bdd_decode(cw)
                flip = (dec ^ cw).astype(bool) & ~fail[:, None]
                flip[virt, :NCOLS] = False  # never touch virtual front bits
                idx = np.concatenate([fi, bi], axis=1)
                M[idx[flip]] ^= 1
                n_flips += int(flip.sum())
            self._clamp_known(M)  # re-assert what the decoder already knows
            if n_flips == 0:
                break
        return M[fr.info_matrix_idx]

    # ------------------------------------------------------------------ SD
    def decode_soft(self, llrs, n_iter=4, n_lrb=6,
                    alphas=(0.20, 0.40, 0.65, 0.90),
                    betas=(0.30, 0.55, 0.85, 1.20),
                    hd_cleanup=2):
        """Chase-Pyndiah turbo product decoding of the braided structure.

        llrs: channel LLRs in transmit order, positive means bit 0.
        Returns the payload bits.
        """
        fr = self.frame
        llrs = np.asarray(llrs, dtype=np.float32)
        scale = float(np.mean(np.abs(llrs))) or 1.0
        L = np.zeros(fr.n_bits, dtype=np.float32)
        L[fr.tx_perm] = llrs / scale
        L[fr.leader_bit_idx] = BIG_LLR * (1.0 - 2.0 * self._leader_bits)
        L[fr.term_info_idx] = BIG_LLR

        # Extrinsic messages by role: e_front[pos] comes from the codeword
        # covering pos in its front half, e_back[pos] from the one covering
        # it in its back half. Each codeword reads the opposite role.
        e_front = np.zeros(fr.n_bits, dtype=np.float32)
        e_back = np.zeros(fr.n_bits, dtype=np.float32)

        for it in range(n_iter):
            a = alphas[min(it, len(alphas) - 1)]
            b = betas[min(it, len(betas) - 1)]
            for rows in fr.row_chunks(reverse=bool(it % 2)):
                fi, bi, virt = self._gather(rows)
                lf = L[fi] + a * e_back[fi]
                lf[virt] = BIG_LLR  # startup rows encode against a known zero front
                lb = L[bi] + a * e_front[bi]
                lin = np.concatenate([lf, lb], axis=1)
                soft = chase_pyndiah(lin, n_lrb, b)
                w = soft - lin
                e_front[fi[~virt]] = w[~virt, :NCOLS]
                e_back[bi] = w[:, NCOLS:]

        post = L + e_front + e_back
        M = (post < 0).astype(np.uint8)
        if hd_cleanup:
            return self.decode_hard(matrix=M, n_iter=hd_cleanup)
        self._clamp_known(M)
        return M[fr.info_matrix_idx]


def chase_pyndiah(lin, n_lrb, beta):
    """Soft-in soft-out Chase-2 decoding of a batch of eBCH words.

    lin: (n, 256) input LLRs. Returns soft outputs of the same shape.
    Test patterns flip every subset of the n_lrb least reliable bits; the
    winner is the valid codeword with the least analog weight, and per-bit
    reliabilities come from the best competitor, Pyndiah style.
    """
    n = lin.shape[0]
    mag = np.abs(lin)
    hard = (lin < 0).astype(np.uint8)

    lr = np.argpartition(mag, n_lrb, axis=1)[:, :n_lrb]
    npat = 1 << n_lrb
    pat = ((np.arange(npat)[:, None] >> np.arange(n_lrb)) & 1).astype(np.uint8)

    cand = np.repeat(hard[:, None, :], npat, axis=1)
    ii = np.arange(n)[:, None, None]
    pp = np.arange(npat)[None, :, None]
    jj = np.broadcast_to(lr[:, None, :], (n, npat, n_lrb))
    cand[ii, pp, jj] ^= pat[None, :, :]

    dec, fail = ebch.bdd_decode(cand.reshape(-1, 256))
    dec = dec.reshape(n, npat, 256)
    diff = dec != hard[:, None, :]
    metric = np.where(diff, mag[:, None, :], 0.0).sum(axis=2, dtype=np.float32)
    metric[fail.reshape(n, npat)] = np.inf

    best = np.argmin(metric, axis=1)
    rows = np.arange(n)
    m_best = metric[rows, best]
    d = dec[rows, best]

    # cheapest candidate disagreeing with the decision, per bit
    away = dec != d[:, None, :]
    m_comp = np.where(away, metric[:, :, None], np.inf).min(axis=1)

    sgn = 1.0 - 2.0 * d.astype(np.float32)
    # With a competitor, reliability is the metric gap. Without one the
    # decoder is *more* sure, not less: boost the input by beta instead of
    # replacing it (replacing is the textbook recipe, but summing two roles
    # of beta-sized outputs would then argue against perfectly clean bits).
    soft = np.where(np.isfinite(m_comp),
                    (m_comp - m_best[:, None]) * sgn,
                    lin + beta * sgn)

    # every pattern failed BDD (rare): pass the input through untouched
    dead = ~np.isfinite(m_best)
    if dead.any():
        soft[dead] = lin[dead]
    return soft
