"""oFEC bit-interleaving structure (OpenROADM MSA / OpenZR+ sec. 7.1).

The code lives on a semi-infinite binary matrix of N = 128 columns viewed
as 16x16 square blocks. A bit has address {R, C, r, c}: block row R,
block column C = 0..7, and r, c = 0..15 inside the square block.
Constituent codeword (R, r) reads its first half from earlier rows and
writes 111 fresh info bits plus 17 parity bits into block row R:

    W[R,r](k)     = V((R^1) - 20 + 2*(k//16), k//16, (k%16)^r, r)   k = 0..127
    W[R,r](128+j) = V(R, j//16, r, (j%16)^r)                        j = 0..127

(^ is bitwise XOR; 20 = 2G + 2N/B with guard G = 2.) Every bit lands in
exactly two codewords 5..21 block rows apart, which is what gives the
braided/zipper behaviour. Rate 111/128, overhead 17/111 = 15.3%.

Finite-frame layout used here:

    [ leader | payload | termination ]

* leader (default 20 rows) carries fixed pseudorandom info bits known at
  the receiver. It soaks up the spec startup transient: the first 20 block
  rows encode against an all-zero front (the W H' = 0 rule), so without a
  leader the first payload rows are measurably under-protected at cold
  start. Set leader_rows=0 to study exactly that.
* termination (21 rows, padded so the total is even) carries zero info so
  the payload tail keeps its second protection.

Payload rows therefore sit in steady state; rows within ~21 rows of either
boundary see slightly *stronger* protection than a true infinite stream
because their neighbours are known. Use a few hundred payload rows when
chasing exact threshold numbers.
"""

import numpy as np

B = 16          # rows/cols of a square block
NCOLS = 128
NBLK_COLS = 8   # NCOLS // B
WARMUP_ROWS = 20   # 2G + 2N/B: codewords here take an all-zero front
TERM_ROWS = 21     # coupling span; last payload row is re-protected up to R+21
INFO_PER_ROW = B * 111  # 1776
ROW_BITS = B * NCOLS    # 2048


def _flat(Rb, C, r, c):
    return (Rb * B + r) * NCOLS + C * B + c


def _info_positions(rows):
    """Matrix flat indices of the 111*16 fresh-info bits of each block row."""
    R = np.asarray(rows)[:, None, None]
    r = np.arange(B)[None, :, None]
    j = np.arange(111)[None, None, :]
    return _flat(R, j // B, r, (j % B) ^ r)


class OfecFrame:
    """Precomputed index maps for one finite oFEC frame."""

    def __init__(self, n_data_rows, leader_rows=WARMUP_ROWS):
        if n_data_rows < 1:
            raise ValueError("need at least one payload row")
        if leader_rows % 2:
            raise ValueError("leader_rows must be even (output rectangles pair rows)")
        self.n_data_rows = n_data_rows
        self.leader_rows = leader_rows
        n = leader_rows + n_data_rows + TERM_ROWS
        self.n_rows = n + n % 2
        self.n_bits = self.n_rows * ROW_BITS
        self.n_info = n_data_rows * INFO_PER_ROW

        R = np.arange(self.n_rows)[:, None, None]
        r = np.arange(B)[None, :, None]
        k = np.arange(NCOLS)[None, None, :]

        # Back half: the positions codeword (R, r) owns inside block row R.
        # j = 0..110 info, 111..126 BCH parity, 127 the extension bit.
        self.back_idx = _flat(R, k // B, r, (k % B) ^ r)

        # Front half: sources in earlier rows. Only dereferenced for
        # R >= WARMUP_ROWS, where every source row index is >= 0.
        src = (R ^ 1) - 20 + 2 * (k // B)
        self.front_idx = _flat(np.maximum(src, 0), k // B, (k % B) ^ r, r)

        # Fresh-info serialisation, the spec input map: u(i) -> W[R,r](128+j),
        # arriving in 3552-bit rectangles that cover a pair of block rows.
        lo, hi = leader_rows, leader_rows + n_data_rows
        Rd = np.arange(lo, hi)[:, None, None]
        rd = np.arange(B)[None, :, None]
        j = np.arange(111)[None, None, :]
        p = (Rd % 2) * B + rd  # leader_rows is even, parity is preserved
        u = ((Rd - lo) // 2) * 3552 + p * (16 - j // 96) + (j // 16) * 512 + (j % 16)
        m = _info_positions(np.arange(lo, hi))
        self.info_matrix_idx = np.zeros(self.n_info, dtype=np.int64)
        self.info_matrix_idx[u.ravel()] = m.ravel()

        # Bits the decoder knows in advance: every leader-row bit (the whole
        # rows are a fixed pattern) and the zero info bits of the termination.
        self.leader_bit_idx = np.arange(leader_rows * ROW_BITS)
        self.term_info_idx = _info_positions(np.arange(hi, self.n_rows)).ravel()

        # Transmit order, the spec output map: 4096-bit rectangles per row
        # pair, square blocks in raster order (even row block, then odd).
        Rb = np.arange(self.n_rows)[:, None, None, None]
        C = np.arange(NBLK_COLS)[None, :, None, None]
        rr = np.arange(B)[None, None, :, None]
        cc = np.arange(B)[None, None, None, :]
        t = (Rb // 2) * 4096 + (Rb % 2) * 256 + C * 512 + rr * 16 + cc
        self.tx_perm = np.zeros(self.n_bits, dtype=np.int64)
        self.tx_perm[t.ravel()] = _flat(Rb, C, rr, cc).ravel()

    # Codewords of rows R..R+4 never share a bit (closest coupling is 5 rows),
    # so the decoders batch safe chunks of consecutive rows.
    CHUNK = 5

    def row_chunks(self, reverse=False):
        starts = range(0, self.n_rows, self.CHUNK)
        chunks = [np.arange(s, min(s + self.CHUNK, self.n_rows)) for s in starts]
        return chunks[::-1] if reverse else chunks
