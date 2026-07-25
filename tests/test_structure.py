import numpy as np
import pytest

from ofmphy.ofec.structure import (B, NCOLS, TERM_ROWS, WARMUP_ROWS, OfecFrame)


@pytest.fixture(scope="module")
def frame():
    return OfecFrame(40)


def test_back_halves_tile_the_matrix(frame):
    assert np.array_equal(np.sort(frame.back_idx.ravel()), np.arange(frame.n_bits))


def test_front_references_unique(frame):
    fronts = frame.front_idx[WARMUP_ROWS:].ravel()
    assert np.bincount(fronts, minlength=frame.n_bits).max() <= 1


def test_payload_rows_doubly_protected(frame):
    cover = np.bincount(frame.front_idx[WARMUP_ROWS:].ravel(), minlength=frame.n_bits)
    rowcov = cover.reshape(frame.n_rows, B * NCOLS)
    lo, hi = frame.leader_rows, frame.leader_rows + frame.n_data_rows
    assert (rowcov[lo:hi] == 1).all(), "every payload bit must appear in two codewords"


def test_chunked_rows_are_disjoint(frame):
    for rows in frame.row_chunks():
        fi = frame.front_idx[rows]
        virt = np.asarray(rows) < WARMUP_ROWS
        idx = [frame.back_idx[rows].ravel(), fi[~virt].ravel()]
        idx = np.concatenate(idx)
        assert idx.size == np.unique(idx).size


def test_tx_perm_is_a_permutation(frame):
    assert np.array_equal(np.sort(frame.tx_perm), np.arange(frame.n_bits))


def test_info_map_is_a_bijection(frame):
    idx = frame.info_matrix_idx
    assert idx.size == frame.n_info
    assert np.unique(idx).size == idx.size


def test_coupling_span_matches_spec():
    # front sources of an even row R are the odd rows R-19 .. R-5; of an odd
    # row, the even rows R-21 .. R-7 (the R^1 term always flips parity)
    fr = OfecFrame(30)
    for R, lo in ((WARMUP_ROWS + 10, -19), (WARMUP_ROWS + 11, -21)):
        src = np.unique(fr.front_idx[R] // NCOLS // B)
        assert np.array_equal(src, np.arange(R + lo, R + lo + 16, 2))
        assert ((src % 2) != (R % 2)).all()


def test_odd_total_padded_even():
    fr = OfecFrame(30)  # 20 + 30 + 21 = 71 -> padded to 72
    assert fr.n_rows == 30 + WARMUP_ROWS + TERM_ROWS + 1
    assert fr.n_rows % 2 == 0
