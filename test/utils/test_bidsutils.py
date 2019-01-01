import teneto
import numpy as np
import matplotlib.pyplot as plt
import pytest


def test_suffixstripping():
    fname = 'sub-01_run-02_preproc_mean.npy.gz'
    fname_base, file_format = teneto.utils.drop_bids_suffix(fname)
    assert file_format == '.npy.gz'
    assert fname_base == 'sub-01_run-02'