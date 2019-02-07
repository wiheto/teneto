import teneto


def test_suffixstripping():
    fname = 'sub-01_run-02_preproc_mean.npy.gz'
    fname_base, file_format = teneto.utils.drop_bids_suffix(fname)
    if not file_format == '.npy.gz':
        raise AssertionError()
    if not fname_base == 'sub-01_run-02':
        raise AssertionError()

