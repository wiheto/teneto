import teneto
from teneto.utils import make_parcellation
import templateflow.api as tf
import pandas as pd
import pytest

def test_parcellation():
    datafile = teneto.__path__[0]
    datafile += '/data/testdata/dummybids/derivatives/fmriprep/sub-001/func/'
    datafile += 'sub-001_task-a_run-01_desc-preproc_bold.nii.gz'
    # Run correctly
    parc = make_parcellation(
        datafile, atlas='Schaefer2018', atlas_desc='100Parcels17Networks')
    if not parc.shape == (2, 100):
        raise AssertionError()
    # Error returns if too many atlases found
    with pytest.raises(ValueError):
        parc = make_parcellation(datafile, atlas='Schaefer2018')
    # Run correctly and get meta data
    parc, parcinfo = make_parcellation(
        datafile, atlas='Schaefer2018', atlas_desc='100Parcels17Networks', return_meta=True)
    if not parc.shape == (2, 100):
        raise AssertionError()
    if not isinstance(parcinfo, pd.DataFrame):
        raise AssertionError()
