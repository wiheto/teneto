import teneto


def test_define():
    dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'
    tnet = teneto.TenetoBIDS(
        dataset_path, pipeline='fmriprep', raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 6
    tnet = teneto.TenetoBIDS(
        dataset_path, pipeline='fmriprep', raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1, forfile={'sub': '001'})) == 3
    fname = 'sub-001_task-a_run-beta_bold_preproc.nii.gz'
    assert len(tnet.get_selected_files(quiet=1, forfile=fname)) == 1


def test_define_sub_then_task():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='fmriprep', bids_tags={'sub': '001'}, raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 3
    tnet.set_bids_tags({'task': 'a'})
    assert len(tnet.get_selected_files(quiet=1)) == 2


def test_define_run_then_sub():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='fmriprep', bids_tags={'run': 'alpha'}, raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 4
    tnet.set_bids_tags({'sub': '001'})
    assert len(tnet.get_selected_files(quiet=1)) == 2


def test_define_task_then_run():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='fmriprep', bids_tags={'task': 'a'}, raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 4
    tnet.set_bids_tags({'run': 'beta'})
    assert len(tnet.get_selected_files(quiet=1)) == 2


def test_get_pipeline_alternatives():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='teneto-tests', bids_tags={'task': 'a'}, raw_data_exists=False)
    pipeline = tnet.get_pipeline_alternatives()
    assert 'fmriprep' in pipeline
    assert 'teneto-tests' in pipeline


def test_set_bad_subjects():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_tags={'task': 'a'}, raw_data_exists=False)
    tnet.set_bad_subjects('001')
    assert len(tnet.bad_files) == 2
    tnet.set_bad_subjects('001', reason='last', oops=True)
    assert len(tnet.bad_files) == 0
    assert '001' in tnet.bids_tags['sub']
    assert len(tnet.bids_tags['sub']) == 2
    tnet.set_bad_subjects('001')
    tnet.set_bad_subjects(['002'])
    assert len(tnet.bad_subjects) == 2


def test_print():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='teneto-tests', bids_tags={'task': 'a'}, raw_data_exists=False)
    tnet.print_dataset_summary()
