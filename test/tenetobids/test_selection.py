import teneto


def test_define():
    dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'
    tnet = teneto.TenetoBIDS(
        dataset_path, pipeline='fmriprep', raw_data_exists=False)
    if not len(tnet.get_selected_files(quiet=1)) == 6:
        raise AssertionError()
    tnet = teneto.TenetoBIDS(
        dataset_path, pipeline='fmriprep', raw_data_exists=False)
    if not len(tnet.get_selected_files(quiet=1, forfile={'sub': '001'})) == 3:
        raise AssertionError()
    fname = 'sub-001_task-a_run-beta_bold_preproc.nii.gz'
    if not len(tnet.get_selected_files(quiet=1, forfile=fname)) == 1:
        raise AssertionError()


def test_define_sub_then_task():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='fmriprep', bids_tags={'sub': '001'}, raw_data_exists=False)
    if not len(tnet.get_selected_files(quiet=1)) == 3:
        raise AssertionError()
    tnet.set_bids_tags({'task': 'a'})
    if not len(tnet.get_selected_files(quiet=1)) == 2:
        raise AssertionError()


def test_define_run_then_sub():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='fmriprep', bids_tags={'run': 'alpha'}, raw_data_exists=False)
    if not len(tnet.get_selected_files(quiet=1)) == 4:
        raise AssertionError()
    tnet.set_bids_tags({'sub': '001'})
    if not len(tnet.get_selected_files(quiet=1)) == 2:
        raise AssertionError()


def test_define_task_then_run():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='fmriprep', bids_tags={'task': 'a'}, raw_data_exists=False)
    if not len(tnet.get_selected_files(quiet=1)) == 4:
        raise AssertionError()
    tnet.set_bids_tags({'run': 'beta'})
    if not len(tnet.get_selected_files(quiet=1)) == 2:
        raise AssertionError()


def test_get_pipeline_alternatives():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='teneto-tests', bids_tags={'task': 'a'}, raw_data_exists=False)
    pipeline = tnet.get_pipeline_alternatives()
    if not 'fmriprep' in pipeline:
        raise AssertionError()
    if not 'teneto-tests' in pipeline:
        raise AssertionError()

def test_get_pipeline_subdir_alternatives():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='teneto-tests', bids_tags={'task': 'a'}, raw_data_exists=False)
    subdir = tnet.get_pipeline_subdir_alternatives()
    if not 'parcellation' in subdir:
        raise AssertionError()
    if not 'tvc' in subdir:
        raise AssertionError()
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', bids_tags={'task': 'a'}, raw_data_exists=False)
    subdir = tnet.get_pipeline_subdir_alternatives()
    if not subdir is None:
        raise AssertionError()

def test_set_bad_subjects():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_tags={'task': 'a'}, raw_data_exists=False)
    tnet.set_bad_subjects('001')
    if not len(tnet.bad_files) == 2:
        raise AssertionError()
    tnet.set_bad_subjects('001', reason='last', oops=True)
    if not len(tnet.bad_files) == 0:
        raise AssertionError()
    if not '001' in tnet.bids_tags['sub']:
        raise AssertionError()
    if not len(tnet.bids_tags['sub']) == 2:
        raise AssertionError()
    tnet.set_bad_subjects('001')
    tnet.set_bad_subjects(['002'])
    if not len(tnet.bad_subjects) == 2:
        raise AssertionError()


def test_print():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',
                             pipeline='teneto-tests', bids_tags={'task': 'a'}, raw_data_exists=False)
    tnet.print_dataset_summary()
