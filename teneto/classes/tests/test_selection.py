import teneto
import pytest

def test_define():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='fmriprep', raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 6


def test_define_sub_then_task():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='fmriprep', subjects='001', raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 3
    tnet.set_tasks('a')
    assert len(tnet.get_selected_files(quiet=1)) == 2


def test_define_run_then_sub():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='fmriprep', runs='alpha', raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 4
    tnet.set_subjects('001')
    assert len(tnet.get_selected_files(quiet=1)) == 2


def test_define_task_then_run():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='fmriprep', tasks='a', raw_data_exists=False)
    assert len(tnet.get_selected_files(quiet=1)) == 4
    tnet.set_runs('beta')
    assert len(tnet.get_selected_files(quiet=1)) == 2


def test_get_pipeline_alternatives():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='teneto-tests', tasks='a', raw_data_exists=False)
    pipeline = tnet.get_pipeline_alternatives()
    assert 'fmriprep' in pipeline
    assert 'teneto-tests' in pipeline


def test_set_bad_subjects():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='teneto-tests', tasks='a', raw_data_exists=False)
    tnet.set_bad_subjects('001')
    tnet.set_bad_subjects(['002'])
    assert len(tnet.bad_subjects) == 2


def test_set_space_error():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='teneto-tests', tasks='a', raw_data_exists=False,njobs=1)
    with pytest.raises(ValueError):
        tnet.set_space('bla')


def test_print():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='teneto-tests', tasks='a', raw_data_exists=False)
    tnet.print_dataset_summary()

def test_setanalysisstep():
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='teneto-tests', tasks='a', raw_data_exists=False)
    tnet.set_analysis_steps('a')
    tnet.set_analysis_steps('b',add_step=True)
    assert tnet.analysis_steps == ['a','b']
    tnet.set_analysis_steps(['a','b'])
    assert tnet.analysis_steps == ['a','b']
    tnet.set_analysis_steps(['c','d'],add_step=True)
    assert tnet.analysis_steps == ['a','b','c','d']
    
def test_halftests():
    #these tests could be made better
    tnet = teneto.TenetoBIDS(teneto.__path__[
                             0] + '/data/testdata/dummybids/', pipeline='teneto-tests', tasks='a', raw_data_exists=False)
    tnet.get_space_alternatives()
