
from teneto import TenetoWorkflow
import numpy as np
import teneto


def test_workflow_temporalnetwork():
    G = np.random.normal(0, 1, [3, 3, 5])
    Gth = teneto.utils.binarize(
        G, threshold_type='percent', threshold_level=0.05)
    Gmag = teneto.utils.binarize(
        G, threshold_type='magnitude', threshold_level=0)
    Dth = teneto.networkmeasures.temporal_degree_centrality(Gth)
    Dmag = teneto.networkmeasures.temporal_degree_centrality(Gmag)
    twf = TenetoWorkflow()
    twf.add_node('network_create', 'TemporalNetwork',
                 func_params={'from_array': G})
    twf.add_node('binarize_percent', 'binarize', func_params={
                 'threshold_type': 'percent', 'threshold_level': 0.05})
    twf.add_node('degree_th-percent', 'calc_networkmeasure',
                 func_params={'networkmeasure': 'temporal_degree_centrality'})
    twf.add_node('binarize_magnitude', 'binarize', depends_on='network_create',
                 func_params={'threshold_type': 'magnitude', 'threshold_level': 0})
    twf.add_node('degree_th-magnitude', 'calc_networkmeasure', depends_on='binarize_magnitude',
                 func_params={'networkmeasure': 'temporal_degree_centrality'})
    twf.run()
    twf.make_workflow_figure()
    if not all(Dth == twf.output_['degree_th-percent']):
        raise AssertionError()
    if not all(Dmag == twf.output_['degree_th-magnitude']):
        raise AssertionError()
