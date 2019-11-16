"""Many helper functions for Teneto"""

from .bidsutils import make_directories, drop_bids_suffix, \
    get_bids_tag, load_tabular_file, \
    get_sidecar, confound_matching, \
    process_exclusion_criteria
from .utils import graphlet2contact, contact2graphlet,\
    binarize_percent, binarize_rdp, binarize_magnitude,\
    binarize, set_diagonal, gen_nettype, check_input,\
    get_distance_function, process_input,\
    clean_community_indexes, multiple_contacts_get_values,\
    df_to_array, check_distance_funciton_input,\
    create_traj_ranges, get_dimord, get_network_when,\
    create_supraadjacency_matrix, check_TemporalNetwork_input,\
    df_drop_ij_duplicates
from .io import tnet_to_nx
from .fmriutils import make_parcellation
__all__ = ['make_parcellation', 'tnet_to_nx', 'make_directories', 'drop_bids_suffix',
           'get_bids_tag', 'load_tabular_file',
           'get_sidecar', 'confound_matching',
           'process_exclusion_criteria',
           'graphlet2contact', 'contact2graphlet',
           'binarize_percent', 'binarize_rdp', 'binarize_magnitude',
           'binarize', 'set_diagonal', 'gen_nettype', 'check_input',
           'get_distance_function', 'process_input',
           'clean_community_indexes', 'multiple_contacts_get_values',
           'df_to_array', 'check_distance_funciton_input',
           'create_traj_ranges', 'get_dimord', 'get_network_when',
           'create_supraadjacency_matrix', 'check_TemporalNetwork_input',
           'df_drop_ij_duplicates']
