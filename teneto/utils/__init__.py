"""Many helper functions for Teneto"""

from .utils import graphlet2contact, contact2graphlet,\
    binarize_percent, binarize_rdp, binarize_magnitude,\
    binarize, set_diagonal, gen_nettype, \
    get_distance_function, \
    clean_community_indexes, multiple_contacts_get_values,\
    df_to_array, check_distance_funciton_input,\
    get_dimord, get_network_when,\
    create_supraadjacency_matrix,\
    df_drop_ij_duplicates, is_jsonable,\
    process_input, check_input
from .io import tnet_to_nx
__all__ = ['graphlet2contact', 'contact2graphlet',
           'binarize_percent', 'binarize_rdp', 'binarize_magnitude',
           'binarize', 'set_diagonal', 'gen_nettype', 'check_input',
           'get_distance_function', 'process_input',
           'clean_community_indexes', 'multiple_contacts_get_values',
           'df_to_array', 'check_distance_funciton_input',
           'get_dimord', 'get_network_when',
           'create_supraadjacency_matrix',
           'df_drop_ij_duplicates', 'tnet_to_nx', 'is_jsonable']
