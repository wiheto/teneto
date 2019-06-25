import templateflow.api as tf
from niworkflows.interfaces.images import SignalExtraction
from .bidsutils import load_tabular_file
from nilearn.input_data import NiftiLabelsMasker

def make_parcellation(data_path, atlas, template='MNI152NLin2009cAsym', atlas_desc=None, resolution=2, parc_params=None, return_meta=False):
    """
    Performs a parcellation which reduces voxel space to regions of interest (brain data).

    Parameters
    ----------

    data_path : str
        Path to .nii image.
    parcellation : str
        Specify which parcellation that you would like to use. For MNI: 'gordon2014_333', 'power2012_264', For TAL: 'shen2013_278'.
        It is possible to add the OH subcotical atlas on top of a cortical atlas (e.g. gordon) by adding:
            '+OH' (for oxford harvard subcortical atlas) and '+SUIT' for SUIT cerebellar atlas.
            e.g.: gordon2014_333+OH+SUIT'
    parc_type : str
        Can be 'sphere' or 'region'. If nothing is specified, the default for that parcellation will be used.
    parc_params : dict
        **kwargs for nilearn functions

    Returns
    -------

    data : array
        Data after the parcellation.

    NOTE
    ----
    These functions make use of nilearn. Please cite templateflow and nilearn if used in a publicaiton.
    """

    if not parc_params:
        parc_params = {}

    tf_get_params = {
        'template': template,
        'resolution': resolution,
        'atlas': atlas
    }
    if atlas_desc is not None: 
        tf_get_params['desc'] = atlas_desc
    file = tf.get(**tf_get_params, extensions='nii.gz')

    if isinstance(file, list): 
        raise ValueError('More than one template file found. Specify the type of file you need (often atlas_desc). Run: templateflow.api.TF_LAYOUT.get_descs(atlas=' + atlas + ') to see available desc for atlas')

    region = NiftiLabelsMasker(str(file), **parc_params)
    data = region.fit_transform(data_path)

    if return_meta: 
        meta_info = tf.get(template=template, atlas=atlas, desc=atlas_desc, extensions='tsv')
        meta_info = load_tabular_file(str(meta_info))
        return data, meta_info
    else: 
        return data