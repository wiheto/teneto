import templateflow.api as tf
from .bidsutils import load_tabular_file
from nilearn.input_data import NiftiLabelsMasker
import pandas as pd

atlas='Schaefer2018'
atlas_desc='100Parcels17Networks' 
return_meta=True
def make_parcellation(data_path, atlas, template='MNI152NLin2009cAsym', atlas_desc=None, resolution=2, parc_params=None, return_meta=False):
    """
    Performs a parcellation which reduces voxel space to regions of interest (brain data).

    Parameters
    ----------

    data_path : str
        Path to .nii image.
    atlas : str
        Specify which atlas you want to use (see github.com/templateflow/)
    template : str
        What space you data is in. If fmriprep, leave as MNI152NLin2009cAsym.
    atlas_desc : str
        Specify which description of atlas.
    resolution : int
        Resolution of atlas. Can be 1 or 2.
    parc_params : dict
        **kwargs for nilearn functions.
    return_meta : bool
        If true, tries to return any meta-information that exists about parcellation.

    Returns
    -------

    data : dataframe
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
    file = tf.get(**tf_get_params, extension='nii.gz')

    if isinstance(file, list):
        raise ValueError('More than one template file found. Specify the type of file you need (often atlas_desc). Run: templateflow.api.TF_LAYOUT.get_descs(atlas=' +
                         atlas + ') to see available desc for atlas')

    region = NiftiLabelsMasker(str(file), **parc_params)
    data = region.fit_transform(data_path).transpose()
    data = pd.DataFrame(data=data)
    meta_info = tf.get(template=template, atlas=atlas,
                        desc=atlas_desc, extension='tsv')
    if len(str(meta_info)) > 0: 
        meta_info = load_tabular_file(str(meta_info))
        data.index = meta_info['name'].values
    if return_meta:
        return data, meta_info
    else:
        return data
