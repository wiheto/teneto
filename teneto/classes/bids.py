import teneto
import os
import bids
import numpy as np
import inspect
import json
from teneto.neuroimagingtools import load_tabular_file, get_bids_tag, get_sidecar, confound_matching, process_exclusion_criteria, drop_bids_suffix, make_directories
import pandas as pd
from . import TemporalNetwork


class TenetoBIDS:
    """Class for analysing data in BIDS.

    TenetoBIDS allows for an analysis to be performed across a dataset.
    All different functions from Teneto can be applied to all files in a dataset organized in BIDS.
    Data should be first preprocessed (e.g. fMRIPrep).

    Parameters
    ----------

    bids_dir : str
        string to BIDS directory
    selected_pipeline : str or dict
        the directory that is in the bids_dir/derivatives/<selected_pipeline>/.
        This fine will be used as the input to any teneto function (first argument).
        If multiple inputs are required for a function, then you can specify:
            {'netin': 'tvc',
            'communities': 'coms'}
        With this, the input for netin with be from bids_dir/derivatives/[teneto-]tvc/,
        and the input for communities will be from bids_dir/derivatives/[teneto-]coms/.
        The keys in this dictionary must match the names of the teneto funciton inputs.

    bids_filters : dict
    history : bool
    update_pipeline : bool
        If true, the output_pipeline becomes the new selected_pipeline
    overwrite : bool
        If False, will not overwrite existing directories
    """

    with open(teneto.__path__[0] + '/config/tenetobids/tenetobids_description.json') as f:
        tenetobids_description = json.load(f)
    tenetobids_description['PipelineDescription']['Version'] = teneto.__version__

    with open(teneto.__path__[0] + '/config/tenetobids/tenetobids_structure.json') as f:
        tenetobids_structure = json.load(f)

    def __init__(self, bids_dir, selected_pipeline, bids_filters=None, bidsvalidator=False,
                 update_pipeline=True, history=None, overwrite=False, layout=None):
        if layout is None:
            self.BIDSLayout = bids.BIDSLayout(bids_dir, derivatives=True)
        else:
            self.BIDSLayout = layout
        self.bids_dir = bids_dir
        self.selected_pipeline = selected_pipeline
        if bids_filters is None:
            self.bids_filters = {}
        else:
            self.bids_filters = bids_filters
        if history is not None:
            self.history = {}
        self.overwrite = overwrite

    # def set_selected_pipeline(self, selected_pipeline):
    #    bids.

    def update_bids_layout(self):
        self.BIDSLayout = bids.BIDSLayout(self.bids_dir, derivatives=True)

    def create_output_pipeline(self, runc_func, output_pipeline_name, overwrite=None):
        """
        Parameters
        ----------
        output_pipeline : str
            name of output pipeline
        overwrite : bool


        Returns
        -------
        Creates the output pipeline directory in:
            bids_dir/teneto-[output_pipeline]/

        """
        if overwrite is not None:
            self.overwrite = overwrite
        output_pipeline = 'teneto-'
        output_pipeline += runc_func.split('.')[-1]
        output_pipeline = output_pipeline.replace('_', '-')
        if output_pipeline_name is not None:
            output_pipeline += '_' + output_pipeline_name
        output_pipeline_path = self.bids_dir + '/derivatives/' + output_pipeline
        if os.path.exists(output_pipeline_path) and self.overwrite == False:
            raise ValueError(
                'output_pipeline already exists and overwrite is set to False.')
        os.makedirs(output_pipeline_path, exist_ok=self.overwrite)
        # Initiate with dataset_description
        datainfo = self.tenetobids_description.copy()
        datainfo['PipelineDescription']['Name'] = output_pipeline
        with open(output_pipeline_path + '/dataset_description.json', 'w') as fs:
            json.dump(datainfo, fs)
        self.update_bids_layout()
        return output_pipeline

    def run(self, run_func, input_params, output_desc=None, output_pipeline_name=None, bids_filters=None, update_pipeline=True, overwrite=None):
        """
        Parameters
        ---------------
        run_func : str
            str should correspond to a teneto function. So to run the funciton teneto.timeseries.derive_temporalnetwork
            the input should be: 'timeseries.derive_temporalnetwork'
        input_params : dict
            keyword and value pairing of arguments.
            The input data to each function will be located automatically and should not be included.
            For any other input that needs to be loaded loaded within the teneto_bidsstructure (communities, events, confounds),
            you can pass the value "bids" if they can be found within the current selected_pipeline.
            If they are found within a different selected_pipeline, type "bids_[selected_pipeline]".
        output_desc : str
            If none, no desc is used (removed any previous file)
            If 'keep', then desc is preserved.
            If any other str, desc is set to that string
        output_pipeline_name : str
            If set, then the data is saved in teneto_[functionname]_[output_pipeline_name]. If run_func is
            teneto.timeseries.derive_temporalnetwork and output_pipeline_name is jackknife
            then then the pipeline the data is saved in is
            teneto-generatetemporalnetwork_jackknife
        update_pipeline : bool
        overwrite : bool
        """
        if overwrite is not None:
            self.overwrite = overwrite
        output_pipeline = self.create_output_pipeline(
            run_func, output_pipeline_name, self.overwrite)

        input_files = self.get_selected_files(run_func.split('.')[-1])
        if not input_files:
            raise ValueError('No input files')

        func = teneto
        for f in self.tenetobids_structure[run_func]['module'].split('.'):
            func = getattr(func, f)
        func = getattr(func, run_func)
        # Check number of required arguments for the folder
        sig = inspect.signature(func)
        funcparams = sig.parameters.items()
        required_args = 0
        input_args = 0
        for p_name, p in funcparams:
            if p.default == inspect._empty:
                required_args += 1
                if p_name in input_params:
                    input_args += 1
        get_confounds = 0
        if required_args - input_args != 1:
            if 'confounds' not in input_params and 'confounds' in dict(funcparams) and required_args == 2:
                # Get confounds automatically
                get_confounds = 1
            else:
                raise ValueError(
                    'Expecting one unspecified input argument. Enter all required input arguments in input_params except for the data files.')
        for f in input_files:
            gf = bf = 0
            if get_confounds == 1:
                input_params['confounds'] = self.get_confounds(f)
            data, sidecar = self._load_file(f)
            if data is not None:
                result = func(data, **input_params)
                f_entities = f.get_entities()
                if output_desc is None and 'desc' in f_entities:
                    f_entities.pop('desc')
                elif output_desc == 'keep':
                    pass
                elif output_desc is not None:
                    f_entities['desc'] = output_desc
                f_entities.update(
                    self.tenetobids_structure[run_func.split('.')[-1]]['output'])
                output_pattern = '/sub-{subject}/[ses-{ses}/]func/sub-{subject}[_ses-{ses}][_run-{run}]_task-{task}[_desc-{desc}]_{suffix}.{extension}'
                save_name = tnet.BIDSLayout.build_path(
                    f_entities, path_patterns=output_pattern, validate=False)
                save_path = self.bids_dir + '/derivatives/' + output_pipeline
                os.makedirs(
                    '/'.join((save_path + save_name).split('/')[:-1]), exist_ok=self.overwrite)
                # Save file
                # Probably should check the output type in tenetobidsstructure
                # Table needs column header
                if type(result) is np.ndarray:
                    if len(result.shape) == 3:
                        # THIS CAN BE MADE TO A DENSE HDF5
                        result = teneto.TemporalNetwork(
                            from_array=result, forcesparse=True).network
                    elif len(result.shape) == 2:
                        result = pd.DataFrame(result)
                    elif len(result.shape) == 1:
                        result = pd.Series(result)
                    else:
                        raise ValueError(
                            'Output was array with more than 3 dimensions (unexpected)')
                elif type(result) is list:
                    result = pd.DataFrame(result)
                elif type(result) is int:
                    result = pd.Series(result)
                elif isinstance(result, float):
                    result = pd.Series(result)
                if type(result) is pd.DataFrame or isinstance(result, pd.Series):
                    result.to_csv(save_path + save_name, sep='\t', header=True)
                else:
                    raise ValueError('Unexpected output type')
                # add information to sidecar
                sidecar['DerivativeSource'] = f.path
                sidecar['TenetoFunction'] = {}
                sidecar['TenetoFunction']['Name'] = run_func
                # For aux_input more is needed here too.
                if get_confounds == 1:
                    input_params['confounds'] = 'Loaded automatically via TenetoBIDS'
                elif 'confounds' in input_params:
                    input_params['confounds'] = 'Passed as argument'

                sidecar['TenetoFunction']['Parameters'] = input_params
                # Save sidecar
                with open(save_path + save_name.replace('.tsv', '.json'), 'w') as f:
                    json.dump(sidecar, f)
                gf += 1
            else:
                bf += 1

        report = '## ' + run_func + '\n'
        report += str(gf) + ' files were included (' + str(bf) + ' excluded)'
        self.report = report

        if update_pipeline:
            self.selected_pipeline = output_pipeline
        self.update_bids_layout()

    def get_selected_files(self, output=None):
        """
        Uses information in selected_pipeline and the bids layout and shows the files that will be processed when calling TenetoBIDS.run().

        If you specify a particular output, it will tell you which files will get selected for that output
        """
        if output is not None:
            filters = self.tenetobids_structure[output]['input']
        else:
            # input can only be these files
            filters = {'extension': ['tsv', 'nii', 'nii.gz']}
        # Add predefined filters to te check
        filters.update(self.bids_filters)
        return self.BIDSLayout.get(scope=self.selected_pipeline, **filters)

    def get_run_options(self):
        """Returns the different function names that can be called using TenetoBIDS.run()"""
        funcs = self.tenetobids_structure.keys()
        return ', '.join(funcs)

    def get_confounds(self, bidsfile, confound_filters=None):
        """Tries to automatically get the confounds file of an input file, and loads it

        Paramters
        ==========
        bidsfile : BIDSDataFile or BIDSImageFile
            The BIDS file that the confound file is gong to be matched.
        """
        if confound_filters is None:
            confound_filters = {}
        # Get the entities of the filename
        file_entities = bidsfile.get_entities()
        # Ensure that the extension and suffix are correct
        file_entities['suffix'] = 'regressors'
        file_entities['extension'] = 'tsv'
        if 'desc' in file_entities:
            file_entities.pop('desc')
        confoundsfile = self.BIDSLayout.get(**file_entities)
        if len(confoundsfile) == 0:
            raise ValueError('Non confounds found')
        elif len(confoundsfile) > 1:
            raise ValueError('More than one confounds file found')
        # Load the confounds file
        confounds = load_tabular_file(
            confoundsfile[0].dirname + '/' + confoundsfile[0].filename)
        return confounds

    def load_data(self, bids_filters=None):
        """Returns data, default is the input data.

        bids_filters : dict
            default is None. If set, load data will load all files found by the bids_filters.
            Otherwise, tnet.get_selected_files is loaded.
            Note, this can select files outside of input pipeline.
        """
        if bids_filters is None:
            files = self.get_selected_files()
        else:
            files = tnet.BIDSLayout.get(**bids_filters)
        data = {}
        for f in files:
            if f.filename in data:
                raise ValueError('Same name appears twice in selected files')
            data[f.filename], _ = self._load_file(f)
        return data

    def _load_file(self, bidsfile):
        """Aux function to load the data and sidecar from a BIDSFile

        Paramters
        ==========
        bidsfile : BIDSDataFile or BIDSImageFile
            The BIDS file that the confound file is gong to be matched.

        """
        # Get sidecar and see if file has been rejected at a previous step
        # (note sidecar could be called in input_files, but this will require loading sidecar twice)
        sidecar = get_sidecar(bidsfile.dirname + '/' + bidsfile.filename)
        if not sidecar['BadFile']:
            if hasattr(bidsfile, 'get_image'):
                data = bidsfile.get_image()
            elif hasattr(bidsfile, 'get_df'):
                # This can be changed if/when pybids is updated. Assumes index_col=0 in tsv file
                data = load_tabular_file(
                    bidsfile.dirname + '/' + bidsfile.filename)
        else:
            data = None
        return data, sidecar
