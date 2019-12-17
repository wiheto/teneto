import os
import bids
import numpy as np
import inspect
import json
from ..neuroimagingtools import load_tabular_file, get_sidecar
import pandas as pd
from .network import TemporalNetwork
from teneto import __path__ as tenetopath
from teneto import __version__ as tenetoversion
import teneto


class TenetoBIDS:
    """
    Class for analysing data in BIDS.

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

    bids_filter : dict
    history : bool
    update_pipeline : bool
        If true, the output_pipeline becomes the new selected_pipeline
    exist_ok : bool
        If False, will raise an error if the output directory already exist_ok.
        If True, will not raise an error.
        This can lead to files being overwritten, if desc is not set.
    """

    def __init__(self, bids_dir, selected_pipeline, bids_filter=None, bidsvalidator=False,
                 update_pipeline=True, history=None, exist_ok=False, layout=None):
        if layout is None:
            self.BIDSLayout = bids.BIDSLayout(bids_dir, derivatives=True)
        else:
            self.BIDSLayout = layout
        self.bids_dir = bids_dir
        self.selected_pipeline = selected_pipeline
        if bids_filter is None:
            self.bids_filter = {}
        else:
            self.bids_filter = bids_filter
        if history is not None:
            self.history = {}
        self.exist_ok = exist_ok

        with open(tenetopath[0] + '/config/tenetobids/tenetobids_description.json') as f:
            self.tenetobids_description = json.load(f)
        self.tenetobids_description['PipelineDescription']['Version'] = tenetoversion

        with open(tenetopath[0] + '/config/tenetobids/tenetobids_structure.json') as f:
            self.tenetobids_structure = json.load(f)

    # def set_selected_pipeline(self, selected_pipeline):
    #    bids.

    def update_bids_layout(self):
        self.BIDSLayout = bids.BIDSLayout(self.bids_dir, derivatives=True)

    def create_output_pipeline(self, runc_func, output_pipeline_name, exist_ok=None):
        """Creates the directories of the saved file.

        Parameters
        ----------
        output_pipeline : str
            name of output pipeline
        exist_ok : bool
            If False, will raise error if pipeline already exist_ok.
            If True, will not raise an error.
            This can lead to files being overwritten, if desc is not set.
            If None, will use the exist_ok set during init.

        Returns
        -------
        Creates the output pipeline directory in:
            bids_dir/teneto-[output_pipeline]/

        """
        if exist_ok is not None:
            self.exist_ok = exist_ok
        output_pipeline = 'teneto-'
        output_pipeline += runc_func.split('.')[-1]
        output_pipeline = output_pipeline.replace('_', '-')
        if output_pipeline_name is not None:
            output_pipeline += '_' + output_pipeline_name
        output_pipeline_path = self.bids_dir + '/derivatives/' + output_pipeline
        if os.path.exists(output_pipeline_path) and not self.exist_ok:
            raise ValueError(
                'Output_pipeline already exists. Set exist_ok to True if this is desired behaviour.')
        os.makedirs(output_pipeline_path, exist_ok=self.exist_ok)
        # Initiate with dataset_description
        datainfo = self.tenetobids_description.copy()
        datainfo['PipelineDescription']['Name'] = output_pipeline
        with open(output_pipeline_path + '/dataset_description.json', 'w') as fs:
            json.dump(datainfo, fs)
        self.update_bids_layout()
        return output_pipeline

    def run(self, run_func, input_params, output_desc=None, output_pipeline_name=None, bids_filter=None, update_pipeline=True, exist_ok=None):
        """Runs a runction on the selected files.

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
        exist_ok : bool
        """
        if exist_ok is not None:
            self.exist_ok = exist_ok

        func = teneto
        for f in self.tenetobids_structure[run_func]['module'].split('.'):
            func = getattr(func, f)
        functype = self.tenetobids_structure[run_func]['functype']
        func = getattr(func, run_func)

        # Only set up an output pipeline if the functype is ondata
        if functype == 'on_data':
            output_pipeline = self.create_output_pipeline(
                run_func, output_pipeline_name, self.exist_ok)

        input_files = self.get_selected_files(run_func.split('.')[-1])
        if not input_files:
            raise ValueError('No input files')

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
        matched_input_arguments_defecit = 1
        if 'sidecar' in dict(funcparams) and functype == 'on_data':
            matched_input_arguments_defecit += 1
        if required_args - input_args != matched_input_arguments_defecit:
            if 'confounds' not in input_params and 'confounds' in dict(funcparams) and required_args - input_args == matched_input_arguments_defecit + 1:
                # Get confounds automatically
                get_confounds = 1
            else:
                raise ValueError(
                    'Expecting one unspecified input argument. Enter all required input arguments in input_params except for the data files.')
        gf = bf = 0
        for f in input_files:
            f_entities = f.get_entities()
            if get_confounds == 1:
                input_params['confounds'] = self.get_confounds(f)
            data, sidecar = self.load_file(f)
            if 'sidecar' in dict(funcparams):
                input_params['sidecar'] = sidecar
            if data is not None:
                if functype == 'on_data':
                    result = func(data, **input_params)
                    # if sidecar is in input_params, then sidecar is also returned
                    if 'sidecar' in dict(funcparams):
                        result, sidecar = result
                        # if output_desc is None, then keep desc
                    if output_desc is None and 'desc' in f_entities:
                        f_entities.pop('desc')
                    elif output_desc == 'keep':
                        pass
                    elif output_desc is not None:
                        f_entities['desc'] = output_desc
                    f_entities.update(
                        self.tenetobids_structure[run_func.split('.')[-1]]['output'])
                    output_pattern = '/sub-{subject}/[ses-{ses}/]func/sub-{subject}[_ses-{ses}][_run-{run}]_task-{task}[_desc-{desc}]_{suffix}.{extension}'
                    save_name = self.BIDSLayout.build_path(
                        f_entities, path_patterns=output_pattern, validate=False)
                    save_path = self.bids_dir + '/derivatives/' + output_pipeline
                    # Exist ok here has to be true, otherwise multiple runs causes an error
                    # Any exist_ok is caught in create pipeline.
                    os.makedirs(
                        '/'.join((save_path + save_name).split('/')[:-1]), exist_ok=True)
                    # Save file
                    # Probably should check the output type in tenetobidsstructure
                    # Table needs column header
                    if isinstance(result, np.ndarray):
                        if len(result.shape) == 3:
                            # THIS CAN BE MADE TO A DENSE HDF5
                            result = TemporalNetwork(
                                from_array=result, forcesparse=True).network
                        elif len(result.shape) == 2:
                            result = pd.DataFrame(result)
                        elif len(result.shape) == 1:
                            result = pd.Series(result)
                        else:
                            raise ValueError(
                                'Output was array with more than 3 dimensions (unexpected)')
                    elif isinstance(result, list):
                        result = pd.DataFrame(result)
                    elif isinstance(result, int) or isinstance(result, float):
                        result = pd.Series(result)
                    if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                        result.to_csv(save_path + save_name,
                                      sep='\t', header=True)
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
                    if 'sidecar' in input_params:
                        input_params['sidecar'] = 'Loaded automatically via TenetoBIDS'
                    sidecar['TenetoFunction']['Parameters'] = input_params
                elif functype == 'on_sidecar':
                    sidecar = func(**input_params)
                    update_pipeline = False
                    save_path = f.dirname + '/'
                    save_name = f.filename
                # Save sidecar
                with open(save_path + save_name.replace('.tsv', '.json'), 'w') as f:
                    json.dump(sidecar, f)
                gf += 1
            else:
                bf += 1

        report = '## ' + run_func + '\n'
        report += str(gf) + ' files were included (' + \
            str(bf) + ' excluded from run)'
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
        # Add predefined filters to the check
        filters.update(self.bids_filter)
        files = self.BIDSLayout.get(scope=self.selected_pipeline, **filters)
        return files

    def get_run_options(self, for_selected=True):
        """Returns the different function names that can be called using TenetoBIDS.run()
        
        Parameters
        ===========
        for_selected : bool
            If True, only return run options for the selected files.
            If False, returns all options. 
        
        Returns
        ========
        options : str
            a list of options that can be run.
        """
        funcs = self.tenetobids_structure.keys()
        if for_selected:
            funcs_filter = []
            files = self.get_selected_files()
            suffix = [f.get_entities()['suffix'] for f in files]
            suffix = list(np.unique(suffix))
            for t in list(funcs):
                s = self.tenetobids_structure[t]['input']['suffix']
                if isinstance(s, str): 
                    s = [s]
                for su in suffix:
                    if su in s:
                        funcs_filter.append(t)
            funcs = list(set(funcs_filter))
        return ', '.join(funcs)

    def update_bids_filter(self, filter_addons):
        """Updates TenetoBIDS.bids_filter

        Parameters
        ==========
        filter_addons : dict
            dictionary that updates TenetoBIDS.bids_filter
        """ 
        self.bids_filter.update(filter_addons)

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
            confoundsfile[0].dirname + '/' + confoundsfile[0].filename, index_col=False)
        return confounds

    def load_data(self, bids_filter=None):
        """Returns data, default is the input data.

        bids_filter : dict
            default is None. If set, load data will load all files found by the bids_filter.
            Any preset BIDS filter is used as well, but will get overwritten by this input.
        """
        if bids_filter is None:
            files = self.get_selected_files()
        else:
            filters = dict(self.bids_filter)
            filters.update(bids_filter)
            files = self.BIDSLayout.get(**filters)
        data = {}
        for f in files:
            if f.filename in data:
                raise ValueError('Same name appears twice in selected files')
            data[f.filename], _ = self.load_file(f)
        return data

    def load_file(self, bidsfile):
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
