"""TenetoBIDS is a class to use Teneto functions with data organized with BIDS (neuroimaging data)."""
import os
import inspect
import json
# import bids
import importlib
import numpy as np
import pandas as pd
import bids
from .. import __path__ as tenetopath
from .. import __version__ as tenetoversion
from ..neuroimagingtools import load_tabular_file, get_sidecar
#from .network import TemporalNetwork

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
    nettsv : str can be nn-t or ijt.
        nn-t means networks are node-node x time.
        ijt means daframs are ijt columns.
    """

    def __init__(self, bids_dir, selected_pipeline, bids_filter=None, bidsvalidator=False,
                 update_pipeline=True, history=None, exist_ok=False, layout=None, nettsv='nn-t'):

        if layout is None:
            self.BIDSLayout = bids.BIDSLayout(bids_dir, derivatives=True, validate=bidsvalidator)
        else:
            self.BIDSLayout = layout
        self.bids_dir = bids_dir
        self.selected_pipeline = selected_pipeline
        self.nettsv = nettsv
        self.bids_filter = {} if bids_filter is None else bids_filter
        if history is not None:
            self.history = {}
        self.exist_ok = exist_ok
        self.update_pipeline = update_pipeline

        with open(tenetopath[0] + '/config/tenetobids/tenetobids_description.json') as f:
            self.tenetobids_description = json.load(f)
        self.tenetobids_description['PipelineDescription']['Version'] = tenetoversion

        with open(tenetopath[0] + '/config/tenetobids/tenetobids_structure.json') as f:
            self.tenetobids_structure = json.load(f)

    # def set_selected_pipeline(self, selected_pipeline):
    #    bids.

    def update_bids_layout(self):
        """
        Function that upddates to new bids l
        """
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

    def run(self, run_func, input_params, output_desc=None, output_pipeline_name=None, bids_filter=None, update_pipeline=True, exist_ok=None, troubleshoot=False):
        """Runs a runction on the selected files.

        Parameters
        ---------------
        run_func : str
            str should correspond to a teneto function.
            So to run the funciton teneto.timeseries.derive_temporalnetwork
            the input should be: 'timeseries.derive_temporalnetwork'
        input_params : dict
            keyword and value pairing of arguments for the function being run.
            The input data to each function will be located automatically.
            This input_params does not need to include the input network.
            For any other input that needs to be loaded loaded within the teneto_bidsstructure
            (communities, events, confounds),
            you can pass the value "bids" if they can be found within the current selected_pipeline.
            If they are found within a different selected_pipeline, type "bids_[selected_pipeline]".
        output_desc : str
            If none, no desc is used (removed any previous file)
            If 'keep', then desc is preserved.
            If any other str, desc is set to that string
        output_pipeline_name : str
            If set, then the data is saved in teneto_[functionname]_[output_pipeline_name].
            If run_func is teneto.timeseries.derive_temporalnetwork and output_pipeline_name
            is jackknife then then the pipeline the data is saved in is
            teneto-generatetemporalnetwork_jackknife
        update_pipeline : bool
            If set to True (default), then the selected_pipeline updates to output of function
        exist_ok : bool
            If set to True, then overwrites direcotry is possible.
        troubleshoot : bool 
            If True, prints out certain information during running.
            Useful to run if reporting a bug.
        """
        if exist_ok is not None:
            self.exist_ok = exist_ok
        # Import teneto if it has not been already
        if 'teneto' not in globals():
            teneto = importlib.import_module('teneto')
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
        if troubleshoot:
            self.troubleshoot('Initial input files', {'input_files': input_files})

        # Check number of required arguments for the function
        funcparams, get_confounds = self._check_run_function_args(func, input_params, functype)

        good_files = bad_files = 0
        for f in input_files:
            f_entities = f.get_entities()
            if get_confounds == 1:
                input_params['confounds'] = self.get_aux_file(f, filetype='confounds')
            data, sidecar = self.load_file(f)
            if troubleshoot:
                self.troubleshoot('Input file name', {'f': f,
                                                    'f_entities': f_entities,
                                                    'sidecar': sidecar})
            if 'sidecar' in dict(funcparams):
                input_params['sidecar'] = sidecar
            if data is None:
                # Skip if data not found
                bad_files += 1
            else:
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
                    output_pattern = '/sub-{subject}/[ses-{session}/]func/sub-{subject}[_ses-{ses}][_run-{run}]_task-{task}[_desc-{desc}]_{suffix}.{extension}'
                    save_name = self.BIDSLayout.build_path(
                        f_entities, path_patterns=output_pattern, validate=False)
                    save_path = self.bids_dir + '/derivatives/' + output_pipeline
                    if troubleshoot:
                        self.troubleshoot('File name consruction', {'f_entities': f_entities,
                                                                    'save_name': save_name,
                                                                    'save_path': save_path})

                    # Exist ok here has to be true, otherwise multiple runs causes an error
                    # Any exist_ok is caught in create pipeline.
                    os.makedirs(
                        '/'.join((save_path + save_name).split('/')[:-1]), exist_ok=True)
                    # Save file
                    # Probably should check the output type in tenetobidsstructure
                    # Table needs column header
                    if isinstance(result, np.ndarray):
                        if len(result.shape) == 3:
                            # Should be made hdf5 at sometime
                            # Idea here is to make 3D array to 2D by concatenating node dimensions.
                            # At reload: to ([np.sqrt(shape[0]), np.sqrt(shape[0]), np.sqrt(shape[1])])
                            shape = result.shape
                            result = result.reshape([shape[0] * shape[1], shape[2]])
                            result = pd.DataFrame(result)
                        elif len(result.shape) == 2:
                            result = pd.DataFrame(result)
                        elif len(result.shape) == 1:
                            result = pd.Series(result)
                        else:
                            raise ValueError(
                                'Output was array with more than 3 dimensions (unexpected)')
                    elif isinstance(result, list):
                        result = pd.DataFrame(result)
                    elif isinstance(result, (int, float)):
                        result = pd.Series(result)
                    if isinstance(result, (pd.DataFrame, pd.Series)):
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
                    if 'sidecar' in input_params:
                        input_params['sidecar'] = 'Loaded automatically via TenetoBIDS'
                    # Loop through input params content and make any nparray input to list for sidecar
                    sidecar['TenetoFunction']['Parameters'] = {}
                    for key, value in input_params.items():
                        if teneto.utils.is_jsonable(value):
                            sidecar['TenetoFunction']['Parameters'][key] = input_params[key]
                        else:
                            if isinstance(input_params[key], np.ndarray):
                                sidecar['TenetoFunction']['Parameters'][key] = input_params[key].tolist()
                            else:
                                print('Warning: Dropping input (' + key + ') from sidecar (not JSONable).')
                elif functype == 'on_sidecar':
                    sidecar = func(**input_params)
                    update_pipeline = False
                    save_path = f.dirname + '/'
                    save_name = f.filename
                # Save sidecar
                with open(save_path + save_name.replace('.tsv', '.json'), 'w') as f:
                    json.dump(sidecar, f)
                good_files += 1
        report = '## ' + run_func + '\n'
        report += str(good_files) + ' files were included (' + \
            str(bad_files) + ' excluded from run)'
        self.report = report

        if update_pipeline:
            if functype == 'on_data':
                self.selected_pipeline = output_pipeline
            # Create new bids_filter dictionary that only contains
            # sub/ses/run/task as other tags are dropped.
            bids_filter = dict(self.bids_filter)
            self.bids_filter = {}
            bids_filters_allowed = ['subject', 'ses', 'run', 'task']
            [self.update_bids_filter({f: bids_filter[f]})
            for f in bids_filters_allowed
            if f in bids_filter]

        self.update_bids_layout()

    def _check_run_function_args(self, func, input_params, functype):
        """
        Helper function for TenetoBIDS.run. 
        
        Function checks that the input parametes match the function.

        Returns
        ========
        funcparams : dict 
            parameters of the input function
        get_confounds : bool
            1 if confound files need to be loaded.
        """
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
        expected_arg_defecit = 1
        if 'sidecar' in dict(funcparams) and functype == 'on_data':
            expected_arg_defecit += 1
        # Calculate the different betwee n required and input arguments
        arg_diff = required_args - input_args
        if arg_diff != expected_arg_defecit:
            # Three conditoinals to be met in order to get confounds
            confounds_not_input = 'confounds' not in input_params
            confounds_in_func = 'confounds' in  dict(funcparams)
            arg_needed = arg_diff == expected_arg_defecit + 1
            if confounds_not_input and confounds_in_func and arg_needed:
                # Get confounds automatically
                get_confounds = 1
            else:
                raise ValueError(
                    'Expecting one unspecified input argument.\
                    Enter all required input arguments in input_params except for the data files.')
        return funcparams, get_confounds


    def get_selected_files(self, output=None):
        """
        Uses information in selected_pipeline and the bids layout and shows the files that will be processed when calling TenetoBIDS.run().

        If you specify a particular output, it will tell you which files will get selected for that output
        """
        if output is not None:
            filters = self.tenetobids_structure[output]['input']
        else:
            # input can only be these files
            filters = {'extension': ['.tsv', '.nii', '.nii.gz']}
        # Add predefined filters to the check
        filters.update(self.bids_filter)
        return self.BIDSLayout.get(scope=self.selected_pipeline, **filters)

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
            funcs = sorted(list(set(funcs_filter)))
        return ', '.join(funcs)

    def update_bids_filter(self, filter_addons):
        """Updates TenetoBIDS.bids_filter

        Parameters
        ==========
        filter_addons : dict
            dictionary that updates TenetoBIDS.bids_filter
        """
        self.bids_filter.update(filter_addons)

    def get_aux_file(self, bidsfile, filetype='confounds'):
        """Tries to automatically get auxiliary data for input files, and loads it

        Paramters
        ==========
        bidsfile : BIDSDataFile or BIDSImageFile
            The BIDS file that the confound file is gong to be matched.
        filetype : string
            Can be confounds, events. 
            Specified if you want to get the confound or events data.
        """
        if filetype == 'confounds':
            suffix = 'regressors'
        elif filetype == 'events': 
            suffix = 'events'
        else:
            raise ValueError('unknown file type')
        # Get the entities of the filename
        file_entities = bidsfile.get_entities()
        # Ensure that the extension and suffix are correct
        file_entities['suffix'] = suffix
        file_entities['extension'] = '.tsv'
        if 'desc' in file_entities:
            file_entities.pop('desc')
        auxfile = self.BIDSLayout.get(**file_entities)
        if len(auxfile) == 0:
            raise ValueError('Non auxiliary file (type: ' + filetype + ') found')
        elif len(auxfile) > 1:
            raise ValueError('More than one auxiliary file (type: ' + filetype + ') found')
        # Load the aux file
        aux = load_tabular_file(
            auxfile[0].dirname + '/' + auxfile[0].filename, index_col=False)
        return aux

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
        # Since temporal networks are currently saved in 2D collapsed arrays
        # The following checks if they should be resized, and resizes
        if '_temporalconnectivity.tsv' in bidsfile.filename:
            dimord = sidecar['TenetoFunction']['Parameters']['params']['dimord']
            if (self.nettsv == 'nn-t' or dimord == 'node,time'):
                n_nodes = int(np.sqrt(data.shape[0]))
                n_time = data.shape[1]
                data = data.values.reshape([n_nodes, n_nodes, n_time])
                print(data.shape)
        return data, sidecar

    def troubleshoot(self, stepname, status):
        """
        Prints ongoing info to assist with troubleshooting
        """
        print('******** TROUBLESHOOT STEP: ' + stepname + ', start ********')
        for step in status:
            print('++++++++')
            print(step)
            print('------')
            print(status[step])
            print('++++++++')
        print('******** TROUBLESHOOT STEP: ' + stepname + ', end ********')

    def load_events(self):
        """
        Loads event data for selected files
        """
        input_files = self.get_selected_files()
        events = {}
        for f in input_files:
            events[f.filename] = self.get_aux_file(f, filetype='events')
        return events
