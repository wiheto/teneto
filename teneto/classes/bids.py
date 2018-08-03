import itertools
import teneto
import os
import re
from bids.grabbids import BIDSLayout
import numpy as np
import inspect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import traceback
import nilearn
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d
import time 

#class NetworkMeasures:
#    def __init__(self,**kwargs):
#        pass

    #def temporal_degree_centrality(self,**kwargs):
    #    print(self)
    #    print(teneto.networkmeasures.temporal_degree_centrality(self,**kwargs))


class TenetoBIDS:

    #networkmeasures = NetworkMeasures(self)

    def __init__(self, BIDS_dir, pipeline=None, pipeline_subdir=None, parcellation=None, space=None, subjects='all', sessions='all', runs='all', tasks='all', last_analysis_step=None, analysis_steps=None, bad_subjects=None, confound_pipeline=None, raw_data_exists=True, njobs=None):
        """
        Parameters
        ----------

        BIDS_dir : str
            string to BIDS directory
        pipeline : str
            the directory that is in the BIDS_dir/derivatives/<pipeline>/
        pipeline_subdir : str, optional
            the directory that is in the BIDS_dir/derivatives/<pipeline>/sub-<subjectnr/func/ses-<sesnr>/<pipeline_subdir>
        parcellation : str, optional
            parcellation name
        space : str, optional
            different nomralized spaces
        subjects : str or list, optional
            can be part of the BIDS file name
        sessions : str or list, optional
            can be part of the BIDS file name
        runs : str or list, optional
            can be part of the BIDS file name
        tasks : str or list, optional
            can be part of the BIDS file name
        analysis_steps : str or list, optional
            any tags that exist in the filename (e.g. 'bold' or 'preproc')
        bad_subjects : list or str, optional
            Removes these subjects from the analysis
        confound_pipeline : str, optional
            If the confounds file is in another derivatives directory than the pipeline directory, set it here.
        raw_data_exists : bool, optional
            Default is True. If the unpreprocessed data is not present in BIDS_dir, set to False. Some BIDS funcitonality will be lost.
        njobs : int, optional
            How many parallel jobs to run. Default: 1. The set value can be overruled in individual functions.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)

        self.contact = []
        if raw_data_exists:
            self.BIDS = BIDSLayout(BIDS_dir)
        else:
            self.BIDS = 'Raw data was flagged as not present in directory structure.'

        self.BIDS_dir = os.path.abspath(BIDS_dir)
        self.pipeline = pipeline
        self.confound_pipeline = confound_pipeline
        self.raw_data_exists = raw_data_exists
        if not pipeline_subdir:
            self.pipeline_subdir = ''
        else:
            self.pipeline_subdir = pipeline_subdir
        self.parcellation = parcellation
        self.space = space
        if self.BIDS_dir[-1] != '/':
            self.BIDS_dir = self.BIDS_dir + '/'
        if subjects == 'all':
            if self.raw_data_exists:
                self.subjects = self.BIDS.get_subjects()
            else:
                self.set_subjects()
        else: 
            self.set_subjects(subjects)
        if sessions == 'all' and self.raw_data_exists:
            self.sessions = self.BIDS.get_sessions()
        elif self.raw_data_exists:
            self.set_sessions(sessions)
        else:
            self.sessions = []
        if tasks == 'all' and self.raw_data_exists:
            self.tasks = self.BIDS.get_tasks()
        elif tasks != 'all':
            self.set_tasks(tasks)
        else:
            self.tasks = []
        if runs == 'all' and self.raw_data_exists:
            self.runs = self.BIDS.get_runs()
        elif runs != 'all':
            self.set_runs(runs)
        else:
            self.runs = []
        if not last_analysis_step:
            self.last_analysis_step = ''
        else:
            self.last_analysis_step = last_analysis_step
        if isinstance(analysis_steps,str):
            self.analysis_steps = [analysis_steps]
        elif isinstance(analysis_steps,list):
            self.analysis_steps = analysis_steps
        else:
            self.analysis_steps = []

        if bad_subjects == None:
            self.bad_subjects = None
        else:
            self.set_bad_subjects(bad_subjects)

        if not njobs:
            self.njobs = 1
        else:
            self.njobs = njobs
        self.bad_files = []
        self.confounds = None

    def add_history(self, fname, fargs, init=0):
        """
        Adds a processing step to TenetoBIDS.history.
        """
        if init == 1:
            self.history = []
        self.history.append([fname,fargs])

    def derive(self, params, update_pipeline=True, tag=None, njobs=1, confound_corr_report=True):

        """
        Derive time-varying connectivity on the selected files.

        Parameters
        ----------
        params : dict.
            See teneto.derive.derive for the structure of the param dictionary.

        update_pipeline : bool
            If true, the object updates the selected files with those derived here.

        njobs : int
            How many parallel jobs to run

        confound_corr_report : bool 
            If true, histograms and summary statistics of TVC and confounds are plotted in a report directory. 

        tag : str 
            any additional tag that will be placed in the saved file name. 

        Returns 
        ------- 
        dfc : files 
            saved in .../derivatives/teneto/sub-xxx/tvc/..._tvc.npy
        """
        if not njobs:
            njobs = self.njobs
        self.add_history(inspect.stack()[0][3], locals(), 1)

        files = self.get_selected_files(quiet=1)
        confound_files = self.get_confound_files(quiet=1)
        if confound_files:
            confounds_exist = True
        else: 
            confounds_exist = False
        if not confound_corr_report: 
            confounds_exist = False 

        if not tag:
            tag = ''
        else:   
            tag = '_' + tag

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_derive,f,i,tag,params,confounds_exist,confound_files) for i,f in enumerate(files)}
            for j in as_completed(job):
                j.result()

        if update_pipeline == True:
            if not self.confound_pipeline and len(self.get_confound_files(quiet=1)) > 0:
                self.set_confound_pipeline = self.pipeline
            self.set_pipeline('teneto_' + teneto.__version__)
            self.set_pipeline_subdir('tvc')
            self.set_last_analysis_step('tvc')

    def _run_derive(self,f,i,tag,params,confounds_exist,confound_files):
        """
        Funciton called by TenetoBIDS.derive for parallel processing.
        """
        # ADD MORE HERE (csv, json, nifti)
        if f.split('.')[-1] == 'npy':
            data = np.load(f)
        else:
            raise ValueError('derive can only load npy files at the moment')

        save_name, save_dir, base_dir = self._save_namepaths_bids_derivatives(f,'_tvcmethod-' + params['method'] + tag + '_tvc','tvc')

        if 'weight-var' in params.keys():
            if params['weight-var'] == 'from-subject-fc':
                fc_dir = base_dir + '/fc/'
                fc = os.listdir(fc_dir)
                i = 0
                for ff in fc: 
                    if ff.split('_fc.npy')[0] in f:
                        params['weight-var'] = np.load(fc_dir + ff)
                        i += 1
                if i != 1: 
                    raise ValueError('Cannot correct find FC files')

        if 'weight-mean' in params.keys():
            if params['weight-mean'] == 'from-subject-fc':
                fc_dir = base_dir + '/fc/'
                fc = os.listdir(fc_dir)
                i = 0
                for ff in fc: 
                    if ff.split('_fc.npy')[0] in f:
                        params['weight-mean'] = np.load(fc_dir + ff)
                        i += 1
                if i != 1: 
                    raise ValueError('Cannot correct find FC files')


        params['report'] = 'yes'
        params['report_path'] =  save_dir + '/report/'
        params['report_filename'] =  save_name + '_derivationreport.html'

        if not os.path.exists(params['report_path']):
            os.makedirs(params['report_path'])

        dfc = teneto.derive.derive(data,params)
        np.save(save_dir + save_name + '.npy', dfc)

        if confounds_exist:
            analysis_step = 'tvc-derive'
            if confound_files[i].split('.')[-1] == 'csv':
                delimiter = ','
            elif confound_files[i].split('.')[-1] == 'tsv':
                delimiter = '\t'
            df = pd.read_csv(confound_files[i],sep=delimiter)
            df = df.fillna(df.median())
            ind = np.triu_indices(dfc.shape[0], k=1)
            dfc_df = pd.DataFrame(dfc[ind[0],ind[1],:].transpose())
            # If windowed, prune df so that it matches with dfc_df 
            if len(df) != len(dfc_df): 
                df = df.iloc[int(np.round((params['windowsize']-1)/2)):int(np.round((params['windowsize']-1)/2)+len(dfc_df))]
                df.reset_index(inplace=True,drop=True)
            #NOW CORRELATE DF WITH DFC BUT ALONG INDEX NOT DF.
            dfc_df_z = (dfc_df - dfc_df.mean())
            df_z = (df - df.mean())
            R_df = dfc_df_z.T.dot(df_z).div(len(dfc_df)).div(df_z.std(ddof=0)).div(dfc_df_z.std(ddof=0), axis=0)
            R_df_describe = R_df.describe()
            desc_index = R_df_describe.index
            confound_report_dir = params['report_path'] + '/' + save_name + '_confoundcorr/'
            confound_report_figdir = confound_report_dir + 'figures/'
            if not os.path.exists(confound_report_figdir):
                os.makedirs(confound_report_figdir)
            report = '<html><body>'
            report += '<h1> Correlation of ' + analysis_step + ' and confounds.</h1>'
            for c in R_df.columns:
                fig,ax = plt.subplots(1)
                ax = sns.distplot(R_df[c],hist=False, color='m', ax=ax, kde_kws={"shade": True})
                fig.savefig(confound_report_figdir + c + '.png')
                plt.close(fig)
                report += '<h2>' + c + '</h2>'
                for ind_name,r in enumerate(R_df_describe[c]):
                    report += str(desc_index[ind_name]) + ': '
                    report += str(r) + '<br>'
                report += 'Distribution of corrlation values:'
                report += '<img src=' + os.path.abspath(confound_report_figdir) + '/' + c + '.png><br><br>'
            report += '</body></html>'

            with open(confound_report_dir + save_name + '_confoundcorr.html', 'w') as file:
                file.write(report)


    def make_functional_connectivity(self,njobs=None,returngroup=False,file_hdr=None,file_idx=None):
        """
        Makes connectivity matrix for each of the subjects.

        Parameters
        ----------
        returngroup : bool, default=False
            If true, returns the group average connectivity matrix.
        njobs : int
            How many parallel jobs to run
        file_idx : bool 
            Default False, true if to ignore index column in loaded file. 
        file_hdr : bool 
            Default False, true if to ignore header row in loaded file. 

        Returns
        -------
        Saves data in derivatives/teneto_<version>/.../fc/
        R_group : array
            if returngroup is true, the average connectivity matrix is returned.

        """
        if not njobs:
            njobs = self.njobs
        self.add_history(inspect.stack()[0][3], locals(), 1)
        files = self.get_selected_files(quiet=1)

        R_group = []

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_make_functional_connectivity,f,file_hdr,file_idx) for f in files}
            for j in as_completed(job):
                R_group.append(j.result())

        if returngroup:
            # Fisher tranform -> mean -> inverse fisher tranform
            R_group = np.tanh(np.mean(np.arctanh(np.array(R_group)), axis=0))
            return np.array(R_group)

    def _run_make_functional_connectivity(self,f,file_hdr,file_idx):
            save_name, save_dir, base_dir = self._save_namepaths_bids_derivatives(f,'_fc','fc')

            data = self._load_file(f,output='array',header=file_hdr,index_col=file_idx)

            R = teneto.misc.corrcoef_matrix(data)[0]
            # Fisher transform of subject R values before group average
            np.save(save_dir + save_name + '.npy', R)
            return R

    def get_functional_connectivity_files(self,quiet=1):
        """
        Load functional connectivity files. Requires make_functional_connectivity to be run

        Parameters
        ----------
        quiet: int
            If 1, prints results. If 0, no results printed.

        Returns
        -------
        found_files : list
            Get list of files where functional connecitivty is stored.
        """
        # This could be mnade better
        file_dict = {
            'sub': self.subjects,
            'ses': self.sessions,
            'task': self.tasks,
            'run': self.runs}
        # Only keep none empty elemenets
        file_types = []
        file_components = []
        for k in ['sub', 'ses', 'task', 'run']:
            if file_dict[k]:
                file_types.append(k)
                file_components += [file_dict[k]]
        file_list = list(itertools.product(*file_components))
        # Specify main directory
        mdir = self.BIDS_dir + '/derivatives/teneto_' + teneto.__version__ + '/'
        found_files = []

        for f in file_list:
            wdir = str(mdir)
            fstr = ''
            for i,k in enumerate(file_types):
                if k == 'sub' or k == 'ses':
                    wdir += '/' + k + '-' + f[i] + '/'
                if k != 'sub':
                    fstr += '_'
                else:
                    wdir += 'func/'
                fstr += k + '-' + f[i] + '.*'
            #wdir += '/' + self.pipeline_subdir + '/'
            if not self.space:
                space = ''
            else:
                space = '_space-' + self.space

            wdir += '/fc/'

            r = re.compile('^' + fstr + '.*' + space + '.*' + '_fc[.].*')
            if os.path.exists(wdir):
                # make filenames
                found = list(filter(r.match, os.listdir(wdir)))
                # Include only if all analysis step tags are present
                found = [i for i in found if all(x in i for x in self.analysis_steps)]
                # Exclude if confounds tag is present
                found = [i for i in found if '_confounds' not in i]
                # Make full paths
                found = list(map(str.__add__,[re.sub('/+','/',wdir)]*len(found),found))
                found = [i for i in found if not any([bf in i for bf in self.bad_files])]
                if found:
                    found_files += found

            if quiet==-1:
                print(wdir)

        if quiet == 0:
            print(found_files)
        return found_files


    def _save_namepaths_bids_derivatives(self,f,save_tag,save_directory):
        """
        Creates output directory and output name

        Paramters 
        ---------   
        f : str
            input files, includes the file suffix
        save_tag : str
            what should be added to f in the output file.
        save_directory : str
            additional directory that the output file should go in

        Returns 
        -------
        save_name : str 
            previous filename with new save_tag 
        save_dir : str
            directory where it will be saved 
        base_dir : str 
            subjective base directory (i.e. derivatives/teneto/func[/anythingelse/])

        """
        file_name = f.split('/')[-1].split('.')[0]
        if save_tag[0] != '_':
            save_tag = '_' + save_tag
        save_name = file_name + save_tag
        paths_post_pipeline = f.split(self.pipeline)
        if self.pipeline_subdir:
            paths_post_pipeline = paths_post_pipeline[1].split(self.pipeline_subdir)[0]
        else:
            paths_post_pipeline = paths_post_pipeline[1].split(file_name)[0]
        base_dir = self.BIDS_dir + '/derivatives/' + 'teneto_' + teneto.__version__ + '/' + paths_post_pipeline + '/'
        save_dir = base_dir + '/' + save_directory + '/'
        if not os.path.exists(save_dir):
            # A case has happened where this has been done in parallel and an error was raised. So do try/except
            try:
                os.makedirs(save_dir)
            except:
                #Wait 2 seconds so that the error does not try and save something in the directory before it is created
                time.sleep(2)
        return save_name, save_dir, base_dir


    def get_space_alternatives(self,quiet=0):
        """
        Returns which space alternatives can be identified in the BIDS derivatives structure. Spaces are denoted with the prefix "space-".
        """
        if not self.pipeline:
            print('Please set pipeline first.')
            self.get_pipeline_alternatives()
        else:
            space_alternatives = []
            if self.sessions:
                ses = '/ses-' + self.sessions + '/'
            else:
                ses = ''
            for s in self.subjects:
                derdir_files = os.listdir(self.BIDS_dir + '/derivatives/' + self.pipeline + '/' + self.pipeline_subdir +'/sub-' + s + '/' + ses + 'func/')
                space_alternatives += [re.split('[_.]',f.split('_space-')[1])[0] for f in derdir_files if re.search('_space-',f)]
            space_alternatives = set(space_alternatives)
            if quiet == 0:
                print('Space alternatives: ' + ', '.join(space_alternatives))
            return list(space_alternatives)

    def get_pipeline_alternatives(self,quiet=0):
        """
        The pipeline are the different outputs that are placed in the ./derivatives directory.

        get_pipeline_alternatives gets those which are found in the specified BIDS directory structure.
        """
        if not os.path.exists(self.BIDS_dir + '/derivatives/'):
            print('Derivative directory not found. Is the data preprocessed?')
        else:
            pipeline_alternatives = os.listdir(self.BIDS_dir + '/derivatives/')
            if quiet == 0:
                print('Derivative alternatives: ' + ', '.join(pipeline_alternatives))
            return list(pipeline_alternatives)

    def get_pipeline_subdir_alternatives(self,quiet=0):
        """
        Note
        -----

        This function currently returns the wrong folders and will be fixed in the future.

        This function should return ./derivatives/pipeline/sub-xx/[ses-yy/][func/]/pipeline_subdir
        But it does not care about ses-yy at the moment.
        """
        if not self.pipeline:
            print('Please set pipeline first.')
            self.get_pipeline_alternatives()
        else:
            pipeline_subdir_alternatives = []
            for s in self.subjects:
                derdir_files = os.listdir(self.BIDS_dir + '/derivatives/' + self.pipeline + '/' + s + '/')
                pipeline_subdir_alternatives += [f for f in derdir_files if os.path.isdir(f)]
            pipeline_subdir_alternatives = set(pipeline_subdir_alternatives)
            if quiet == 0:
                print('Pipeline_subdir alternatives: ' + ', '.join(pipeline_subdir_alternatives))
            return list(pipeline_subdir_alternatives)

    def load_community_data(self,community_type,tag=None):
        """
        Load derived communities\

        Parameters 
        ---------- 
        community_type: str 
            Either static or temporal 
        tag : str or list
            any additional tag that must be in file name. After the tag there must either be a underscore or period (following bids). 

        Returns 
        -------
        loads TenetoBIDS.community_data_ and TenetoBIDS.community_info_
        """ 
        self.add_history(inspect.stack()[0][3], locals(), 1)
        
        data_list=[]
        info_list = []

        community_type = 'communitytype-' + community_type 

        if not tag:
            tag = ['']
        elif isinstance(tag,str): 
            tag = [tag]

        for s in self.subjects:
            # Define base folder
            base_path = self.BIDS_dir + '/derivatives/' + self.pipeline
            base_path += '/sub-' + s + '/func/communities/'
            if os.path.exists(base_path):
                file_list=os.listdir(base_path)
                for f in file_list:
                    # Include only if all analysis step tags are present
                    if community_type in f and all([t + '_' in f or t + '.' in f for t in tag]):
                        # Get all BIDS tags. i.e. in 'sub-AAA', get 'sub' as key and 'AAA' as item.
                        bid_tags=re.findall('[a-zA-Z]*-',f)
                        bids_tag_dict = {}
                        for t in bid_tags:
                            key = t[:-1]
                            bids_tag_dict[key]=re.findall(t+'[A-Za-z0-9.,*+]*',f)[0].split('-')[-1]
                        if f.split('.')[-1] == 'npy':
                            data = np.load(base_path+f)
                            data_list.append(data)
                            info = pd.DataFrame(bids_tag_dict,index=[0])
                            info_list.append(info)
                        else:
                            print('Warning: Could not find data for a subject')

        #Get time-shape of data loaded 
        if community_type == 'communitytype-' + 'temporal': 
            shape = np.array([n.shape[-1] for n in data_list])
            if len(np.unique(shape)) != 1:
                print("Warning: Unequal time dimension. Returning networkcommunity_data_ as list.")    
                self.community_data_ = data_list
            else:         
                self.community_data_ = np.array(data_list)
        else: 
            self.community_data_ = np.array(data_list)
        if info_list:
            out_info = pd.concat(info_list)
            out_info.reset_index(inplace=True,drop=True)
            self.community_info_ = out_info


    def get_selected_files(self,quiet=0):
        """
        Parameters
        ----------
        quiet: int
            If 1, prints results. If 0, no results printed.

        Returns
        -------
        found_files : list
            The files which are currently selected with the current using the set pipeline, pipeline_subdir, space, parcellation, tasks, runs, subjects etc. There are the files that will generally be used if calling a make_ function.
        """
        # This could be mnade better
        file_dict = {
            'sub': self.subjects,
            'ses': self.sessions,
            'task': self.tasks,
            'run': self.runs}
        # Only keep none empty elemenets
        file_types = []
        file_components = []
        for k in ['sub', 'ses', 'task', 'run']:
            if file_dict[k]:
                file_types.append(k)
                file_components += [file_dict[k]]
        file_list = list(itertools.product(*file_components))
        # Specify main directory
        mdir = self.BIDS_dir + '/derivatives/' + self.pipeline
        found_files = []

        for f in file_list:
            wdir = str(mdir)
            fstr = ''
            for i,k in enumerate(file_types):
                if k == 'sub' or k == 'ses':
                    wdir += '/' + k + '-' + f[i] + '/'
                if k != 'sub':
                    fstr += '_'
                else:
                    wdir += 'func/'
                fstr += k + '-' + f[i] + '.*'
            wdir += '/' + self.pipeline_subdir + '/'
            if not self.space:
                space = ''
            else:
                space = '_space-' + self.space

            r = re.compile('^' + fstr + '.*' + space + '.*' + self.last_analysis_step + '[.].*')
            if os.path.exists(wdir):
                # make filenames
                found = list(filter(r.match, os.listdir(wdir)))
                # Include only if all analysis step tags are present
                found = [i for i in found if all(x in i for x in self.analysis_steps)]
                # Exclude if confounds tag is present
                found = [i for i in found if '_confounds' not in i]
                # Make full paths
                found = list(map(str.__add__,[re.sub('/+','/',wdir)]*len(found),found))
                found = [i for i in found if not any([bf in i for bf in self.bad_files])]
                if found:
                    found_files += found

            if quiet==-1:
                print(wdir)

        if quiet == 0:
            print(found_files)
        return found_files


    def set_exclusion_file(self,confound,exclusion_criteria,confound_stat='mean'):
        """
        Excludes subjects given a certain exclusion criteria.

        NOTE THIS ONLY WORKS IF THERE IS 1 SESSIONS, TASK and RUN.

        Parameters
        ----------
            confound : str or list
                string or list of confound name(s) from confound files
            exclusion_criteria  : str or list
                for each confound, an exclusion_criteria should be expressed as a string. It starts with >,<,>= or <= then the numerical threshold. Ex. '>0.2' will entail every subject with the avg greater than 0.2 of confound will be rejected.
            confound_stat : str or list
                Can be median, mean, std. How the confound data is aggregated (so if there is a meaasure per time-point, this is averaged over all time points. If multiple confounds specified, this has to be a list.).
        Returns
        ------
            calls TenetoBIDS.set_bad_files with the files meeting the exclusion criteria.
        """
        if isinstance(confound,str):
            confound = [confound]
        if not isinstance(exclusion_criteria,list):
            exclusion_criteria = [exclusion_criteria]
        if isinstance(confound_stat,str):
            confound_stat = [confound_stat]
        if len(exclusion_criteria)!=len(confound):
            raise ValueError('Same number of confound names and exclusion criteria must be given')
        if len(confound_stat)!=len(confound):
            raise ValueError('Same number of confound names and confound stats must be given')
        rel = []
        crit = []
        for ec in exclusion_criteria:
            if ec[0:2] == '>=':
                rel.append(np.greater_equal)
                crit.append(float(ec[2:]))
            elif ec[0:2] == '<=':
                rel.append(np.less_equal)
                crit.append(float(ec[2:]))
            elif ec[0] == '>':
                rel.append(np.greater)
                crit.append(float(ec[1:]))
            elif ec[0] == '<':
                rel.append(np.less)
                crit.append(float(ec[1:]))
            else:
                raise ValueError('exclusion crieria must being with >,<,>= or <=')
        # Load filelist and confound list
        files = sorted(self.get_selected_files(quiet=1))
        confound_files = sorted(self.get_confound_files(quiet=1))
        # Check integerity of confound list
        if len(files) != len(confound_files):
            print('WARNING: number of confound files does not equal number of selected files')
        for n in range(len(files)):
            if confound_files[n].split('_confounds')[0].split('func')[1] not in files[n]:
                raise ValueError('Confound matching with data did not work.')
        bad_files = []
        bs = 0
        for s, cfile in enumerate(confound_files):
            if cfile.split('.')[-1] == 'csv':
                delimiter = ','
            elif cfile.split('.')[-1] == 'tsv':
                delimiter = '\t'
            df = pd.read_csv(cfile,sep=delimiter)
            found_bad_subject = False
            for i in range(len(confound)):
                if confound_stat[i] == 'median':
                    if rel[i](df[confound[i]].median(),crit[i]):
                        found_bad_subject = True
                elif confound_stat[i] == 'mean':
                    if rel[i](df[confound[i]].mean(),crit[i]):
                        found_bad_subject = True
                elif confound_stat[i] == 'std':
                    if rel(df[i][confound[i]].std(),crit[i]):
                        found_bad_subject = True
            if found_bad_subject:
                bad_files.append(files[s])
                bs += 1
        self.set_bad_files(bad_files)
        print('Removed ' + str(bs) + ' files from inclusion.')

    def _load_file(self,fname,output=None,header=None,index_col=None):
        """
        Given a file name loads that file

        Parameters
        ----------
        fname : str
            file name and path. Can be csv, tsv, npy. 
        output : str 
            array or pd (pandas dataframe). Default = array 
        header : bool (default false)
            if there is a header in the csv or tsv file, true will use first row in file. 
        index_col : bool (default false)
            if there is an index column in the csv or tsv file, true will use first row in file. 

        Returns 
        -------
        f : array or pd (pandas dataframe) 
            The loaded file
        """ 
        if index_col: 
            index_col = 0 
        else: 
            index_col = None
        if header: 
            header = 0 
        else:
            header = None 

        if not output: 
            output = 'array' 
        fsuf = fname.split('.')[-1]
        if fsuf == 'tsv':
            dtype = 'csv'
            delim = '\t'
        if fsuf == 'csv':
            dtype = 'csv'
            delim = ',' 
        elif fsuf == 'npy':
            dtype = 'npy'
        if dtype == 'csv': 
            f = pd.read_csv(fname,header=header,index_col=index_col,sep=delim) 
            if output == 'array':
                f = f.values 
        elif dtype == 'npy': 
            f = np.load(fname)
            if output == 'pd': 
                f = pd.DataFrame(f) 
        elif dtype == 'pkl':
            f = pd.read_pickle(fname)
            if output == 'array': 
                f = f.values     
        return f 


    def set_exclusion_timepoint(self,confound,exclusion_criteria,replace_with,tol=1,file_hdr=False,file_idx=False,confound_hdr=True,confound_idx=False):
        """
        Excludes subjects given a certain exclusion criteria. Does not work on nifti files, only csv, numpy or tsc. Assumes data is node,time

        Parameters
        ----------
            confound : str or list
                string or list of confound name(s) from confound files. Assumes data is node,time
            exclusion_criteria  : str or list
                for each confound, an exclusion_criteria should be expressed as a string. It starts with >,<,>= or <= then the numerical threshold. Ex. '>0.2' will entail every subject with the avg greater than 0.2 of confound will be rejected.
            replace_with : str
                Can be 'nan' (bad values become nans) or 'cubicspline' (bad values are interpolated). If bad value occurs at 0 or -1 index, then these values are kept and no interpolation occurs.
            tol = float 
                Tolerance of exlcuded time-points allowed before becoming a bad subject. 

        Returns
        ------
            Loads the TenetoBIDS.selected_files and replaces any instances of confound meeting the exclusion_criteria with replace_with.
        """
        if isinstance(confound,str):
            confound = [confound]
        if isinstance(exclusion_criteria,str):
            exclusion_criteria = [exclusion_criteria]
        if len(exclusion_criteria)!=len(confound):
            raise ValueError('Same number of confound names and exclusion criteria must be given')
        rel = []
        crit = []
        for ec in exclusion_criteria:
            if ec[0:2] == '>=':
                rel.append(np.greater_equal)
                crit.append(float(ec[2:]))
            elif ec[0:2] == '<=':
                rel.append(np.less_equal)
                crit.append(float(ec[2:]))
            elif ec[0] == '>':
                rel.append(np.greater)
                crit.append(float(ec[1:]))
            elif ec[0] == '<':
                rel.append(np.less)
                crit.append(float(ec[1:]))
            else:
                raise ValueError('exclusion crieria must being with >,<,>= or <=')
        files = sorted(self.get_selected_files(quiet=1))
        confound_files = sorted(self.get_confound_files(quiet=1))
        # Check integerity of confound list
        if len(files) != len(confound_files):
            print('WARNING: number of confound files does not equal number of selected files')
        # Make sure confound file name is in the file name.
        for n in range(len(files)):
            if confound_files[n].split('_confounds')[0].split('func')[1] not in files[n]:
                raise ValueError('Confound matching with data did not work.')
        bad_files = []
        bad_time = []
        bs = 0
        for i, cfile in enumerate(confound_files):
            data = self._load_file(files[i],output='array',header=file_hdr,index_col=file_idx)
            df = self._load_file(cfile,output='pd',header=confound_hdr,index_col=confound_idx)
            deleted_timepoints_txt = ''
            ind = []
            # Is set to 1 if subject should be saved ("goodsubject")
            gs=0
            # Can't interpolate values if nanind is at the beginning or end. So keep these as their original values. 
            for ci,c in enumerate(confound):
                ind = df[rel[ci](df[c],crit[ci])].index
                if np.array(len(ind))/np.array(len(df))>tol: 
                    bad_files.append(files[i])
                    bs += 1 
                else: 
                    bad_time.append(len(ind))
                    gs = 1
                if replace_with == 'cubicspline': 
                    if 0 in ind: 
                        ind = np.delete(ind,np.where(ind==0))
                    if df.index.max(): 
                        ind = np.delete(ind,np.where(ind==df.index.max()))             
                data[:,ind] = np.nan
            nanind = np.where(np.isnan(data[0,:]))[0]
            nonnanind = np.where(np.isnan(data[0,:])==0)[0]
            nanind = nanind[nanind>nonnanind.min()]
            nanind = nanind[nanind<nonnanind.max()]
            deleted_timepoints_txt += 'number of deleted timepoints (' + c + '): ' + str(len(nanind)) + '\n'
            deleted_timepoints_txt += 'problematic timepoints timepoints (' + c + '): '
            deleted_timepoints_txt += str(nanind)
            if replace_with == 'cubicspline':
                for n in range(data.shape[0]):
                    interp = interp1d(nonnanind,data[n,nonnanind],kind='cubic')
                    data[n,nanind] = interp(nanind)
            # only save if the subject is not excluded 
            if gs == 1: 
                np.save(files[i][:-4] + '_scrub',data)
            sdir = ''
            if files[0] == '/':
                sdir += '/'
            sdir += '/'.join(files[i].split('/')[:-1])
            with open(files[i][:-4] + '_scrub_exclusion_info.txt', 'w') as text_file:
                text_file.write(deleted_timepoints_txt)
        self.analysis_steps += list(self.last_analysis_step)
        self.last_analysis_step = 'scrub'
        self.set_bad_files(bad_files)
        print('Removed ' + str(bs) + ' files from inclusion.')
        print('Average ' + str(np.array(bad_time).mean()) + ' time-points were smoothed (in non-excluded files).')


    def get_confound_files(self,quiet=0):
        """
        Returns confound files that are currently selected

        Parameters
        ----------
        quiet: int
            If 1, prints results. If 0, no results printed.

        Returns
        -------
        found_files : list
            The files which are currently selected wihch will be used if removing confounds.

            This is specified by confound_pipeline (or pipeline if unset), confound_pipeline_subdir (or pipeline_subdir if unset), space, parcellation, tasks, runs, subjects etc.
        """
        # This could be mnade better
        file_dict = {
            'sub': self.subjects,
            'ses': self.sessions,
            'task': self.tasks,
            'run': self.runs}
        # Only keep none empty elemenets
        file_types = []
        file_components = []
        for k in ['sub', 'ses', 'task', 'run']:
            if file_dict[k]:
                file_types.append(k)
                file_components += [file_dict[k]]
        file_list = list(itertools.product(*file_components))
        # Specify main directory
        if self.confound_pipeline:
            mdir = self.BIDS_dir + '/derivatives/' + self.confound_pipeline
        else:
            mdir = self.BIDS_dir + '/derivatives/' + self.pipeline
        found_files = []
        for f in file_list:
            wdir = str(mdir)
            fstr = ''
            for i,k in enumerate(file_types):
                if k == 'sub' or k == 'ses':
                    wdir += '/' + k + '-' + f[i] + '/'
                if k != 'sub':
                    fstr += '_'
                else:
                    wdir += 'func/'
                fstr += k + '-' + f[i] + '.*'
            wdir_pipesub = wdir + '/' + self.pipeline_subdir + '/'
            # Allow for pipeline_subdir to not be there (ToDo: perhaps add confound_pipeline_subdir in future)
            if os.path.exists(wdir_pipesub):
                wdir = wdir_pipesub
            r = re.compile('^' + fstr + '.*' + '_confounds' + '.*')
            if os.path.exists(wdir):
                found = list(filter(r.match, os.listdir(wdir)))
                found = list(map(str.__add__,[re.sub('/+','/',wdir)]*len(found),found))
                # Fix this. sublist in sublist.
                found = [f for f in found if all(f.split('_confounds')[0] not in bf for bf in self.bad_files)]
                # found = [f for f in found if f.split('_confounds')[0].split('func')[1] not in bad_files_clipped]
                if found:
                    found_files += found


        if quiet == 0:
            print(found_files)
        return found_files



    def get_confound_alternatives(self,quiet=0):
        # This could be mnade better
        file_list = self.get_confound_files(quiet=1)

        confounds = []
        for f in file_list:
            file_format = f.split('.')[-1]
            if  file_format == 'tsv':
                confounds += list(pd.read_csv(f,delimiter='\t').keys())
            elif file_format == 'csv':
                confounds += list(pd.read_csv(f,delimiter=',').keys())

        confounds = sorted(list(set(confounds)))

        if quiet == 0:
            print('Confounds in confound files: \n - ' + '\n - '.join(confounds))
        return confounds


    def make_parcellation(self,parcellation,parc_type=None,parc_params=None,network='defaults',update_pipeline=True,removeconfounds=False,tag=None,njobs=None,clean_params=None):

        """
        Reduces the data from voxel to parcellation space. Files get saved in a teneto folder in the derivatives with a roi tag at the end.

        Parameters
        -----------

        parcellation : str
            specify which parcellation that you would like to use. For MNI: 'power2012_264', 'gordon2014_333'. TAL: 'shen2013_278'
        parc_type : str
            can be 'sphere' or 'region'. If nothing is specified, the default for that parcellation will be used.
        parc_params : dict
            **kwargs for nilearn functions
        network : str
            if "defaults", it selects static parcellation, _if available_ (other options will be made available soon).
        removeconfounds : bool
            if true, regresses out confounds that are specfied in self.set_confounds with linear regression.
        update_pipeline : bool
            TenetoBIDS gets updated with the parcellated files being selected.
        tag : str or list
            any additional tag that must be in file name. After the tag there must either be a underscore or period (following bids). 
        clean_params : dict
            **kwargs for nilearn function nilearn.signal.clean
        njobs : n 
            number of processes to run. Overrides TenetoBIDS.njobs

        Returns
        -------
        Files are saved in ./BIDS_dir/derivatives/teneto_<version>/.../parcellation/.
        To load these files call TenetoBIDS.load_parcellation.

        NOTE
        ----
        These functions make use of nilearn. Please cite nilearn if used in a publicaiton.
        """
        if not njobs:
            njobs = self.njobs
        self.add_history(inspect.stack()[0][3], locals(), 1)

        parc_name = parcellation.split('_')[0].lower()

        # Check confounds have been specified
        if not self.confounds and removeconfounds:
            raise ValueError('Specified confounds are not found. Make sure that you have run self.set_confunds([\'Confound1\',\'Confound2\']) first.')

        # Check confounds have been specified
        if update_pipeline == False and removeconfounds:
            raise ValueError('Pipeline must be updated in order to remove confounds within this funciton.')

        # In theory these should be the same. So at the moment, it goes through each element and checks they are matched.
        # A matching algorithem may be needed if cases arise where this isnt the case
        files = self.get_selected_files(quiet=1)
        # Load network communities, if possible. 
        self.set_network_communities(parcellation)

        if not tag:
            tag = ''
        else:
            tag = '_' + tag

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_make_parcellation,f,i,tag,parcellation,parc_name,parc_type,parc_params) for i,f in enumerate(files)}
            for j in as_completed(job):
                j.result()

        if update_pipeline == True:
            if not self.confound_pipeline and len(self.get_confound_files(quiet=1)) > 0:
                self.set_confound_pipeline(self.pipeline)
            self.set_pipeline('teneto_' + teneto.__version__)
            self.set_pipeline_subdir('parcellation')
            self.analysis_steps += list(self.last_analysis_step)
            if tag:
                self.analysis_steps += [tag[1:]]
            self.set_last_analysis_step('roi')
            self.parcellation = parcellation

            if removeconfounds: 
                self.removeconfounds(clean_params=clean_params,transpose=True,njobs=njobs)
            
    def _run_make_parcellation(self,f,i,tag,parcellation,parc_name,parc_type,parc_params):
        save_name, save_dir, base_dir = self._save_namepaths_bids_derivatives(f,'_parc-' + parc_name + tag + '_roi','parcellation')
        roi = teneto.utils.make_parcellation(f,parcellation,parc_type,parc_params)
        #Make data node,time
        roi = roi.transpose()
        np.save(save_dir + save_name + '.npy', roi)



    # def communitydetection(self,community_detection_params,community_type='temporal',tag=None,file_hdr=False,file_idx=False,njobs=None):
    #     """
    #     Calls temporal_louvain_with_consensus on connectivity data

    #     Parameters
    #     ----------

    #     community_detection_params : dict 
    #         kwargs for detection. See teneto.communitydetection.louvain.temporal_louvain_with_consensus
    #     community_type : str
    #         Either 'temporal' or 'static'. If temporal, community is made per time-point for each timepoint.         
    #     file_idx : bool (default false) 
    #         if true, index column present in data and this will be ignored 
    #     file_hdr : bool (default false) 
    #         if true, header row present in data and this will be ignored 
    #     njobs : int 
    #         number of processes to run. Overrides TenetoBIDS.njobs

    #     Note 
    #     ----
    #     All non-positive edges are made to zero. 


    #     Returns 
    #     ------- 
    #     List of communities for each subject. Saved in BIDS_dir/derivatives/teneto/communitydetection/
    #     """
    #     if not njobs:
    #         njobs = self.njobs
    #     self.add_history(inspect.stack()[0][3], locals(), 1)

    #     if not tag:
    #         tag = ['']
    #     if isinstance(tag,str):
    #         tag = [tag]        

    #     if community_type == 'temporal':
    #         files = self.get_selected_files(quiet=True) 
    #         # Run check to make sure files are tvc input
    #         for f in files: 
    #                 if 'tvc' not in f: 
    #                     raise ValueError('tvc tag not found in filename. TVC data must be used in communitydetection (perhaps run TenetoBIDS.derive first?).')
    #     elif community_type == 'static': 
    #         files = self.get_functional_connectivity_files(quiet=True) 

    #     with ProcessPoolExecutor(max_workers=njobs) as executor:
    #         job = {executor.submit(self._run_communitydetection,f,community_detection_params,community_type,file_hdr,file_idx) for i,f in enumerate(files) if all([t + '_' in f or t + '.' in f for t in tag])}
    #         for j in as_completed(job):
    #             j.result()

    # def _run_communitydetection(self,f,params,community_type,file_hdr=False,file_idx=False): 
    #     tag = 'communitytype-' + community_type
    #     if 'resolution_parameter' in params:    
    #         tag += '_gamma-' + str(np.round(params['resolution_parameter'],5))
    #     if 'interslice_weight' in params: 
    #         tag += '_omega-' + str(np.round(params['interslice_weight'],5))
    #     tag += '_louvain'
    #     if community_type == 'temporal': 
    #         save_name, save_dir, base_dir = self._save_namepaths_bids_derivatives(f,tag,'communities')
    #     else: 
    #         save_name, a, b = self._save_namepaths_bids_derivatives(f,tag,'')
    #         save_dir = f.split('fc')[0] + '/communities/'
    #     if not os.path.exists(save_dir): 
    #         try: 
    #             os.makedirs(save_dir)
    #         except: 
    #             #Wait 2 seconds so that the error does not try and save something in the directory before it is created
    #             time.sleep(2)
    #     data = self._load_file(f,output='array',header=file_idx,index_col=file_idx) 
    #     # Only put positive edges into clustering (more advanced methods can be added here later )
    #     data[data<0] = 0
    #     C = teneto.communitydetection.temporal_louvain_with_consensus(data, **params)
    #     np.save(save_dir + save_name,C)


    def removeconfounds(self,confounds=None,clean_params=None,transpose=False,njobs=None,update_pipeline=True,confound_hdr=True,confound_idx=False,file_hdr=False,file_idx=False): 
        """ 
        Removes specified confounds using nilearn.signal.clean 

        Parameters
        ----------
        confounds : list 
            List of confounds. Can be prespecified in set_confounds 
        clean_params : dict 
            Dictionary of kawgs to pass to nilearn.signal.clean 
        transpose : bool (default False) 
            Default removeconfounds works on time,node dimensions. Pass transpose=True to transpose pre and post confound removal. 
        njobs : int 
            Number of jobs. Otherwise tenetoBIDS.njobs is run. 
        update_pipeline : bool 
            update pipeline with '_clean' tag for new files created 
        confound_hdr : bool (default True) 
            if header is present in confound data (often the case in csv). Note, at present there is not an option for there to be header but no index. 
        confound_idx : bool (default False) 
            if index is present in confound data (often the case in csv). Note, at present there is not an option for there to be header but no index. This is generally not as important as getting the header correct header. 
        file_hdr : bool (default True) 
            if header is present in data (often the case in csv). Note, at present there is not an option for there to be header but no index. 
        file_idx : bool (default False) 
            if index is present in data (often the case in csv). Note, at present there is not an option for there to be header but no index. 
        
        Returns
        -------
        Says all TenetBIDS.get_selected_files with confounds removed with _rmconfounds at the end.

        Note 
        ----
        There may be some issues regarding loading non-cleaned data through the TenetoBIDS functions instead of the cleaned data. This depeneds on when you clean the data. 
        """
        if not njobs:
            njobs = self.njobs
        self.add_history(inspect.stack()[0][3], locals(), 1)

        if not self.confounds and not confounds:
            raise ValueError('Specified confounds are not found. Make sure that you have run self.set_confunds([\'Confound1\',\'Confound2\']) first or pass confounds as input to function.')

        if confounds: 
            self.set_confounds(confounds)
        files = sorted(self.get_selected_files(quiet=1))
        confound_files = sorted(self.get_confound_files(quiet=1))
        if len(files) != len(confound_files):
            print('WARNING: number of confound files does not equal number of selected files')
        for n in range(len(files)):
            if confound_files[n].split('_confounds')[0].split('func')[1] not in files[n]:
                raise ValueError('Confound matching with data did not work.')

        if not clean_params:
            clean_params = {}

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_removeconfounds,f,confound_files[i],clean_params,transpose,confound_hdr,confound_idx,file_hdr,file_idx) for i,f in enumerate(files)}
            for j in as_completed(job):
                j.result()

        if update_pipeline == True:
            self.analysis_steps += list(self.last_analysis_step)
            self.set_last_analysis_step('clean')

    def _run_removeconfounds(self,file_path,confound_path,clean_params,transpose=False,confound_hdr=True,confound_idx=True,file_hdr=False,file_idx=False): 
        df = self._load_file(confound_path,output='pd',header=confound_hdr,index_col=confound_idx)
        df = df[self.confounds]
        roi = self._load_file(file_path,output='array',header=file_hdr,index_col=file_idx)
        if transpose: 
            roi = roi.transpose() 
        if df.isnull().any().any():
            # Not sure what is the best way to deal with this.
            # The time points could be ignored. But if multiple confounds, this means these values will get ignored
            print('WARNING: Some confounds were NaNs. Setting these values to median of confound.')
            df = df.fillna(df.median())
        roi = nilearn.signal.clean(roi,confounds=df.values,**clean_params)
        if transpose: 
            roi = roi.transpose() 
        np.save(file_path.split(self.last_analysis_step)[0] + self.last_analysis_step + '_clean.npy',roi)

    def networkmeasures(self, measure=None, measure_params={}, load_tag=None, save_tag=None, njobs=None):
        """
        Calculates a network measure

        For available funcitons see: teneto.networkmeasures

        Parameters
        ----------

        measure : str or list
            Mame of function(s) from teneto.networkmeasures that will be run.

        measure_params : dict or list of dctionaries)
            Containing kwargs for the argument in measure.
            See note regarding Communities key. 

        load_tag : str
            Add additional tag to loaded filenames.

        save_tag : str
            Add additional tag to saved filenames.

        Note
        ----
        In measure_params, if communities can equal 'template', 'static', or 'temporal'. 
        These options must be precalculated. If template, Teneto tries to load default for parcellation. If static, loads static communities 
        in BIDS_dir/teneto_<version>/sub-.../func/communities/..._communitytype-static....npy. If temporal, loads static communities 
        in BIDS_dir/teneto_<version>/sub-.../func/communities/..._communitytype-temporal....npy 
 
        Returns
        -------

        Saves in ./BIDS_dir/derivatives/teneto/sub-NAME/func/tvc/temporal-network-measures/MEASURE/
        Load the measure with tenetoBIDS.load_network_measure
        """
        if not njobs:
            njobs = self.njobs
        self.add_history(inspect.stack()[0][3], locals(), 1)

        module_dict = inspect.getmembers(teneto.networkmeasures)
        # Remove all functions starting with __
        module_dict={m[0]:m[1] for m in module_dict if m[0][0:2]!='__'}
        # measure can be string or list
        if isinstance(measure, str):
            measure = [measure]
        # measure_params can be dictionaary or list of dictionaries
        if isinstance(measure_params, dict):
            measure_params = [measure_params]
        if measure_params and len(measure) != len(measure_params):
            raise ValueError('Number of identified measure_params (' + str(len(measure_params)) + ') differs from number of identified measures (' + str(len(measure)) + '). Leave black dictionary if default methods are wanted')

        # Check that specified measure is valid.
        flag = [n for n in measure if n not in module_dict.keys()]
        if flag:
            print('Specified measure(s): ' + ', '.join(flag) + ' not valid.')
        if not measure or flag:
            print('Following measures are valid (specified as string or list): \n - ' + '\n - '.join(module_dict.keys()))

        files = self.get_selected_files(quiet=1)


        if not load_tag:
            load_tag = ''

        if not save_tag:
            save_tag = ''
        else:
            save_tag = '_' + save_tag

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_networkmeasures,f,load_tag,save_tag,measure,measure_params,module_dict) for f in files if load_tag in f}
            for j in as_completed(job):
                j.result()

    def _run_networkmeasures(self,f,load_tag,save_tag,measure,measure_params,module_dict):
        # ADD MORE HERE (csv, json, nifti)
        if f.split('.')[-1] == 'npy':
            data = np.load(f)
        else:
            raise ValueError('derive can only load npy files at the moment')

        save_dir_base = '/'.join(f.split('/')[:-1]) + '/temporal-network-measures/'

        file_name = f.split('/')[-1].split('.')[0]

        for i, m in enumerate(measure):

            # The following 12 lines get the dimord
            if 'calc' in measure_params[i]:
                c = measure_params[i]['calc']
                cs = '_calc-' + c
            else:
                c = ''
                cs = ''
            if 'communities' in measure_params[i]:
                s = 'communities'
            else:
                s = ''
            dimord = teneto.utils.get_dimord(m,c,s)
            dimord_str = ''
            if dimord != 'unknown':
                dimord_str = '_dimord-' + dimord

            if 'communities' in measure_params[i]:
                if isinstance(measure_params[i]['communities'],str):
                    save_tag += '_communitytype-' + measure_params[i]['communities']
                    if measure_params[i]['communities'] == 'template':
                        measure_params[i]['communities'] = np.array(self.network_communities_['network_id'].values)
                    elif measure_params[i]['communities'] == 'static':
                        self.load_community_data('static',tag=file_name.split('tvc')[0].split('_'))
                        measure_params[i]['communities'] = np.squeeze(self.community_data_) 
                    elif measure_params[i]['communities'] == 'temporal':  
                        self.load_community_data('temporal',tag=file_name)
                        measure_params[i]['communities'] = np.squeeze(self.community_data_)
                    else: 
                        raise ValueError('Unknown community string')

            sname = m.replace('_','-')
            if not os.path.exists(save_dir_base + sname):
                #Received error when paralleled so throw a try/error 
                try: 
                    os.makedirs(save_dir_base + sname)
                except: 
                    pass
            
            save_name = file_name + '_' + sname + cs + dimord_str + save_tag
            netmeasure = module_dict[m](data,**measure_params[i])

            np.save(save_dir_base + sname + '/' + save_name, netmeasure)


    def set_bad_subjects(self,bad_subjects):

        if isinstance(bad_subjects,str):
            bad_subjects = [bad_subjects]

        for bs in bad_subjects:
            if bs in self.subjects:
                self.subjects.remove(bs)
            else:
                print('WARNING: subject: ' + str(bs) + ' is not found in TenetoBIDS.subjects')

        if not self.bad_subjects:
            self.bad_subjects = bad_subjects
        else:
            self.bad_subjects += bad_subjects


    def set_bad_files(self,bad_files):

        if isinstance(bad_files,str):
            bad_files = [bad_files]
        
        bad_files = [''.join(f.split('.')[0:-1]) for f in bad_files]

        if not self.bad_files:
            self.bad_files = bad_files
        else:
            self.bad_files += bad_files


    def set_confound_pipeline(self,confound_pipeline):
        """
        There may be times when the pipeline is updated (e.g. teneto) but you want the confounds from the preprocessing pipieline (e.g. fmriprep).
        To do this, you set the confound_pipeline to be the preprocessing pipeline where the confound files are.

        Parameters
        ----------

        confound_pipeline : str
            Directory in the BIDS_dir where the confounds file is.


        """

        self.add_history(inspect.stack()[0][3], locals(), 1)

        if not os.path.exists(self.BIDS_dir + '/derivatives/' + confound_pipeline):
            print('Specified direvative directory not found.')
            self.get_pipeline_alternatives()
        else:
            # Todo: perform check that pipeline is valid
            self.confound_pipeline = confound_pipeline



    def set_confounds(self,confounds,quiet=0):
        # This could be mnade better

        self.add_history(inspect.stack()[0][3], locals(), 1)

        file_list = self.get_confound_files(quiet=1)
        if isinstance(confounds,str):
            confounds = [confounds]

        for f in file_list:
            file_format = f.split('.')[-1]
            if  file_format == 'tsv':
                sub_confounds = list(pd.read_csv(f,delimiter='\t').keys())
            elif file_format == 'csv':
                sub_confounds = list(pd.read_csv(f,delimiter=',').keys())
            for c in confounds:
                if c not in sub_confounds:
                    print('Warning: the confound (' + c + ') not found in file: ' + f)

        self.confounds = confounds

    def set_network_communities(self,parcellation):
        """

        parcellation : str
            path to csv or name of default parcellation.

        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        # Sett if seperate subcortical atlas is specified
        if '+' in parcellation:
            # Need to add subcortical info to network_communities and network_communities_info_
            parcin = parcellation.split('+')
            parcellation = parcin[0]
            subcortical = parcin[1]
        else:
            subcortical = None

        net_path = teneto.__path__[0] + '/data/parcellation_defaults/' + parcellation + '_network.csv'
        if os.path.exists(parcellation):
            self.network_communities_ = pd.read_csv(parcellation,index_col=0)
            self.network_communities_info_ = self.network_communities_.drop_duplicates().sort_values('network_id').reset_index(drop=True)
            self.network_communities_info_['number_of_nodes'] = self.network_communities_.groupby('network_id').count()
        elif os.path.exists(net_path):
            self.network_communities_ = pd.read_csv(net_path,index_col=0)
            self.network_communities_info_ = self.network_communities_.drop_duplicates().sort_values('network_id').reset_index(drop=True)
            self.network_communities_info_['number_of_nodes'] = self.network_communities_.groupby('network_id').count()
        else:
            print('No (static) network community file found.')

        if subcortical:
            #Assuming only OH atlas exists for subcortical at the moment.
            node_num=21
            sub = pd.DataFrame(data={'Community': ['Subcortical (OH)']*node_num,'network_id':np.repeat(self.network_communities_['network_id'].max()+1,node_num)})
            self.network_communities_ = self.network_communities_.append(sub)
            self.network_communities_.reset_index(drop=True,inplace=True)


    def set_last_analysis_step(self,last_analysis_step):
        """
        The last analysis step is the final tag that is present in files.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        self.last_analysis_step = last_analysis_step

    def set_analysis_steps(self,analysis_step,add_step=False):
        """
        Specify which analysis steps are part of the selected files.

        Parameters
        -----------

        analysis_step : str or list
            Analysis tags that are found in the file names of interest. E.g. 'preproc' will only select files with 'preproc' in them.
        add_step : Bool
            If true, then anything in self.analysis_steps is already kept.

        Returns
        -------
        TenetoBIDS.analysis_steps gets updated.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if isinstance(analysis_step,str):
            if add_step:
                self.analysis_steps = self.analysis_steps + [analysis_step]
            else:
                self.analysis_steps = [analysis_step]
        elif isinstance(analysis_step,list):
            if add_step:
                self.analysis_steps += analysis_step
            else:
                self.analysis_steps = analysis_step

        else:
            raise ValueError('Invalud input')



    def set_pipeline(self,pipeline):
        """
        Specify the pipeline. See get_pipeline_alternatives to see what are avaialble. Input should be a string.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if not os.path.exists(self.BIDS_dir + '/derivatives/' + pipeline):
            print('Specified direvative directory not found.')
            self.get_pipeline_alternatives()
        else:
            # Todo: perform check that pipeline is valid
            self.pipeline = pipeline

    def set_pipeline_subdir(self,pipeline_subdir):
        self.add_history(inspect.stack()[0][3], locals(), 1)
#        if not os.path.exists(self.BIDS_dir + '/derivatives/' + self.pipeline + '/' + pipeline_subdir):
#            print('Specified direvative sub-directory not found.')
#            self.get_pipeline_subdir_alternatives()
#        else:
#            # Todo: perform check that pipeline is valid
        self.pipeline_subdir = pipeline_subdir


    def set_runs(self,runs):
        """
        Specify the runs which all selected files must include. See get_run_alternatives to see what are avaialble. Input can be string or list of strings.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if isinstance(runs,str):
            runs=[runs]
        if self.raw_data_exists:
            runs_in_dataset = self.BIDS.get_runs()
            if len(set(runs).intersection(runs_in_dataset))==len(runs):
                self.runs = sorted(list(runs))
            else:
                raise ValueError('Specified run(s) not founds in BIDS dataset')
        else:
            self.runs = sorted(list(runs))

    def set_sessions(self,sessions):
        """
        Specify the sessions which all selected files must include. See get_session_alternatives to see what are avaialble.  Input can be string or list of strings.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if isinstance(sessions,str):
            sessions=[sessions]
        if self.raw_data_exists:
            sessions_in_dataset = self.BIDS.get_sessions()
            if len(set(sessions).intersection(sessions_in_dataset))==len(sessions):
                self.sessions = sorted(list(sessions))
            else:
                raise ValueError('Specified session(s) not founds in BIDS dataset')
        else:
            self.sessions = sorted(list(tasks))

    def set_space(self,space):
        """
        Specify the space which all selected files must include. See get_space_alternatives to see what are avaialble.  Input can be string or list of strings.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)

        space_alternatives = self.get_space_alternatives(quiet=1)
        if space not in space_alternatives:
            raise ValueError('Specified space cannot be found for any subjects. Run TN.get_space_alternatives() to see the optinos in directories.')
        self.space = space

    def set_subjects(self,subjects=None):
        """
        Specify the subjects which are selected files for the analysis.   Input can be string or list of strings.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if isinstance(subjects,str):
            subjects=[subjects]
        # GEt from raw data or from derivative structure
        if self.raw_data_exists:
            subjects_in_dataset = self.BIDS.get_subjects()
            if len(set(subjects).intersection(subjects_in_dataset))==len(subjects):
                self.subjects = sorted(list(subjects))
            else:
                raise ValueError('Specified subject(s) not founds in BIDS dataset')
        elif not self.raw_data_exists:
            if not self.pipeline:
                raise ValueError('Pipeline must be set if raw_data_exists = False')
            elif not subjects:
                subjects_in_dataset = os.listdir(self.BIDS_dir + '/derivatives/' + self.pipeline)
                subjects_in_dataset = [f.split('sub-')[1] for f in subjects_in_dataset if os.path.isdir(self.BIDS_dir + '/derivatives/' + self.pipeline + '/' + f)]
                self.subjects = subjects_in_dataset
            else:
                self.subjects = subjects



    def set_tasks(self,tasks):
        """
        Specify the space which all selected files must include. See get_task_alternatives to see what are avaialble.  Input can be string or list of strings.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if isinstance(tasks,str):
            tasks=[tasks]
        if self.raw_data_exists:
            tasks_in_dataset = self.BIDS.get_tasks()
            if len(set(tasks).intersection(tasks_in_dataset))==len(tasks):
                self.tasks = sorted(list(tasks))
            else:
                raise ValueError('Specified task(s) not founds in BIDS dataset')
        else:
            self.tasks = sorted(list(tasks))

    def print_dataset_summary(self):

        """
        Prints information about the the BIDS data and the files currently selected.
        """

        print('--- DATASET INFORMATION ---')

        print('--- Subjects ---')
        if self.raw_data_exists:
            if self.BIDS.get_subjects():
                print('Number of subjects (in dataset): ' + str(len(self.BIDS.get_subjects())))
                print('Subjects (in dataset): ' + ', '.join(self.BIDS.get_subjects()))
            else:
                print('NO SUBJECTS FOUND (is the BIDS directory specified correctly?)')

        print('Number of subjects (selected): ' + str(len(self.subjects)))
        print('Subjects (selected): ' + ', '.join(self.subjects))
        if isinstance(self.bad_subjects,list):
            print('Bad subjects: ' + ', '.join(self.bad_subjects))
        else:
            print('Bad subjects: 0')

        print('--- Tasks ---')
        if self.raw_data_exists:
            if self.BIDS.get_tasks():
                print('Number of tasks (in dataset): ' + str(len(self.BIDS.get_tasks())))
                print('Tasks (in dataset): ' + ', '.join(self.BIDS.get_tasks()))
        if self.tasks:
            print('Number of tasks (selected): ' + str(len(self.tasks)))
            print('Tasks (selected): ' + ', '.join(self.tasks))
        else:
            print('No task names found')

        print('--- Runs ---')
        if self.raw_data_exists:
            if self.BIDS.get_runs():
                print('Number of runs (in dataset): ' + str(len(self.BIDS.get_runs())))
                print('Runs (in dataset): ' + ', '.join(self.BIDS.get_runs()))
        if self.runs:
            print('Number of runs (selected): ' + str(len(self.runs)))
            print('Rubs (selected): ' + ', '.join(self.runs))
        else:
            print('No run names found')


        print('--- Sessions ---')
        if self.raw_data_exists:
            if self.BIDS.get_sessions():
                print('Number of runs (in dataset): ' + str(len(self.BIDS.get_sessions())))
                print('Sessions (in dataset): ' + ', '.join(self.BIDS.get_sessions()))
        if self.sessions:
            print('Number of sessions (selected): ' + str(len(self.sessions)))
            print('Sessions (selected): ' + ', '.join(self.sessions))
        else:
            print('No session names found')

        print('--- PREPROCESSED DATA (Pipelines/Derivatives) ---')

        if not self.pipeline:
            print('Derivative pipeline not set. To set, run TN.set_pipeline()')
        else:
            print('Pipeline: ' + self.pipeline)
        if self.pipeline_subdir:
            print('Pipeline subdirectories: ' + self.pipeline_subdir)
        if not self.space:
            print('Space not set. To set, run TN.set_space()')
        else:
            print('Space: ' + self.space)
        if not self.parcellation:
            print('No parcellation specified. To set, run TN.set_parcellation()')
        else:
            print('Parcellation: ' + self.parcellation)

        selected_files = self.get_selected_files(quiet=1)
        if selected_files:
            print('--- SELECTED DATA ---')
            print('Numnber of selected files: ' + str(len(selected_files)))
            print('\n - '.join(selected_files))

    def save_aspickle(self, fname):
        if fname[-4:] != '.pkl':
            fname += '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_frompickle(cls,fname,reload_object=False):
        """
        Loaded saved instance of 

        fname : str
            path to pickle object (output of TenetoBIDS.save_aspickle) 
        reload_object : bool (default False)
            reloads object by calling teneto.TenetoBIDS (some information lost, for development)

        Returns 
        ------- 
            self : 
                TenetoBIDS instance 
        """
        if fname[-4:] != '.pkl':
            fname += '.pkl'
        with open(fname, 'rb') as f:
            lnet = pickle.load(f)
        if reload_object: 
            lnet = teneto.TenetoBIDS(lnet.BIDS_dir, pipeline=lnet.pipeline, pipeline_subdir=lnet.pipeline_subdir, parcellation=lnet.parcellation, space=lnet.space, subjects=lnet.subjects, sessions=lnet.sessions, runs=lnet.runs, tasks=lnet.tasks, last_analysis_step=lnet.last_analysis_step, analysis_steps=lnet.analysis_steps, bad_subjects=lnet.bad_subjects, confound_pipeline=lnet.confound_pipeline, raw_data_exists=lnet.raw_data_exists, njobs=lnet.njobs) 
        return lnet


    def load_parcellation_data(self,parcellation=None,tag=None):
        """
        Function returns the data created by. The default grabs all data in the teneto/../func/parcellation directory.

        **INPUT**

        parcellation : str
            Specify parcellation (optional). Default will grab everything that can be found.

        tag : str or list
            any additional tag that must be in file name. After the tag there must either be a underscore or period (following bids). 

        **RETURNS**

        :parcellation_data_: numpy array containing the parcellation data. Each file is appended to the first dimension of the numpy array.
        :parcellation_trialinfo_: pandas data frame containing the subject info (all BIDS tags) in the numpy array.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        data_list=[]
        trialinfo_list = []
        if parcellation:
            parc = parcellation
        if not self.parcellation:
            parc = ''
        else:
            parc = self.parcellation.split('_')[0]

        if not tag:
            tag = ['']
        if isinstance(tag,str):
            tag = [tag]


        for s in self.subjects:
            # Define base folder
            base_path = self.BIDS_dir + '/derivatives/' + self.pipeline
            base_path += '/sub-' + s + '/func/parcellation/'
            file_list=os.listdir(base_path)
            for f in file_list:
                ok = True
                if not any(['run-' + t in f for t in self.runs]) and self.runs:
                    ok = False
                if not any(['task-' + t in f for t in self.tasks if t]) and self.tasks:
                    ok = False
                if not any(['ses-' + t in f for t in self.sessions if t]) and self.sessions:
                    ok = False
                if self.last_analysis_step: 
                    if not f.split('.')[-2].endswith(self.last_analysis_step):
                        ok = False
                # Include only if all analysis step tags are present
                if parc in f and all([t + '_' in f or t + '.' in f for t in tag]) and ok:
                    # Get all BIDS tags. i.e. in 'sub-AAA', get 'sub' as key and 'AAA' as item.
                    bid_tags=re.findall('[a-zA-Z]*-',f)
                    bids_tag_dict = {}
                    for t in bid_tags:
                        key = t[:-1]
                        bids_tag_dict[key]=re.findall(t+'[A-Za-z0-9.,*+]*',f)[0].split('-')[-1]
                    if f.split('.')[-1] == 'npy':
                        data = np.load(base_path+f)
                        data_list.append(data)
                        trialinfo = pd.DataFrame(bids_tag_dict,index=[0])
                        trialinfo_list.append(trialinfo)
                    else:
                        print('Warning: Could not find data for a subject')

            self.parcellation_data_ = np.array(data_list)
            if trialinfo_list:
                out_trialinfo = pd.concat(trialinfo_list)
                out_trialinfo.reset_index(inplace=True,drop=True)
                self.parcellation_trialinfo_ = out_trialinfo

    #timelocked average
    #np.stack(a['timelocked-tvc'].values).mean(axis=0)

    def load_timelocked_data(self,measure,calc=None,tag=None,avg=None,event=None,groupby=None): 

        if not calc:
            calc = ''
        else:
            calc = 'calc-' + calc

        if not tag:
            tag = ['']
        elif isinstance(tag,str): 
            tag = [tag]

        if avg: 
            finaltag = 'timelocked_avg.npy'
        else:
            finaltag =  'timelocked.npy'
            
        self.add_history(inspect.stack()[0][3], locals(), 1)
        trialinfo_list = []
        data_list = []
        std_list = []
        for s in self.subjects:

            base_path = self.BIDS_dir + '/derivatives/' + self.pipeline
            if measure == 'tvc':
                base_path += '/sub-' + s + '/func/tvc/timelocked/'
            else:
                base_path += '/sub-' + s + '/func/tvc/temporal-network-measures/' + measure + '/timelocked/'

            if not os.path.exists(base_path):
                print('Warning: cannot find data for subject: ' + s)

            for f in os.listdir(base_path):
                if os.path.isfile(base_path + f) and f.split('.')[-1] == 'npy':
                    if calc in f and all([t + '_' in f or t + '.' in f for t in tag]) and finaltag in f:
                        if avg: 
                            f = f.split('_avg')[0]
                            f_suff = '.npy'
                        else: 
                            f_suff = ''
                        bids_tags=re.findall('[a-zA-Z]*-',f)
                        bids_tag_dict = {}
                        for t in bids_tags:
                            key = t[:-1]
                            bids_tag_dict[key]=re.findall(t+'[A-Za-z0-9.,*+]*',f)[0].split('-')[-1]
                        trialinfo_eventinfo = pd.read_csv(base_path + '.'.join(f.split('timelocked')[0:-1]) + 'timelocked_trialinfo.csv')
                        trialinfo = pd.DataFrame(bids_tag_dict,index=np.arange(0,len(trialinfo_eventinfo)))
                        trialinfo = pd.concat([trialinfo,trialinfo_eventinfo],axis=1)
                        trialinfo_list.append(trialinfo)
                        if avg:
                            data_list.append(np.load(base_path + f + '_avg.npy'))
                            std_list.append(np.load(base_path + f + '_std.npy'))
                        else: 
                            data_list.append(np.load(base_path + f))
        if avg: 
            self.timelocked_data_ = {}
            self.timelocked_data_['avg'] = np.stack(np.array(data_list))
            self.timelocked_data_['std'] = np.stack(np.array(std_list))
        else: 
            self.timelocked_data_ = np.stack(np.array(data_list))
        

        if trialinfo_list:
            out_trialinfo = pd.concat(trialinfo_list)
            out_trialinfo = out_trialinfo.drop('Unnamed: 0',axis=1)
            out_trialinfo.reset_index(inplace=True,drop=True)
            self.timelocked_trialinfo_ = out_trialinfo

    def load_tvc_data(self,tag=None):
        """
        Function loads time-varying connectivity estimates created by the TenetoBIDS.derive function.
        The default grabs all data (in numpy arrays) in the teneto/../func/tvc/ directory.
        Data is placed in teneto.tvc_data_

        **INPUT**

        tag : str or list
            any additional tag that must be in file name. After the tag there must either be a underscore or period (following bids). 

        timelocked : bool 
            Load timelocked data if true.

        **RETURNS**

        tvc_data_ : numpy array
            Containing the parcellation data. Each file is appended to the first dimension of the numpy array.
        tvc_trialinfo_ : pandas data frame
            Containing the subject info (all BIDS tags) in the numpy array.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        data_list=[]
        trialinfo_list = []

        if not tag:
            tag = ['']
        elif isinstance(tag,str): 
            tag = [tag]

        for s in self.subjects:
            # Define base folder
            base_path = self.BIDS_dir + '/derivatives/' + self.pipeline
            base_path += '/sub-' + s + '/func/tvc/'
            file_list=os.listdir(base_path)
            for f in file_list:
                # Include only if all analysis step tags are present
                if os.path.isfile(base_path + f) and all([t + '_' in f or t + '.' in f for t in tag]):
                    # Get all BIDS tags. i.e. in 'sub-AAA', get 'sub' as key and 'AAA' as item.
                    bid_tags=re.findall('[a-zA-Z]*-',f)
                    bids_tag_dict = {}
                    for t in bid_tags:
                        key = t[:-1]
                        bids_tag_dict[key]=re.findall(t+'[A-Za-z0-9.,*+]*',f)[0].split('-')[-1]
                    trialinfo = pd.DataFrame(bids_tag_dict,index=[0])
                    trialinfo_list.append(trialinfo)
                    if f.split('.')[-1] == 'npy':
                        data = np.load(base_path+f)
                        data_list.append(data)
                    else:
                        print('Warning: Could not find data for a subject (expecting numpy array)')
            self.tvc_data_ = np.array(data_list)
            if trialinfo_list:
                out_trialinfo = pd.concat(trialinfo_list)
                out_trialinfo.reset_index(inplace=True,drop=True)
                self.tvc_trialinfo_ = out_trialinfo



    def load_network_measure(self,measure,calc=None,tag=None):
        self.add_history(inspect.stack()[0][3], locals(), 1)
        data_list=[]
        trialinfo_list = []

        measure = measure.replace('_','-')

        if not calc:
            calc = ''
        else:
            calc = 'calc-' + calc

        if not tag:
            tag = ['']
        elif isinstance(tag,str): 
            tag = [tag]

        for s in self.subjects:
            # Define base folder
            base_path = self.BIDS_dir + '/derivatives/' + self.pipeline
            base_path += '/sub-' + s + '/func/tvc/temporal-network-measures/' + measure + '/'
            measure_sub = measure
            # Get files
            if os.path.exists(base_path):
                file_list=os.listdir(base_path)
                # Get tags in filename
                for f in file_list:
                    if os.path.isfile(base_path + f):
                        if calc in f and all([t + '_' in f or t + '.' in f for t in tag]):
                            bids_tags=re.findall('[a-zA-Z]*-',f)
                            bids_tag_dict = {}
                            for t in bids_tags:
                                key = t[:-1]
                                bids_tag_dict[key]=re.findall(t+'[A-Za-z0-9.,*+]*',f)[0].split('-')[-1]
                                if bids_tag_dict[key].endswith('.npy'): 
                                    bids_tag_dict[key] = bids_tag_dict[key].split('.npy')[0]        
                                elif bids_tag_dict[key].endswith('.pkl'): 
                                    bids_tag_dict[key] = bids_tag_dict[key].split('.pkl')[0]        
                            # Get data
                            if f.split('.')[-1] == 'pkl':
                                df = pd.read_pickle(base_path+f)
                                data = df[measure_sub].values
                                trialinfo = df.drop(measure_sub, 1)
                                for k in bids_tag_dict.keys():
                                    trialinfo[k] = bids_tag_dict[k]
                                trialinfo_list.append(trialinfo)
                                for d in data:
                                    data_list.append(d)
                            elif f.split('.')[-1] == 'npy':
                                data = np.load(base_path+f)
                                data_list.append(data)
                                trialinfo = pd.DataFrame(bids_tag_dict,index=[0])
                                trialinfo_list.append(trialinfo)

                            else:
                                print('Warning: Could not find pickle data')

        flag = 0
        if not all([n.shape == m.shape for n in data_list for m in data_list]): 
            if not all([n.shape[-1] == m.shape[-1] for n in data_list for m in data_list]) and flag==0:
                shape_max = max([n.shape[-1] for n in data_list])
                for n in range(len(data_list)): 
                    if shape_max - data_list[n].shape[-1] > 0: 
                        ladd = np.ceil((shape_max - data_list[n].shape[-1]) / 2)
                        radd = np.floor((shape_max - data_list[n].shape[-1]) / 2)
                        if len(data_list[n].shape) == 2:  
                            data_list[n] = np.hstack([np.array(np.zeros([data_list[n].shape[0],int(ladd)])*np.nan,ndmin=2),data_list[n],np.array(np.zeros([data_list[n].shape[0],int(radd)])*np.nan,ndmin=2)])
                        if len(data_list[n].shape) == 3:
                            data_list[n] = np.dstack([np.array(np.zeros([data_list[n].shape[0],data_list[n].shape[1],int(ladd)])*np.nan,ndmin=3),data_list[n],np.array(np.zeros([data_list[n].shape[0],data_list[n].shape[1],int(radd)])*np.nan,ndmin=3)])
                        if len(data_list[n].shape) == 1:
                            data_list[n] = np.hstack([np.array(np.zeros([int(ladd)])*np.nan,ndmin=1),data_list[n],np.array(np.zeros([int(radd)])*np.nan,ndmin=1)])
                          
            ax_len = np.max([len(n.shape) for n in data_list])
            for dim in range(ax_len-1):
                if not all([n.shape[dim] == m.shape[dim] for n in data_list for m in data_list]) and flag == 0:
                    print('TENETO-WARNING: unequal non-time dimensions. Returning as list instead of array.')
                    flag = 1 
                    self.networkmeasure_ = data_list

        if flag == 0:
            self.networkmeasure_ = np.array(data_list)
        if trialinfo_list:
            out_trialinfo = pd.concat(trialinfo_list)
            out_trialinfo.reset_index(inplace=True,drop=True)
            self.trialinfo_ = out_trialinfo



    def make_timelocked_events(self, measure, event_names, event_onsets, toi, calc=None, tag=None, avg=None, offset=0):
        """
        Creates time locked time series of <measure>. Measure must have time in its -1 axis.

        Parameters
        -----------

        measure : str
            temporal network measure that should already exist in the teneto/[subject]/tvc/network-measures directory
        event_names : str or list
            what the event is called (can be list of multiple event names). Can also be TVC to create time-locked tvc. 
        event_onsets: list
            List of onset times (can be list of list for multiple events)
        toi : array
            +/- time points around each event. So if toi = [-10,10] it will take 10 time points before and 10 time points after
        calc : str
            type of network measure calculation.
        tag : str or list
            any additional tag that must be in file name. After the tag there must either be a underscore or period (following bids). 
        offset : int 
            If derive uses a method that has a sliding window, then the data time-points are reduced. Offset should equal half of the window-1. So if the window is 7, offset is 3. This corrects for the missing time points. 

        Note
        ----
        Currently no ability to loop over more than one measure

        Note
        -----
        Events that do not completely fit the specified time period (e.g. due to at beginning/end of data) get ignored.  

        Returns
        -------
        Creates a time-locked output placed in BIDS_dir/derivatives/teneto_<version>/../tvc/temporal-network-measures/<networkmeasure>/timelocked/
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        #Make sure that event_onsets and event_names are lists
        #if  np.any(event_onsets[0]):
        #    event_onsets = [e.tolist() for e in event_onsets[0]]   
        if isinstance(event_onsets[0],int) or isinstance(event_onsets[0],float): 
            event_onsets = [event_onsets]
        if isinstance(event_names,str): 
            event_names = [event_names]
        # Combine the different events into one list                    
        event_onsets_combined = list(itertools.chain.from_iterable(event_onsets))
        event_names_list = [[e]*len(event_onsets[i]) for i,e in enumerate(event_names)]
        event_names_list = list(itertools.chain.from_iterable(event_names_list))
        #time_index = np.arange(toi[0],toi[1]+1)

        if not calc:
            calc = ''
        else:
            calc = 'calc-' + calc

        if not tag:
            tag = ['']
        elif isinstance(tag,str): 
            tag = [tag]

        for s in self.subjects:

            base_path = self.BIDS_dir + '/derivatives/' + self.pipeline + '/sub-' + s + '/func/'
            if measure == 'tvc':    
                base_path += 'tvc/'
            elif not measure: 
                pass 
            else:
                base_path += 'tvc/temporal-network-measures/' + measure + '/'

            if not os.path.exists(base_path):
                print('Warning: cannot find data for subject: ' + s)

            if not os.path.exists(base_path + '/timelocked/'):
                os.makedirs(base_path + '/timelocked/')

            for f in os.listdir(base_path):
                if os.path.isfile(base_path + f):
                    if calc in f and all([t + '_' in f or t + '.' in f for t in tag]):
                        bids_tags=re.findall('[a-zA-Z]*-',f)
                        bids_tag_dict = {}
                        for t in bids_tags:
                            key = t[:-1]
                            bids_tag_dict[key]=re.findall(t+'[A-Za-z0-9.,*+]*',f)[0].split('-')[-1]
                            
                        self_measure = np.load(base_path + '/' + f)
                        # make time dimensions the first dimension
                        self_measure = self_measure.transpose([len(self_measure.shape)-1] + list(np.arange(0,len(self_measure.shape)-1)))
                        tl_data = []
                        for e in event_onsets_combined:
                            # Ignore events which do not completely fit defined segment
                            if e+toi[0]-offset<0 or e+toi[1]-offset>=self_measure.shape[0]: 
                                pass
                            else: 
                                tmp = self_measure[e+toi[0]-offset:e+toi[1]+1-offset]
                                # Make time dimension last dimension
                                tmp = tmp.transpose(list(np.arange(1,len(self_measure.shape))) + [0])
                                tl_data.append(tmp)
                        tl_data = np.stack(tl_data)
                        if avg: 
                            df=pd.DataFrame(data={'event': '+'.join(list(set(event_names_list))), 'event_onset': [event_onsets_combined]})
                        else: 
                            df=pd.DataFrame(data={'event': event_names_list, 'event_onset': event_onsets_combined})
    
                        # Save output
                        save_dir_base = base_path + 'timelocked/'
                        file_name = f.split('/')[-1].split('.')[0] + '_events-' + '+'.join(event_names) + '_timelocked_trialinfo'
                        df.to_csv(save_dir_base + file_name + '.csv')
                        file_name = f.split('/')[-1].split('.')[0] + '_events-' + '+'.join(event_names) + '_timelocked'
                        if avg:
                            tl_data_std = np.std(tl_data,axis=0)
                            tl_data = np.mean(tl_data,axis=0) 
                            np.save(save_dir_base + file_name + '_std',tl_data_std)
                            np.save(save_dir_base + file_name + '_avg',tl_data)
                        else: 
                            np.save(save_dir_base + file_name,tl_data)
                        

    def load_participant_data(self):
        """
        Loads the participanets.tsv file that is placed in BIDS_dir as participants_.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        self.participants_ = pd.read_csv(self.BIDS_dir + 'participants.tsv',delimiter='\t')
