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
import json
import pickle
import traceback
import nilearn
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d
import time 
from ..utils.bidsutils import load_tabular_file, get_bids_tag, get_sidecar, confound_matching, process_exclusion_criteria, drop_bids_suffix, make_directories
import pandas as pd 
from .network import TemporalNetwork

#class NetworkMeasures:
#    def __init__(self,**kwargs):
#        pass

    #def temporal_degree_centrality(self,**kwargs):
    #    print(self)
    #    print(teneto.networkmeasures.temporal_degree_centrality(self,**kwargs))



class TenetoBIDS:

    bids_derivatives_rc_version = '<rc1.0'

    def __init__(self, BIDS_dir, pipeline=None, pipeline_subdir=None, parcellation=None, bids_tags=None, bids_suffix=None, bad_subjects=None, confound_pipeline=None, raw_data_exists=True, njobs=None):
        """
        Parameters
        ----------

        BIDS_dir : str
            string to BIDS directory
        pipeline : str
            the directory that is in the BIDS_dir/derivatives/<pipeline>/
        pipeline_subdir : str, optional
            the directory that is in the BIDS_dir/derivatives/<pipeline>/sub-<subjectnr/[ses-<sesnr>]/func/<pipeline_subdir>
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
        if self.BIDS_dir[-1] != '/':
            self.BIDS_dir = self.BIDS_dir + '/'




        if not bids_suffix:
            self.bids_suffix = ''
        else:
            self.bids_suffix = bids_suffix

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

        self.set_bids_tags() 
        if bids_tags: 
            self.set_bids_tags(bids_tags)

        # Set data variables to Nones
        self.tvc_data_ = []
        self.parcellation_data_ = []
        self.participent_data_ = []
        self.temporalnetwork_data_ = []
        self.fc_data_ = []
        self.tvc_trialinfo_ = []
        self.parcellation_trialinfo_ = []
        self.temporalnetwork_trialinfo_ = []
        self.fc_trialinfo_ = []

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
            any additional tag that will be placed in the saved file name. Will be placed as 'desc-[tag]'

        Returns 
        ------- 
        dfc : files 
            saved in .../derivatives/teneto/sub-xxx/tvc/..._tvc.npy
        """
        if not njobs:
            njobs = self.njobs
        self.add_history(inspect.stack()[0][3], locals(), 1)

        files = self.get_selected_files(quiet=1)
        confound_files = self.get_selected_files(quiet=1, pipeline='confound')
        if confound_files:
            confounds_exist = True
        else: 
            confounds_exist = False
        if not confound_corr_report: 
            confounds_exist = False 

        if not tag:
            tag = ''
        else:   
            tag = 'desc-' + tag

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_derive,f,i,tag,params,confounds_exist,confound_files) for i,f in enumerate(files) if f}
            for j in as_completed(job):
                j.result()

        if update_pipeline == True:
            if not self.confound_pipeline and len(self.get_selected_files(quiet=1, pipeline='confound')) > 0:
                self.set_confound_pipeline = self.pipeline
            self.set_pipeline('teneto_' + teneto.__version__)
            self.set_pipeline_subdir('tvc')
            self.set_bids_suffix('tvcconn')

    def _run_derive(self,f,i,tag,params,confounds_exist,confound_files):
        """
        Funciton called by TenetoBIDS.derive for parallel processing.
        """
        data = load_tabular_file(f, index_col=True, header=True)

        fs, _ = drop_bids_suffix(f)
        save_name, save_dir, _ = self._save_namepaths_bids_derivatives(fs, tag, 'tvc', 'tvcconn')
        if 'weight-var' in params.keys():
            if params['weight-var'] == 'from-subject-fc':
                fc_files = self.get_selected_files(quiet=1, pipeline='functionalconnectivity', forfile=f)
                if len(fc_files) == 1:
                    # Could change to load_data call
                    params['weight-var'] = load_tabular_file(fc_files[0]).values
                else:
                    raise ValueError('Cannot correctly find FC files')

        if 'weight-mean' in params.keys():
            if params['weight-mean'] == 'from-subject-fc':
                fc_files = self.get_selected_files(quiet=1, pipeline='functionalconnectivity', forfile=f)
                if len(fc_files) == 1:
                    # Could change to load_data call
                    params['weight-mean'] = load_tabular_file(fc_files[0]).values
                else:
                    raise ValueError('Cannot correctly find FC files')

        params['report'] = 'yes'
        params['report_path'] =  save_dir + '/report/'
        params['report_filename'] =  save_name + '_derivationreport.html'

        if not os.path.exists(params['report_path']):
            os.makedirs(params['report_path'])

        dfc = teneto.derive.derive(data.values, params)
        dfc_net = TemporalNetwork(from_array=dfc, nettype='wu')
        dfc_net.network.to_csv(save_dir + save_name + '.tsv', sep='\t')

        sidecar = get_sidecar(f) 
        sidecar['tvc'] = params
        if 'weight-var' in sidecar['tvc']:
            sidecar['tvc']['weight-var'] = True
            sidecar['tvc']['fc source'] = fc_files
        if 'weight-mean' in sidecar['tvc']:
            sidecar['tvc']['weight-mean'] = True
            sidecar['tvc']['fc source'] = fc_files  
        sidecar['tvc']['inputfile'] = f
        sidecar['tvc']['description'] = 'Time varying connectivity information.'
        with open(save_dir + save_name  + '.json', 'w') as fs:
            json.dump(sidecar, fs)

        if confounds_exist:
            analysis_step = 'tvc-derive'
            df = pd.read_csv(confound_files[i],sep='\t')
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


    def set_bids_tags(self,indict=None):
        if not hasattr(self,'bids_tags'):  
            #print(hasattr(self,'bids_tags'))
            # Set defaults
            self.bids_tags = {}
            self.bids_tags['sub'] = 'all'
            self.bids_tags['run'] = 'all'
            self.bids_tags['task'] = 'all'
            self.bids_tags['ses'] = 'all'
            self.bids_tags['desc'] = None   
        if indict: 
            for d in indict: 
                self.bids_tags[d] = indict[d]
                if not isinstance(self.bids_tags[d],list):
                    self.bids_tags[d] = [self.bids_tags[d]]


        if 'sub' in self.bids_tags:
            if self.bids_tags['sub'] == 'all':
                if self.raw_data_exists:
                    self.bids_tags['sub'] = self.BIDS.get_subjects()
                else:
                    self.bids_tags['sub'] = self.get_tags('sub')
        if 'ses' in self.bids_tags:
            if self.bids_tags['ses'] == 'all' and self.raw_data_exists:
                self.bids_tags['ses'] = self.BIDS.get_sessions()
            elif not self.raw_data_exists:
                self.bids_tags['ses'] = self.get_tags('ses')  
        if 'task' in self.bids_tags:      
            if self.bids_tags['task'] == 'all' and self.raw_data_exists:
                self.bids_tags['task'] = self.BIDS.get_tasks()
            elif not self.raw_data_exists:
                self.bids_tags['task'] = self.get_tags('task')
        if 'run' in self.bids_tags:
            if self.bids_tags['run'] == 'all' and self.raw_data_exists:
                self.bids_tags['run'] = self.BIDS.get_runs()
            elif not self.raw_data_exists:
                self.bids_tags['run'] = self.get_tags('run')

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
            sf, _ = drop_bids_suffix(f)            
            save_name, save_dir, _ = self._save_namepaths_bids_derivatives(sf,'','fc','conn')
            data = load_tabular_file(f)
            R = data.transpose().corr()
            R.to_csv(save_dir + save_name + '.tsv', sep='\t')
            return R.values


    def _save_namepaths_bids_derivatives(self,f,tag,save_directory,suffix=None):
        """
        Creates output directory and output name

        Paramters 
        ---------   
        f : str
            input files, includes the file bids_suffix
        tag : str
            what should be added to f in the output file.
        save_directory : str
            additional directory that the output file should go in
        suffix : str
            add new suffix to data

        Returns 
        -------
        save_name : str 
            previous filename with new tag 
        save_dir : str
            directory where it will be saved 
        base_dir : str 
            subjective base directory (i.e. derivatives/teneto/func[/anythingelse/])

        """
        file_name = f.split('/')[-1].split('.')[0]
        if tag != '':
            tag = '_' + tag
        if suffix: 
            file_name, _ = drop_bids_suffix(file_name)
            save_name = file_name + tag
            save_name += '_' + suffix
        else: 
            save_name = file_name + tag
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


    def get_tags(self,tag,quiet=1):
        """
        Returns which tag alternatives can be identified in the BIDS derivatives structure. 
        """
        if not self.pipeline:
            print('Please set pipeline first.')
            self.get_pipeline_alternatives(quiet)
        else:
            if tag == 'sub': 
                tag_alternatives = [f.split('sub-')[1] for f in os.listdir(self.BIDS_dir + '/derivatives/' + self.pipeline)]
            elif tag == 'ses': 
                tag_alternatives = []
                for sub in self.bids_tags['sub']: 
                    tag_alternatives += [f.split('ses-')[1] for f in os.listdir(self.BIDS_dir + '/derivatives/' + self.pipeline + '/' + 'sub-' + sub) if 'ses' in f]
                tag_alternatives = set(tag_alternatives)
            else: 
                files = self.get_selected_files(quiet=1)
                tag_alternatives = []
                for f in files:
                    f = f.split('.')[0]
                    f = f.split('/')[-1]
                    tag_alternatives += [t.split('-')[1] for t in f.split('_') if t.split('-')[0] == tag]
                tag_alternatives = set(tag_alternatives)
            if quiet == 0:
                print(tag + ' alternatives: ' + ', '.join(tag_alternatives))
            return list(tag_alternatives)

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
            for s in self.bids_tags['sub']:
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

        for s in self.bids_tags['sub']:
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


    def get_selected_files(self, pipeline='pipeline', forfile=None, quiet=0, accepted_fileformats=['.tsv', '.nii.gz']):
        """
        Parameters
        ----------
        pipeline : string 
            can be \'pipeline\' (main analysis pipeline, self in tnet.set_pipeline) or \'confound\' (where confound files are, set in tnet.set_confonud_pipeline()), 
            \'functionalconnectivity\'
        quiet: int
            If 1, prints results. If 0, no results printed.
        forfile : str or dict 
            A filename or dictionary of file tags. If this is set, only files that match that subject

        Returns
        -------
        found_files : list
            The files which are currently selected with the current using the set pipeline, pipeline_subdir, space, parcellation, tasks, runs, subjects etc. There are the files that will generally be used if calling a make_ function.
        """
        # This could be mnade better
        file_dict = dict(self.bids_tags)
        if forfile: 
            if isinstance(forfile, str): 
                forfile = get_bids_tag(forfile,'all')
            for n in forfile.keys(): 
                file_dict[n] = [forfile[n]]
        non_entries = []
        for n in file_dict: 
            if not file_dict[n]: 
                non_entries.append(n)
        for n in non_entries: 
            file_dict.pop(n)

        # Only keep none empty elemenets
        file_components = []
        for k in ['sub', 'ses', 'task', 'run']:
            if k in file_dict: 
                file_components.append([k + '-' + t for t in file_dict[k]])

        file_list = list(itertools.product(*file_components))

        # Specify main directory
        if pipeline == 'pipeline':
            mdir = self.BIDS_dir + '/derivatives/' + self.pipeline
        elif pipeline == 'confound' and self.confound_pipeline:  
            mdir = self.BIDS_dir + '/derivatives/' + self.confound_pipeline
        elif pipeline == 'confound':             
            mdir = self.BIDS_dir + '/derivatives/' + self.pipeline
        elif pipeline == 'functionalconnectivity':             
            mdir = self.BIDS_dir + '/derivatives/teneto_' + teneto.__version__ 
        else: 
            raise ValueError('unknown request')

        found_files = []

        for f in file_list:
            wdir = str(mdir)
            sub = [t for t in f if t.startswith('sub')]
            ses = [t for t in f if t.startswith('ses')]
            wdir += '/' + sub[0] + '/'
            if ses:
                wdir += '/' + ses[0] + '/'
            wdir += '/func/'

            if pipeline == 'pipeline':
                wdir += '/' + self.pipeline_subdir + '/'
                fileending = [self.bids_suffix + f for f in accepted_fileformats]
            elif pipeline == 'functionalconnectivity':
                wdir += '/fc/'
                fileending = ['conn' + f for f in accepted_fileformats]
            elif pipeline == 'confound':
                fileending = ['confounds' + f for f in accepted_fileformats]
                
    
            if os.path.exists(wdir):
                # make filenames
                found = [] 
                # Check that the tags are in the specified bids tags
                for ff in os.listdir(wdir): 
                    ftags = get_bids_tag(ff,'all')
                    t = [t for t in ftags if t in file_dict and ftags[t] in file_dict[t]]
                    if len(t) == len(file_dict): 
                        found.append(ff)
                found = [f for f in found for e in fileending if f.endswith(e)]
                # Include only if all analysis step tags are present
                # Exclude if confounds tag is present
                if pipeline == 'confound':
                    found = [i for i in found if '_confounds' in i]
                else: 
                    found = [i for i in found if '_confounds' not in i]
                # Make full paths
                found = list(map(str.__add__,[re.sub('/+','/',wdir)]*len(found),found))
                # Remove any files in bad files (could add json subcar reading here)
                found = [i for i in found if not any([bf in i for bf in self.bad_files])]
                if found:
                    found_files += found

            if quiet==-1:
                print(wdir)

        found_files = list(set(found_files))
        if quiet == 0:
            print(found_files)
        return found_files


    def set_exclusion_file(self,confound,exclusion_criteria,confound_stat='mean'):
        """
        Excludes subjects given a certain exclusion criteria.

        Parameters
        ----------
            confound : str or list
                string or list of confound name(s) from confound files
            exclusion_criteria  : str or list
                for each confound, an exclusion_criteria should be expressed as a string. It starts with >,<,>= or <= then the numerical threshold. Ex. '>0.2' will entail every subject with the avg greater than 0.2 of confound will be rejected.
            confound_stat : str or list
                Can be median, mean, std. How the confound data is aggregated (so if there is a meaasure per time-point, this is averaged over all time points. If multiple confounds specified, this has to be a list.).
        Returns
        --------
            calls TenetoBIDS.set_bad_files with the files meeting the exclusion criteria.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if isinstance(confound,str):
            confound = [confound]
        if isinstance(exclusion_criteria,str):
            exclusion_criteria = [exclusion_criteria]
        if isinstance(confound_stat,str):
            confound_stat = [confound_stat]
        if len(exclusion_criteria)!=len(confound):
            raise ValueError('Same number of confound names and exclusion criteria must be given')
        if len(confound_stat)!=len(confound):
            raise ValueError('Same number of confound names and confound stats must be given')
        rel, crit = process_exclusion_criteria(exclusion_criteria)
        files = sorted(self.get_selected_files(quiet=1))
        confound_files = sorted(self.get_selected_files(quiet=1, pipeline='confound'))
        files, confound_files = confound_matching(files, confound_files)
        bad_files = []
        bs = 0
        foundconfound = []
        for s, cfile in enumerate(confound_files):
            df = load_tabular_file(cfile)
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
                    foundconfound.append(confound[i])
            if found_bad_subject:
                bad_files.append(files[s])
                bs += 1
        self.set_bad_files(bad_files, reason='excluded file (confound over specfied stat threshold)')
        for i, f in enumerate(bad_files): 
            sidecar = get_sidecar(f)
            sidecar['file_exclusion'] = {}
            sidecar['exclusion_reason'] = confound[i]
            sidecar['confound'] = confound
            sidecar['threshold'] = exclusion_criteria
            for af in ['.tsv','.nii.gz']: 
                f = f.split(af)[0] 
            f += '.json'
            with open(f, 'w') as fs:
                json.dump(sidecar, fs)
        print('Removed ' + str(bs) + ' files from inclusion.')

    def set_exclusion_timepoint(self,confound,exclusion_criteria,replace_with,tol=1,overwrite=True,desc=None):
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
            tol : float 
                Tolerance of exlcuded time-points allowed before becoming a bad subject. 
            overwrite : bool (default=True)
                If true, if their are files in the teneto derivatives directory with the same name, these will be overwritten with this step.
                The json sidecar is updated with the new information about the file. 
            desc : str
                String to add desc tag to filenames if overwrite is set to true. 

        Returns
        ------
            Loads the TenetoBIDS.selected_files and replaces any instances of confound meeting the exclusion_criteria with replace_with.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        if isinstance(confound,str):
            confound = [confound]
        if isinstance(exclusion_criteria,str):
            exclusion_criteria = [exclusion_criteria]
        if len(exclusion_criteria)!=len(confound):
            raise ValueError('Same number of confound names and exclusion criteria must be given')
        rel, crit = process_exclusion_criteria(exclusion_criteria)
        files = sorted(self.get_selected_files(quiet=1))
        confound_files = sorted(self.get_selected_files(quiet=1, pipeline='confound'))
        files, confound_files = confound_matching(files, confound_files)
        bad_files = []
        for i, cfile in enumerate(confound_files):
            data = load_tabular_file(files[i]).values
            df = load_tabular_file(cfile)
            ind = []
            # Is set to 1 if subject should be saved ("goodsubject")
            gs=1
            # Can't interpolate values if nanind is at the beginning or end. So keep these as their original values. 
            for ci,c in enumerate(confound):
                ind = df[rel[ci](df[c],crit[ci])].index
                if replace_with == 'cubicspline': 
                    if 0 in ind: 
                        ind = np.delete(ind,np.where(ind==0))
                    if df.index.max(): 
                        ind = np.delete(ind,np.where(ind==df.index.max()))             
                data[:,ind] = np.nan
            nanind = np.where(np.isnan(data[0,:]))[0]
            badpoints_n = len(nanind)
            # Bad file if the number of ratio bad points are greater than the tolerance. 
            if badpoints_n / np.array(len(df)) > tol: 
                bad_files.append(files[i])
                gs = 0
            nonnanind = np.where(np.isnan(data[0,:])==0)[0]
            nanind = nanind[nanind>nonnanind.min()]
            nanind = nanind[nanind<nonnanind.max()]
            if replace_with == 'cubicspline':
                for n in range(data.shape[0]):
                    interp = interp1d(nonnanind,data[n,nonnanind],kind='cubic')
                    data[n,nanind] = interp(nanind)
            # only save if the subject is not excluded 
            data = pd.DataFrame(data) 
            sname, _ = drop_bids_suffix(files[i])
            # Move files to teneto derivatives if the pipeline isn't already set to it
            if self.pipeline != 'teneto_' + teneto.__version__:
                sname = sname.split('/')[-1]
                spath = self.BIDS_dir + '/derivatives/' + 'teneto_' + teneto.__version__ + '/'
                tags = get_bids_tag(sname, ['sub','ses'])
                spath += 'sub-' + tags['sub'] + '/'
                if 'ses' in tags:
                    spath += 'ses-' + tags['ses'] + '/'
                spath += 'func/'
                if self.pipeline_subdir:
                    spath += self.pipeline_subdir + '/'
                make_directories(spath)
                sname = spath + sname
            if 'desc' in sname and desc: 
                desctag = get_bids_tags(sname.split('/')[-1], 'desc')
                sname = ''.join(sname.split('desc-' + desctag['desc']))
                sname + '_desc-' + desc
            if os.path.exists(sname + self.bids_suffix + '.tsv') and overwrite==False: 
                raise ValueError('overwrite is set to False, but non-unique filename. Set unique desc tag')
            data.to_csv(sname + '_' +  self.bids_suffix + '.tsv', sep='\t')
            # Update json sidecar
            sidecar = get_sidecar(files[i]) 
            sidecar['scrubbed_timepoints'] = {} 
            sidecar['scrubbed_timepoints']['description'] = 'Scrubbing which censors timepoints where the confounds where above a certain time-points.\
                Censored time-points are replaced with replacement value (nans or cubic spline). \
                Output of teneto.TenetoBIDS.set_exclusion_timepoint.'
            sidecar['scrubbed_timepoints']['confound'] = ','.join(confound)
            sidecar['scrubbed_timepoints']['threshold'] = ','.join(exclusion_criteria)
            sidecar['scrubbed_timepoints']['replacement'] = replace_with
            sidecar['scrubbed_timepoints']['badpoint_number'] = badpoints_n
            sidecar['scrubbed_timepoints']['badpoint_ratio'] = badpoints_n / np.array(len(df))
            sidecar['scrubbed_timepoints']['file_exclusion_when_badpoint_ratio'] = tol
            with open(sname + '_' +  self.bids_suffix + '.json', 'w') as fs:
                json.dump(sidecar, fs)
        self.set_bad_files(bad_files, reason='scrubbing (number of points over threshold)')
        self.set_pipeline('teneto_' + teneto.__version__)
        if desc: 
            self.set_bids_tags({'desc': desc.split('-')[1]})


    def get_confound_alternatives(self,quiet=0):
        # This could be mnade better
        file_list = self.get_selected_files(quiet=1, pipeline='confound')

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
            tag = 'desc-' + tag

        if not parc_params: 
            parc_params = {}

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_make_parcellation,f,i,tag,parcellation,parc_name,parc_type,parc_params) for i,f in enumerate(files)}
            for j in as_completed(job):
                j.result()

        if update_pipeline == True:
            if not self.confound_pipeline and len(self.get_selected_files(quiet=1, pipeline='confound')) > 0:
                self.set_confound_pipeline(self.pipeline)
            self.set_pipeline('teneto_' + teneto.__version__)
            self.set_pipeline_subdir('parcellation')
            if tag:
                self.set_bids_tags({'desc': tag.split('-')[1]})
            self.set_bids_suffix('roi')
            if removeconfounds: 
                self.removeconfounds(clean_params=clean_params,transpose=True,njobs=njobs)
            
    def _run_make_parcellation(self,f,i,tag,parcellation,parc_name,parc_type,parc_params):
        fsave, _ = drop_bids_suffix(f)
        save_name, save_dir, _ = self._save_namepaths_bids_derivatives(fsave, tag, 'parcellation', 'roi')
        roi = teneto.utils.make_parcellation(f,parcellation,parc_type,parc_params)
        #Make data node,time
        roi = roi.transpose()
        roi = pd.DataFrame(roi.transpose())
        roi.to_csv(save_dir + save_name + '.tsv', sep='\t')
        sidecar = get_sidecar(f) 
        sidecar['parcellation'] = parc_params
        sidecar['parcellation']['description'] = 'The parcellation reduces the FD nifti files to time-series for some parcellation. Parcellation is made using teneto and nilearn.'
        sidecar['parcellation']['parcellation'] = parcellation
        sidecar['parcellation']['parc_type'] = parc_type
        with open(save_dir + save_name + '.json', 'w') as fs:
            json.dump(sidecar, fs)


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
    #         files = self.get_selected_files(quiet=True, pipeline='functionalconnectivity') 

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


    def removeconfounds(self,confounds=None,clean_params=None,transpose=False,njobs=None,update_pipeline=True, overwrite=True, tag=None): 
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
        overwrite : bool 
        tag : str

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

        if not tag:
            tag = ''
        else:
            tag = 'desc-' + tag

        if confounds: 
            self.set_confounds(confounds)
        files = sorted(self.get_selected_files(quiet=1))
        confound_files = sorted(self.get_selected_files(quiet=1, pipeline='confound'))
        if len(files) != len(confound_files):
            print('WARNING: number of confound files does not equal number of selected files')
        for n in range(len(files)):
            if confound_files[n].split('_confounds')[0].split('func')[1] not in files[n]:
                raise ValueError('Confound matching with data did not work.')

        if not clean_params:
            clean_params = {}

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_removeconfounds,f,confound_files[i],clean_params,transpose, overwrite, tag) for i,f in enumerate(files)}
            for j in as_completed(job):
                j.result()

        self.set_pipeline('teneto_' + teneto.__version__)
        self.set_bids_suffix('roi')
        if tag:
            self.set_bids_tags({'desc': tag.split('-')[1]})

    def _run_removeconfounds(self,file_path,confound_path,clean_params,transpose, overwrite, tag): 
        df = load_tabular_file(confound_path)
        df = df[self.confounds]
        roi = load_tabular_file(file_path).values
        if transpose: 
            roi = roi.transpose()
        elif len(df) == roi.shape[1] and len(df) != roi.shape[0]: 
            print('Input data appears to be node,time. Transposing.')
            transpose = True 
        warningtxt = ''
        if df.isnull().any().any():
            # Not sure what is the best way to deal with this.
            # The time points could be ignored. But if multiple confounds, this means these values will get ignored
            warningtxt = 'Some confounds were NaNs. Setting these values to median of confound.'
            print('WARNING: ' + warningtxt)
            df = df.fillna(df.median())
        roi = nilearn.signal.clean(roi,confounds=df.values,**clean_params)
        if transpose: 
            roi = roi.transpose() 
        roi = pd.DataFrame(roi)
        sname, _ = drop_bids_suffix(file_path)
        suffix = 'roi'
        # Move files to teneto derivatives if the pipeline isn't already set to it
        if self.pipeline != 'teneto_' + teneto.__version__:
            sname = sname.split('/')[-1]
            spath = self.BIDS_dir + '/derivatives/' + 'teneto_' + teneto.__version__ + '/'
            tags = get_bids_tag(sname, ['sub','ses'])
            spath += 'sub-' + tags['sub'] + '/'
            if 'ses' in tags:
                spath += 'ses-' + tags['ses'] + '/'
            spath += 'func/'
            if self.pipeline_subdir:
                spath += self.pipeline_subdir + '/'
            make_directories(spath)
            sname = spath + sname
        if 'desc' in sname and tag: 
            desctag = get_bids_tags(sname.split('/')[-1], 'desc')
            sname = ''.join(sname.split('desc-' + desctag['desc']))
            sname + '_desc-' + tag
        if os.path.exists(sname + self.bids_suffix + '.tsv') and overwrite==False: 
            raise ValueError('overwrite is set to False, but non-unique filename. Set unique desc tag')
        
        roi.to_csv(sname + '_' + suffix +  '.tsv', sep='\t')
        sidecar = get_sidecar(file_path) 
        # need to remove measure_params[i]['communities'] when saving
        if 'confoundremoval' not in sidecar: 
            sidecar['confoundremoval'] = {}
            sidecar['confoundremoval']['description'] = 'Confounds removed from data using teneto and nilearn.' 
        sidecar['confoundremoval']['params'] = clean_params
        sidecar['confoundremoval']['confounds'] = self.confounds
        sidecar['confoundremoval']['confoundsource'] = confound_path
        if warningtxt: 
            sidecar['confoundremoval']['warning'] = warningtxt
        with open(sname + '_' + suffix + '.json', 'w') as fs:
            json.dump(sidecar, fs)




    def networkmeasures(self, measure=None, measure_params={}, tag=None, njobs=None):
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

        tag : str
            Add additional tag to saved filenames.

        Note
        ----
        In measure_params, if communities can equal 'template', 'static', or 'temporal'. 
        These options must be precalculated. If template, Teneto tries to load default for parcellation. If static, loads static communities 
        in BIDS_dir/teneto_<version>/sub-.../func/communities/..._communitytype-static....npy. If temporal, loads static communities 
        in BIDS_dir/teneto_<version>/sub-.../func/communities/..._communitytype-temporal....npy 
 
        Returns
        -------

        Saves in ./BIDS_dir/derivatives/teneto/sub-NAME/func//temporalnetwork/MEASURE/
        Load the measure with tenetoBIDS.load_network_measure
        """
        if not njobs:
            njobs = self.njobs
        self.add_history(inspect.stack()[0][3], locals(), 1)

        # measure can be string or list
        if isinstance(measure, str):
            measure = [measure]
        # measure_params can be dictionaary or list of dictionaries
        if isinstance(measure_params, dict):
            measure_params = [measure_params]
        if measure_params and len(measure) != len(measure_params):
            raise ValueError('Number of identified measure_params (' + str(len(measure_params)) + ') differs from number of identified measures (' + str(len(measure)) + '). Leave black dictionary if default methods are wanted')

        files = self.get_selected_files(quiet=1)

        if not tag:
            tag = ''
        else:
            tag = 'desc-' + tag

        with ProcessPoolExecutor(max_workers=njobs) as executor:
            job = {executor.submit(self._run_networkmeasures,f,tag,measure,measure_params) for f in files}
            for j in as_completed(job):
                j.result()

    def _run_networkmeasures(self,f,tag,measure,measure_params):
        # Load file
        tvc = load_tabular_file(f)
        # Make a tenetoobject
        tvc = teneto.TemporalNetwork(from_df=tvc)
        
        for i, m in enumerate(measure):
            save_name, save_dir, _ = self._save_namepaths_bids_derivatives(f, tag, 'temporalnetwork-' + m, 'tnet')
            # This needs to be updated for tsv data. 
            if 'communities' in measure_params[i]:
                if isinstance(measure_params[i]['communities'],str):
                    tag += '_communitytype-' + measure_params[i]['communities']
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

            netmeasure = tvc.calc_networkmeasure(m,**measure_params[i])
            netmeasure = pd.DataFrame(netmeasure)
            netmeasure.to_csv(save_dir + save_name + '.tsv', sep='\t')
            sidecar = get_sidecar(f) 
            # need to remove measure_params[i]['communities'] when saving
            sidecar['networkmeasure'] = {}
            sidecar['networkmeasure'][m] = measure_params[i]
            sidecar['networkmeasure'][m]['description'] = 'File contained temporal network estimate: ' + m 
            with open(save_dir + save_name + '.json', 'w') as fs:
                json.dump(sidecar, fs)

    def set_bad_subjects(self,bad_subjects,reason=None,oops=False):

        if isinstance(bad_subjects,str):
            bad_subjects = [bad_subjects]
        if reason == 'last':
            reason = 'last' 
        elif reason:
            reason = 'Bad subject (' + reason + ')'
        else: 
            reason = 'Bad subject'
        for bs in bad_subjects:
            if not oops: 
                badfiles = self.get_selected_files(forfile={'sub': bs}, quiet=1)
            else: 
                badfiles = [bf for bf in self.bad_files if 'sub-' + bs in bf]    
            self.set_bad_files(badfiles,reason=reason, oops=oops)
            if bs in self.bids_tags['sub'] and not oops:
                self.bids_tags['sub'].remove(bs)
            elif oops: 
                self.bids_tags['sub'].append(bs)
            else:
                print('WARNING: subject: ' + str(bs) + ' is not found in TenetoBIDS.subjects')

        if not self.bad_subjects:
            self.bad_subjects = bad_subjects
        elif self.bad_subjects and oops:
            self.bad_subjects = [bf for bf in self.bad_subjects if bf not in bad_subjects]
        else:
            self.bad_subjects += bad_subjects


    def set_bad_files(self, bad_files, reason='Manual', oops=False):

        if isinstance(bad_files,str):
            bad_files = [bad_files]

        for f in bad_files:
            sidecar = get_sidecar(f)
            if not oops:
                sidecar['filestatus']['reject'] = True
                sidecar['filestatus']['reason'].append(reason)
            else: 
                if reason == 'last': 
                    sidecar['filestatus']['reason'].remove(sidecar['filestatus']['reason'][-1])
                else: 
                    sidecar['filestatus']['reason'].remove(reason)
                if len(sidecar['filestatus']['reason']) == 0: 
                    sidecar['filestatus']['reject'] = False
            for af in ['.tsv','.nii.gz']: 
                f = f.split(af)[0] 
            f += '.json'
            with open(f, 'w') as fs:
                json.dump(sidecar, fs)

        #bad_files = [drop_bids_suffix(f)[0] for f in bad_files]

        if not self.bad_files and not oops:
            self.bad_files = bad_files
        elif self.bad_files and oops:
            self.bad_files = [bf for bf in self.bad_files if bf not in bad_files]
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

        file_list = self.get_selected_files(quiet=1, pipeline='confound')
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


    def set_bids_suffix(self,bids_suffix):
        """
        The last analysis step is the final tag that is present in files.
        """
        self.add_history(inspect.stack()[0][3], locals(), 1)
        self.bids_suffix = bids_suffix



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

        print('Number of subjects (selected): ' + str(len(self.bids_tags['sub'])))
        print('Subjects (selected): ' + ', '.join(self.bids_tags['sub']))
        if isinstance(self.bad_subjects,list):
            print('Bad subjects: ' + ', '.join(self.bad_subjects))
        else:
            print('Bad subjects: 0')

        print('--- Tasks ---')
        if self.raw_data_exists:
            if self.BIDS.get_tasks():
                print('Number of tasks (in dataset): ' + str(len(self.BIDS.get_tasks())))
                print('Tasks (in dataset): ' + ', '.join(self.BIDS.get_tasks()))
        if 'task' in self.bids_tags:
            print('Number of tasks (selected): ' + str(len(self.bids_tags['task'])))
            print('Tasks (selected): ' + ', '.join(self.bids_tags['task']))
        else:
            print('No task names found')

        print('--- Runs ---')
        if self.raw_data_exists:
            if self.BIDS.get_runs():
                print('Number of runs (in dataset): ' + str(len(self.BIDS.get_runs())))
                print('Runs (in dataset): ' + ', '.join(self.BIDS.get_runs()))
        if 'run' in self.bids_tags:
            print('Number of runs (selected): ' + str(len(self.bids_tags['run'])))
            print('Rubs (selected): ' + ', '.join(self.bids_tags['run']))
        else:
            print('No run names found')


        print('--- Sessions ---')
        if self.raw_data_exists:
            if self.BIDS.get_sessions():
                print('Number of runs (in dataset): ' + str(len(self.BIDS.get_sessions())))
                print('Sessions (in dataset): ' + ', '.join(self.BIDS.get_sessions()))
        if 'ses' in self.bids_tags:
            print('Number of sessions (selected): ' + str(len(self.bids_tags['ses'])))
            print('Sessions (selected): ' + ', '.join(self.bids_tags['ses']))
        else:
            print('No session names found')

        print('--- PREPROCESSED DATA (Pipelines/Derivatives) ---')

        if not self.pipeline:
            print('Derivative pipeline not set. To set, run TN.set_pipeline()')
        else:
            print('Pipeline: ' + self.pipeline)
        if self.pipeline_subdir:
            print('Pipeline subdirectories: ' + self.pipeline_subdir)

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
            tnet = pickle.load(f)
        if reload_object: 
            reloadnet = teneto.TenetoBIDS(tnet.BIDS_dir, pipeline=tnet.pipeline, pipeline_subdir=tnet.pipeline_subdir, bids_tags=tnet.bids_tags,  bids_suffix=tnet.bids_suffix, bad_subjects=tnet.bad_subjects, confound_pipeline=tnet.confound_pipeline, raw_data_exists=tnet.raw_data_exists, njobs=tnet.njobs) 
            reloadnet.histroy = tnet.history
            tnet = reloadnet 
        return tnet

    #timelocked average
    #np.stack(a['timelocked-tvc'].values).mean(axis=0)
    # Remaked based on added meta data derived from events 
    # def load_timelocked_data(self,measure,calc=None,tag=None,avg=None,event=None,groupby=None): 

    #     if not calc:
    #         calc = ''
    #     else:
    #         calc = 'calc-' + calc

    #     if not tag:
    #         tag = ['']
    #     elif isinstance(tag,str): 
    #         tag = [tag]

    #     if avg: 
    #         finaltag = 'timelocked_avg.npy'
    #     else:
    #         finaltag =  'timelocked.npy'
            
    #     self.add_history(inspect.stack()[0][3], locals(), 1)
    #     trialinfo_list = []
    #     data_list = []
    #     std_list = []
    #     for s in self.bids_tags['sub']:

    #         base_path = self.BIDS_dir + '/derivatives/' + self.pipeline
    #         if measure == 'tvc':
    #             base_path += '/sub-' + s + '/func/tvc/timelocked/'
    #         else:
    #             base_path += '/sub-' + s + '/func//temporalnetwork/' + measure + '/timelocked/'

    #         if not os.path.exists(base_path):
    #             print('Warning: cannot find data for subject: ' + s)

    #         for f in os.listdir(base_path):
    #             if os.path.isfile(base_path + f) and f.split('.')[-1] == 'npy':
    #                 if calc in f and all([t + '_' in f or t + '.' in f for t in tag]) and finaltag in f:
    #                     if avg: 
    #                         f = f.split('_avg')[0]
    #                         f_suff = '.npy'
    #                     else: 
    #                         f_suff = ''
    #                     bids_tags=re.findall('[a-zA-Z]*-',f)
    #                     bids_tag_dict = {}
    #                     for t in bids_tags:
    #                         key = t[:-1]
    #                         bids_tag_dict[key]=re.findall(t+'[A-Za-z0-9.,*+]*',f)[0].split('-')[-1]
    #                     trialinfo_eventinfo = pd.read_csv(base_path + '.'.join(f.split('timelocked')[0:-1]) + 'timelocked_trialinfo.csv')
    #                     trialinfo = pd.DataFrame(bids_tag_dict,index=np.arange(0,len(trialinfo_eventinfo)))
    #                     trialinfo = pd.concat([trialinfo,trialinfo_eventinfo],axis=1)
    #                     trialinfo_list.append(trialinfo)
    #                     if avg:
    #                         data_list.append(np.load(base_path + f + '_avg.npy'))
    #                         std_list.append(np.load(base_path + f + '_std.npy'))
    #                     else: 
    #                         data_list.append(np.load(base_path + f))
    #     if avg: 
    #         self.timelocked_data_ = {}
    #         self.timelocked_data_['avg'] = np.stack(np.array(data_list))
    #         self.timelocked_data_['std'] = np.stack(np.array(std_list))
    #     else: 
    #         self.timelocked_data_ = np.stack(np.array(data_list))
        

    #     if trialinfo_list:
    #         out_trialinfo = pd.concat(trialinfo_list)
    #         out_trialinfo = out_trialinfo.drop('Unnamed: 0',axis=1)
    #         out_trialinfo.reset_index(inplace=True,drop=True)
    #         self.timelocked_trialinfo_ = out_trialinfo

    def load_data(self, datatype='tvc', tag=None, measure=''):
        """
        Function loads time-varying connectivity estimates created by the TenetoBIDS.derive function.
        The default grabs all data (in numpy arrays) in the teneto/../func/tvc/ directory.
        Data is placed in teneto.tvc_data_

        Parameters
        ----------

        datatype : str 
            \'tvc\', \'parcellation\', \'participant\', \'temporalnetwork\'

        tag : str or list
            any additional tag that must be in file name. After the tag there must either be a underscore or period (following bids). 

        timelocked : bool 
            Load timelocked data if true.

        measure : str 
            retquired when datatype is temporalnetwork. A networkmeasure that should be loaded. 

        Returns 
        -------

        tvc_data_ : numpy array
            Containing the parcellation data. Each file is appended to the first dimension of the numpy array.
        tvc_trialinfo_ : pandas data frame
            Containing the subject info (all BIDS tags) in the numpy array.
        """    

        if datatype == 'temporalnetwork' and not measure: 
            raise ValueError('When datatype is temporalnetwork, \'measure\' must also be specified.')

        self.add_history(inspect.stack()[0][3], locals(), 1)
        data_list=[]
        trialinfo_list = []

        for s in self.bids_tags['sub']:
            # Define base folder
            base_path, file_list, datainfo = self._get_filelist(datatype, s, tag, measure=measure)
            if base_path:
                for f in file_list:
                    # Include only if all analysis step tags are present
                    # Get all BIDS tags. i.e. in 'sub-AAA', get 'sub' as key and 'AAA' as item.
                    # Ignore if tsv file is empty
                    try:
                        filetags = get_bids_tag(f, 'all')
                        data_list.append(load_tabular_file(base_path + f))
                        # Only return trialinfo if datatype is trlinfo
                        if datainfo == 'trlinfo':
                            trialinfo_list.append(pd.DataFrame(filetags,index=[0]))
                    except pd.errors.EmptyDataError:
                        pass
                # If group data and length of output is one, don't make it a list
                if datatype == 'group' and len(data_list) == 1: 
                    data_list = data_list[0]
                if measure: 
                    data_list = {measure: data_list}
                setattr(self,datatype + '_data_', data_list)
                if trialinfo_list:
                    out_trialinfo = pd.concat(trialinfo_list)
                    out_trialinfo.reset_index(inplace=True, drop=True)
                    setattr(self,datatype + '_trialinfo_', out_trialinfo)


    # REMAKE BELOW BASED ON THE _events from BIDS 
    # def make_timelocked_events(self, measure, event_names, event_onsets, toi, tag=None, avg=None, offset=0):
    #     """
    #     Creates time locked time series of <measure>. Measure must have time in its -1 axis.

    #     Parameters
    #     -----------

    #     measure : str
    #         temporal network measure that should already exist in the teneto/[subject]/tvc/network-measures directory
    #     event_names : str or list
    #         what the event is called (can be list of multiple event names). Can also be TVC to create time-locked tvc. 
    #     event_onsets: list
    #         List of onset times (can be list of list for multiple events)
    #     toi : array
    #         +/- time points around each event. So if toi = [-10,10] it will take 10 time points before and 10 time points after
    #     calc : str
    #         type of network measure calculation.
    #     tag : str or list
    #         any additional tag that must be in file name. After the tag there must either be a underscore or period (following bids). 
    #     offset : int 
    #         If derive uses a method that has a sliding window, then the data time-points are reduced. Offset should equal half of the window-1. So if the window is 7, offset is 3. This corrects for the missing time points. 

    #     Note
    #     ----
    #     Currently no ability to loop over more than one measure

    #     Note
    #     -----
    #     Events that do not completely fit the specified time period (e.g. due to at beginning/end of data) get ignored.  

    #     Returns
    #     -------
    #     Creates a time-locked output placed in BIDS_dir/derivatives/teneto_<version>/..//temporalnetwork/<networkmeasure>/timelocked/
    #     """
    #     self.add_history(inspect.stack()[0][3], locals(), 1)
    #     #Make sure that event_onsets and event_names are lists
    #     #if  np.any(event_onsets[0]):
    #     #    event_onsets = [e.tolist() for e in event_onsets[0]]   
    #     if isinstance(event_onsets[0],int) or isinstance(event_onsets[0],float): 
    #         event_onsets = [event_onsets]
    #     if isinstance(event_names,str): 
    #         event_names = [event_names]
    #     # Combine the different events into one list                    
    #     event_onsets_combined = list(itertools.chain.from_iterable(event_onsets))
    #     event_names_list = [[e]*len(event_onsets[i]) for i,e in enumerate(event_names)]
    #     event_names_list = list(itertools.chain.from_iterable(event_names_list))
    #     #time_index = np.arange(toi[0],toi[1]+1)

    #     if not tag:
    #         tag = ['']
    #     elif isinstance(tag,str): 
    #         tag = [tag]

    #     for s in self.bids_tags['sub']:
    #         if measure == 'tvc':
    #             base_path, file_list, datainfo = self._get_filelist('timelocked-tvc', s, tag)
    #         elif measure == 'parcellation':
    #             base_path, file_list, datainfo = self._get_filelist('timelocked-parcellation', s, tag)
    #         else:
    #             base_path, file_list, datainfo = self._get_filelist('timelocked-temporalnetwork', s, tag, measure=measure)

    #         for f in file_list:
    #             filetags = get_bids_tag(f, 'all')
    #             df = load_tabular_file(base_path + '/' + f)
    #             # make time dimensions the first dimension
    #             self_measure = df.transpose([len(df.shape)-1] + list(np.arange(0,len(df.shape)-1)))
    #             tl_data = []
    #             for e in event_onsets_combined:
    #                 # Ignore events which do not completely fit defined segment
    #                 if e+toi[0]-offset<0 or e+toi[1]-offset>=self_measure.shape[0]: 
    #                     pass
    #                 else: 
    #                     tmp = self_measure[e+toi[0]-offset:e+toi[1]+1-offset]
    #                     # Make time dimension last dimension
    #                     tmp = tmp.transpose(list(np.arange(1,len(self_measure.shape))) + [0])
    #                     tl_data.append(tmp)
    #             tl_data = np.stack(tl_data)
    #             if avg: 
    #                 df=pd.DataFrame(data={'event': '+'.join(list(set(event_names_list))), 'event_onset': [event_onsets_combined]})
    #             else: 
    #                 df=pd.DataFrame(data={'event': event_names_list, 'event_onset': event_onsets_combined})
    
    #                     # Save output
    #                     save_dir_base = base_path + 'timelocked/'
    #                     file_name = f.split('/')[-1].split('.')[0] + '_events-' + '+'.join(event_names) + '_timelocked_trialinfo'
    #                     df.to_csv(save_dir_base + file_name + '.csv')
    #                     file_name = f.split('/')[-1].split('.')[0] + '_events-' + '+'.join(event_names) + '_timelocked'
    #                     if avg:
    #                         tl_data_std = np.std(tl_data,axis=0)
    #                         tl_data = np.mean(tl_data,axis=0) 
    #                         np.save(save_dir_base + file_name + '_std',tl_data_std)
    #                         np.save(save_dir_base + file_name + '_avg',tl_data)
    #                     else: 
    #                         np.save(save_dir_base + file_name,tl_data)
                        

    def _get_filelist(self, method, sub=None, tags=None, measure=None): 

        method_info = {
            'tvc': {
                'pipeline_subdir': 'tvc', 
                'base': 'pipeline', 
                'bids_suffix': 'tvcconn',
                'datatype': 'trlinfo'
            },
            'parcellation': {
                'pipeline_subdir': 'parcellation', 
                'base': 'pipeline', 
                'bids_suffix': 'roi',
                'datatype': 'trlinfo'
            },
            'participant': {
                'pipeline_subdir': '', 
                'base': 'BIDS_dir', 
                'bids_suffix': 'participant',
                'datatype': 'group'
            },
            'fc': {
                'pipeline_subdir': 'fc', 
                'base': 'pipeline', 
                'bids_suffix': 'conn',
                'datatype': 'trlinfo'
            },
            'temporalnetwork': {
                'pipeline_subdir': 'temporalnetwork-' + measure, 
                'base': 'pipeline', 
                'bids_suffix': 'tnet',
                'datatype': 'trlinfo'
            },
            'timelocked-temporalnetwork': {
                'pipeline_subdir': 'temporalnetwork-' + measure, 
                'base': 'pipeline', 
                'bids_suffix': 'avg',
                'datatype': 'trlinfo'
            },
           'timelocked-parcellation': {
                'pipeline_subdir': 'parcellation', 
                'base': 'pipeline', 
                'bids_suffix': 'avg',
                'datatype': 'trlinfo'
            }
        }
        # a = [{},
        # {'derivative': 'fc', 'base': 'pipeline', 'bids_suffix': 'conn'},
        # {'derivative': 'parcellation', 'base': 'pipeline', 'bids_suffix': 'roi'},
        # {'derivative': 'parcellation', 'base': 'pipeline-networkmeasure', 'bids_suffix': networkmeasure},
        # {'derivative': 'timelocked', 'base': 'pipeline-networkmeasure', 'bids_suffix': 'avg'},
        # {'derivative': 'participant', 'base': 'bidsmain', 'bids_suffix': 'participant'}]

        if method not in method_info.keys(): 
            raise ValueError('Unknown type of data to load.')

        if method_info[method]['base'] == 'pipeline':
            base_path = self.BIDS_dir + '/derivatives/' + self.pipeline
            base_path += '/sub-' + sub + '/func/' + method_info[method]['pipeline_subdir'] + '/'
        elif method_info[method]['base'] == 'BIDS_dir':
            base_path = self.BIDS_dir  
        bids_suffix = method_info[method]['bids_suffix']

        if not tags:
            tags = ['']
        elif isinstance(tags,str): 
            tags = [tags]

        if os.path.exists(base_path):
            file_list = os.listdir(base_path)
            file_list = [f for f in file_list if os.path.isfile(base_path + f) and all([t + '_' in f or t + '.' in f for t in tags]) and f.endswith(bids_suffix + '.tsv')]
            return base_path, file_list, method_info[method]['datatype']
        else:
            return None, None, None