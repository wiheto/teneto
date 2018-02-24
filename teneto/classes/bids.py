import itertools
import teneto
import os
import re
from bids.grabbids import BIDSLayout
import numpy as np
import inspect
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

#class NetworkMeasures:
#    def __init__(self,**kwargs):
#        pass

    #def temporal_degree_centrality(self,**kwargs):
    #    print(self)
    #    print(teneto.networkmeasures.temporal_degree_centrality(self,**kwargs))

class TenetoBIDS:

    #networkmeasures = NetworkMeasures(self)

    def __init__(self, BIDS_dir, pipeline=None, pipeline_subdir=None, parcellation=None, space=None, subjects='all', sessions='all', runs='all', tasks='all', last_analysis_step=None, analysis_steps=None, confound_pipeline=None, raw_data_exists=True):
        """
        **INPUT**
        :BIDS_dir: string to BIDS directory
        :pipeline: the directory that is in the BIDS_dir/derivatives/<pipeline>/
        :pipeline_subdir: the directory that is in the BIDS_dir/derivatives/<pipeline>/sub-<subjectnr/func/ses-<sesnr>/<pipeline_subdir>
        :parcellation: parcellation name
        :space: different nomralized spaces
        :subjects: can be part of the BIDS file name
        :sessions: can be part of the BIDS file name
        :runs: can be part of the BIDS file name
        :tasks: can be part of the BIDS file name
        :analysis_steps: any tags that exist in the filename (e.g. 'bold' or 'preproc')
        :confound_pipeline: if the confounds file is in another directory than the pipeline directory.
        """

        self.contact = []
        if raw_data_exists:
            self.BIDS = BIDSLayout(BIDS_dir)
        else:
            self.BIDS = 'Raw data was flagged as not present in directory structure.'
        self.BIDS_dir = BIDS_dir
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
        if sessions == 'all' and self.raw_data_exists:
            self.sessions = self.BIDS.get_sessions()
        elif self.raw_data_exists:
            self.set_sessions(sessions)
        else:
            self.sessions = []
        if tasks == 'all' and self.raw_data_exists:
            self.tasks = self.BIDS.get_tasks()
        elif self.raw_data_exists:
            self.set_tasks(tasks)
        else:
            self.tasks = []
        if runs == 'all' and self.raw_data_exists:
            self.runs = self.BIDS.get_runs()
        elif self.raw_data_exists:
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
            self.analysis_steps = ''

    def derive(self, params, update_pipeline=True):

        """
        :params: is a dictionary. See teneto.derive.derive for the structure of this.

        :update_pipeline: if true, the object updates with the new directories made during derivation.
        """

        files = self.get_selected_files(quiet=1)
        confound_files = self.get_confound_files(quiet=1)
        if confound_files:
            confounds_exist = True

        for i, f in enumerate(files):

            # ADD MORE HERE (csv, json, nifti)
            if f.split('.')[-1] == 'npy':
                data = np.load(f)
            else:
                raise ValueError('derive can only load npy files at the moment')

            file_name = f.split('/')[-1].split('.')[0]
            save_name = file_name + '_tvcmethod-' + params['method'] + '_tvc'
            paths_post_pipeline = f.split(self.pipeline)

            if self.pipeline_subdir:
                paths_post_pipeline = paths_post_pipeline[1].split(self.pipeline_subdir)[0]
            else:
                paths_post_pipeline = paths_post_pipeline[1].split(file_name)[0]
            save_dir = self.BIDS_dir + '/derivatives/' + 'teneto/' + paths_post_pipeline + '/tvc/'

            params['report'] = 'yes'
            params['report_path'] =  self.BIDS_dir + '/derivatives/' + 'teneto/' + paths_post_pipeline + '/tvc/report/'
            params['report_filename'] =  save_name + '_derivationreport.html'

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

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
                #NOW CORRELATE DF WITH DFC BUT DFC INDEX NOT DF.
                dfc_df_z = (dfc_df - dfc_df.mean())
                df_z = (df - df.mean())
                R_df = dfc_df_z.T.dot(df_z).div(len(dfc_df)).div(df_z.std(ddof=0)).div(dfc_df_z.std(ddof=0), axis=0)
                R_df_describe = R_df.describe()
                desc_index = R_df_describe.index
                confound_report_dir = params['report_path'] + '/' + analysis_step + '_vs_confounds/'
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
                    report += '<img src=' + confound_report_figdir + c + '.png><br><br>'
                report += '</body></html>'

            with open(confound_report_dir + analysis_step + '_vs_confounds.html', 'w') as file:
                file.write(report)
                
            file.close()

        if update_pipeline == True:
            if not self.confound_pipeline and len(self.get_confound_files(quiet=1)) > 0:
                self.set_confound_pipeline = self.pipeline
            self.set_pipeline('teneto')
            self.set_pipeline_subdir('tvc')
            self.set_last_analysis_step('tvc')


    def networkmeasures(self, measure=None, measure_params={}):
        """
        Runs a network measure

        For available funcitons see: teneto.networkmeasures

        *INPUT*

        :measure: (string or list) the function(s) from teneto.networkmeasures.
        :measure_params: (dictionary or list of dictionaries) containing kwargs for the argument in measure.

        *RETURNS*
        Saves in ./BIDS_dir/derivatives/teneto/sub-NAME/func/tvc/temporal-network-measures/MEASURE/
        """

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

        for f in files:

            # ADD MORE HERE (csv, json, nifti)
            if f.split('.')[-1] == 'npy':
                data = np.load(f)
            else:
                raise ValueError('derive can only load npy files at the moment')

            save_dir_base = '/'.join(f.split('/')[:-1]) + '/temporal-network-measures/'

            file_name = f.split('/')[-1].split('.')[0]

            for i, m in enumerate(measure):

                sname = m.replace('_','-')
                if not os.path.exists(save_dir_base + sname):
                    os.makedirs(save_dir_base + sname)

                save_name = file_name + '_' + sname
                netmeasure = module_dict[m](data,**measure_params[i])

                np.savetxt(save_dir_base + sname + '/' + save_name + '.csv', netmeasure, delimiter=",", header=m)


    def get_space_alternatives(self,quiet=0):
        if not self.pipeline:
            print('Please set pipeline first.')
            self.get_pipeline_alternatives()
        else:
            space_alternatives = []
            if self.sessions:
                ses = '/ses-' + self.sessions + '/'
            else:
                ses = ''
            for s in self.BIDS.get_subjects():
                derdir_files = os.listdir(self.BIDS_dir + '/derivatives/' + self.pipeline + '/' + self.pipeline_subdir +'/sub-' + s + '/' + ses + 'func/')
                space_alternatives += [re.split('[_.]',f.split('_space-')[1])[0] for f in derdir_files if re.search('_space-',f)]
            space_alternatives = set(space_alternatives)
            if quiet == 0:
                print('Space alternatives: ' + ', '.join(space_alternatives))
            return list(space_alternatives)

    def get_pipeline_alternatives(self,quiet=0):
        if not os.path.exists(self.BIDS_dir + '/derivatives/'):
            print('Derivative directory not found. Is the data preprocessed?')
        else:
            pipeline_alternatives = os.listdir(self.BIDS_dir + '/derivatives/')
            if quiet == 0:
                print('Derivative alternatives: ' + ', '.join(pipeline_alternatives))
            return list(pipeline_alternatives)

    def get_pipeline_subdir_alternatives(self,quiet=0):
        if not self.pipeline:
            print('Please set pipeline first.')
            self.get_pipeline_alternatives()
        else:
            pipeline_subdir_alternatives = []
            # check code below, why is s not used? 
            for s in self.BIDS.get_subjects():
                derdir_files = os.listdir(self.BIDS_dir + '/derivatives/' + self.pipeline + '/')
                pipeline_subdir_alternatives += [f for f in derdir_files if os.path.isdir(f)]
            pipeline_subdir_alternatives = set(pipeline_subdir_alternatives)
            if quiet == 0:
                print('Pipeline_subdir alternatives: ' + ', '.join(pipeline_subdir_alternatives))
            return list(pipeline_subdir_alternatives)

    def get_selected_files(self,quiet=0):
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

            r = re.compile('^' + fstr + '.*' + space + '.*' + self.last_analysis_step + '.*')
            if os.path.exists(wdir):
                # make filenames
                found = list(filter(r.match, os.listdir(wdir)))
                # Include only if all analysis step tags are present
                found = [i for i in found if all(x in i for x in self.analysis_steps)]
                # Exclude if confounds tag is present
                found = [i for i in found if '_confounds' not in i]
                # Make full paths
                found = list(map(str.__add__,[re.sub('/+','/',wdir)]*len(found),found))

                if found:
                    found_files += found

        if quiet == 0:
            print(found_files)
        return found_files


    def get_confound_files(self,quiet=0):
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

    def set_confound_pipeline(self,confound_pipeline):
        """
        There may be times when the pipeline is updated (e.g. teneto) but you want the confounds from the preprocessing pipieline (e.g. fmriprep).
        To do this, you set the confound_pipeline to be the preprocessing pipeline where the confound files are.
        """
        if not os.path.exists(self.BIDS_dir + '/derivatives/' + confound_pipeline):
            print('Specified direvative directory not found.')
            self.get_pipeline_alternatives()
        else:
            # Todo: perform check that pipeline is valid
            self.confound_pipeline = confound_pipeline



    def set_confounds(self,confounds,quiet=0):
        # This could be mnade better
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


    def make_parcellation(self,parcellation,parc_type=None,parc_params=None,update_pipeline=True,removeconfounds=False):

        parc_name = parcellation.split('_')[0].lower()

        # Check confounds have been specified
        if not self.confounds and removeconfounds:
            raise ValueError('Specified confounds are not found. Make sure that you have run self.set_confunds([\'Confound1\',\'Confound2\']) first.')

        # In theory these should be the same. So at the moment, it goes through each element and checks they are matched.
        # A matching algorithem may be needed if cases arise where this isnt the case
        files = self.get_selected_files(quiet=1)
        if removeconfounds:
            confound_files = self.get_confound_files(quiet=1)
            if len(files) != len(confound_files):
                print('WARNING: number of confound files does not equal number of selected files')
            for n in range(len(files)):
                if confound_files[n].split('_confounds')[0] not in files[n]:
                    raise ValueError('Confound matching with data did not work.')

        for i,f in enumerate(files):

            file_name = f.split('/')[-1].split('.')[0]
            save_name = file_name + '_parc-' + parc_name + '_roi'
            paths_post_pipeline = f.split(self.pipeline)
            if self.pipeline_subdir:
                paths_post_pipeline = paths_post_pipeline[1].split(self.pipeline_subdir)
            paths_post_pipeline = paths_post_pipeline[1].split(file_name)[0]
            save_dir = self.BIDS_dir + '/derivatives/teneto/' + paths_post_pipeline + '/parcellation/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            roi = teneto.utils.make_parcellation(f,parcellation,parc_type,parc_params)
            # Make nodd, time
            roi = roi.transpose()

            # Confounds need to be loaded here.
            if removeconfounds:
                if confound_files[i].split('.')[-1] == 'csv':
                    delimiter = ','
                elif confound_files[i].split('.')[-1] == 'tsv':
                    delimiter = '\t'
                df = pd.read_csv(confound_files[i],sep=delimiter)
                df = df[self.confounds]
                if df.isnull().any().any():
                    # Not sure what is the best way to deal with this.
                    # The time points could be ignored. But if multiple confounds, this means these values will get ignored
                    print('WARNING: Some confounds were NaNs. Setting these values to median of confound.')
                    df = df.fillna(df.median())
                patsy_str_confounds = ' + '.join(self.confounds)
                # Linear regresion to regress out (i.e. perform regression and keep residuals) or confound variables.
                for r in range(roi.shape[0]):
                    # Create dataframe
                    df['y'] = roi[r,:]
                    # Specify model
                    model = smf.ols(formula = 'y ~ ' + patsy_str_confounds,data=df)
                    # Fit model
                    res = model.fit()
                    # Get residuals
                    roi[r,:] = res.resid_pearson


            np.save(save_dir + save_name + '.npy', roi)

        if update_pipeline == True:
            if not self.confound_pipeline and len(self.get_confound_files(quiet=1)) > 0:
                self.set_confound_pipeline(self.pipeline)
            self.set_pipeline('teneto')
            self.set_pipeline_subdir('parcellation')
            self.analysis_steps += self.last_analysis_step
            self.set_last_analysis_step('roi')
            self.parcellation = parcellation



    def set_last_analysis_step(self,last_analysis_step):
        self.last_analysis_step = last_analysis_step

    def set_analysis_steps(self,analysis_step,add=False):
        if isinstance(analysis_step,str):
            if add:
                self.analysis_steps.append()
            else:
                self.analysis_steps = [analysis_step]
        elif isinstance(analysis_step,list):
            if add:
                self.analysis_steps += analysis_step
            else:
                self.analysis_steps = analysis_step

        else:
            raise ValueError('Invalud input')



    def set_pipeline(self,pipeline):
        if not os.path.exists(self.BIDS_dir + '/derivatives/' + pipeline):
            print('Specified direvative directory not found.')
            self.get_pipeline_alternatives()
        else:
            # Todo: perform check that pipeline is valid
            self.pipeline = pipeline

    def set_pipeline_subdir(self,pipeline_subdir):
#        if not os.path.exists(self.BIDS_dir + '/derivatives/' + self.pipeline + '/' + pipeline_subdir):
#            print('Specified direvative sub-directory not found.')
#            self.get_pipeline_subdir_alternatives()
#        else:
#            # Todo: perform check that pipeline is valid
        self.pipeline_subdir = pipeline_subdir


    def set_runs(self,runs):
        if isinstance(runs,str):
            runs=[runs]
        if self.raw_data_exists:
            runs_in_dataset = self.BIDS.get_runs()
            if len(set(runs).intersection(runs_in_dataset))==len(runs):
                self.runs = sorted(list(runs))
            else:
                raise ValueError('Specified run(s) not founds in BIDS dataset')
        else:
            self.runs = sorted(list(tasks))

    def set_sessions(self,sessions):
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
        space_alternatives = self.get_space_alternatives(quiet=1)
        if space not in space_alternatives:
            raise ValueError('Specified space cannot be found for any subjects. Run TN.get_space_alternatives() to see the optinos in directories.')
        self.space = space

    def set_subjects(self,subjects=None):
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
