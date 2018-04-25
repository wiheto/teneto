# Changelog

## V0.3.2

- Added a temporal participation coefficient to networkmeasures (`teneto.networkmeasures.temporal_part_coef`)

- Allow contact representation in louvain clustering.

- Started adding communities argument to replace subnet argument in networkmeasures.

- Added DeprecationWarning for removal of subnet argument in Teneto 0.3.5

- Added networkmeasures.temporal_part_coef and module_degree_zscore option in teneto.networkmeasures.temporal_degree_centrality.

- Added `file_exclusion_criteria` and `temporal_exclusion_criteria` to TenetoBIDS

- Updated confound_files selection in TenetoBIDS

## V0.3.1

- __main__ added (this may be removed at later date as I don't have an interest in maintaining this).

- Added `njobs` argument to `TenetoBIDS` (and various other functions therein) for parallel computing.

- added nilearn.signal.clean for denoising

- removed statsmodels as dependence.

## V0.3.0

- Dockerfile added

- Added weight-var and weight-mean options to jackknife correlation.

- Added make_static_connectivity in TenetoBIDS

- Added from-subject-fc open to jackknife correlation in TenetoBIDS.derive.

- Numpydoc adopted in docstrings (teneto.utils, teneto.generatenetwork).

- Added `bad_subjects` and `set_bad_subjects()` in `TenetoBIDS`

- Added readthedocs and pypi badge

- Fixed `teneto.networkmeasures.volatility` when subnet is given.

- Changed argument subnet_id in `teneto.networkmeasures.volatility` to subnet.  

- Added possibility to append OH subcortical atlas to make_parcellation.

- Added tag option to TenetoBIDS

- Changed cfg variable name to params in teneto.utils.

- Fixed bug in teneto_degree_centrality where decay and subnet set.

- Allow * and + in tag in TenetoBIDS.  

- clean_community_indexes works with temporal clustering

- Added iGraph as requirement

- Added `teneto.communitydetection.louvain`

## V0.2.7

- Added calc option to `TenetoBIDS.make_time_locked_events` allowing for selecting one of multiple networkmeasures

- Corrected bug where timelocked files were not placed in timelocked directory

- Changed all forms of time_locked and event-locked to timelocked

- Added `load_parcellation_data` to `TenetoBIDS`

- Added history to TenetoBIDS

- Added install_requires to `setup.py`

## V0.2.6 (and partly V0.2.5)

- Removed older examples.

- Added functionality for  derivative only BIDS folders (ie if no raw data is present).

- Added load/save TenetoBIDS object

- Added bids to requirements.txt

- Added dependency to seaborn (adde to requirements.txt)

- Added confound reports to TenetoBIDS.derive

- Added draft version of teneto/bids documentation.

- Added Gordon2014_333 parcellation

- Renamed parcellations to [firstauthor][publicationyear]\_[numberofrois]

- Added more documentation to functions

- Added `networkmeasures.sid`

- Added forced division by 2 for within (calc: time and subnetwork specified)

- Added version number to pipeline name in tenetoBIDS

- Added `TenetoBIDS.make_time_locked_events` and `TenetoBIDS.load_network_measure`

- Added `teneto.utils.get_dimord`

- Fixed the `teneto.__version__` that wasn't getting updated (stuck on 0.2.2 for previous versions. So check setup.py in previous versions to be sure what version you were using.)

- Updated readthedocs (fixed bugs)

- Fixed pybids (instead of bids) in requirements. Removed unnecessary requirements

## V0.2.4b

- Reverted back the incorrect fix for #1, added

- Added distutils version to requirements.  

## V0.2.4

- Fixed `bursty_coeff` when ICT is empty.

- Fix for #1  

## V0.2.3

- Added `confound_pipeline` option to TenetoBIDS.

- Added nan-to-median for nans in confounds in removal in `TenetoBIDS.make_parcellation`

- made `TenetoBIDS.networkmeasures` functional (i.e. saving files, removing an error).

- replaced function `betai` with `betainc` for scipy 1.0 compatibility.

## V0.2.2

- Corrected vminmax error in `plot.graphlet_stack_plot` when specifying a string.

- Changed the default of vminmax in `plot.graphlet_stack_plot` to minmax

- Fixed documentation error in vminma in `plot.graphlet_stack_plot`

- Correct cmap bug in `plot.slice_plot`

- Added `utils.load_parcellation`

- Folder ./data is also included with teneto, at the moment shen2013_tal parcellation is included there. (This is currently under development). And may change name to specify that this has to do with brain research.

- Added decay parameter to degree_centrality

- Added additional names to 'multiple temporal derivative' method

- corrected bug in `bursty_coeff` when specifying `nodes='all'` (previously an error was raised).

- Added `utils.binarize` wiith 'binarize_rdp', 'binarize_percent', and 'binarize_rdp' as options

- Added a inputtype item in netinfo dictionary returned by `utils.process_input`

- Modified `utils.contact2graphlet` to ignore an empty nLabs list.

- Added `utils.create_traj_ranges`

- Added the module `trajectory` with `rdp` compression.

- Added subnet argument to `networkmeasures.bursty_coeff` so that B is calculated per module.  

- Changed default of `params['report']` in `derive.derive` to 'no'

- Added `params['report_path']` to `derive.derive`.

- Corrected default dimension order of `derive.derive` to  'node,time'

- Made report_name parameter to `derive.report.gen_report()`

## V0.2.1

- Code now follows PEP8

- Fixed somwe variable names in `stats.shufflegroups`

- Removed `misc.distancefuncitons`

- All distance funcitons are now through `scipy.spatial.distance`

- Renamed the argument "do" in `networkmeasures` to "calc".

- Renamed the argument "sumOverDim" in `networkmeasures.temporal_degree_centrality` to "axis".

- added function `utils.check_distance_funciton_input`

## V0.2.0

- Added the module `teneto.derive.derive` with `teneto.derive.postpro_pipeline` which inclues: fisher transform, box cox transform and z transform. Can be configured to which ones are used. 5 methods of deriving temporal networks.

- Report generation implemented in derive.

- Code made more readable in places.

- Added `withinsubnetwork` and `betweensubnetwork` options for volatility.

- Optimized `shortest_temporal_path` to dramatically increase speed (was looping the same node multiple times in line 70).

- Added the function `utils.multiple_contacts_get_values`

- Added possibility to get degree centrality per time point in. `teneto.networkmeasures.temporal_degree_centrality`

- Added the ability to specify vmax and vmin in `graphlet_stack_plot`

- renamed `clean_subnetwork_indexes` to `clean_community_indexes` (and now included in initializatin)

- added warning message to `derive.postpro_boxcox` if one edge fails to be normal.

- fixed nLabs/tLabs bug in slice_plot

- Added subnetwork option to `temporal_degree_centrality` when do='time''

## V0.1.4 - Released  Feb 2 2017

- Changed some types in `rand_binomial` documentation

- Added some more customizeable options to `graphlets_stack_plot` (border width, grid color, bordercolor etc.)

- Added possibility to call "subnetwork" option in `volatility`

- Added function, `clean_subnetwork_indexes` which network index assignment to range between 0 and max(NrSubnetwork)-1.

- Updated some of the examples. Also added a ./examples/previous/vx.x.x where previous versions of examples are listed (examples are not created with every new update).

- Added (uncommented) the `taxicab_distance` in `misc.distance_functions` option. Won't be added to the documentation of distance functions until I check there is no reason why I commented it. But preliminary testing says it gives the correct answer.

- Functions `misc.correct_pvalues_for_multiple_testing` and `misc.corrcoef_matrix` are added but not yet implemented in any of the main functions.

## V0.1.3 - Released Jan 26 2017

- Provided clearer documentation in `shortest_temporal_path`

- Add possibility of calculating per time point (or per edge/node - but this takes a tone of time) in `volatility`

- Added possibility of calculating `temporal_efficiency` per node (either "\_from" to "\_to")

- Improved documentation and added references to `rand_binomial`.

## V0.1.2 - Released Jan 6 2017

- Fixed bug in `graphlet_stack_plot` which made white colours have black smudges on them. (Also multiple background colours *should* theoretically be possible now.)

- Added option to remove sharpening filter in `graphlet_stack_plot`

- Added `misc` and `distance_functions` (fixing `volatility`)

- Fixed naming of call to `temporal_shortest_path` in `temporal_efficiency`,`reachability_latency` and `temporal_closeness_centrality`

- Added `process_input` function to cut down on repeating code at beginning of networkmeasures

## v0.1.1 - Released Jan 2 2017

### The changes in v0.1.1 make some functions obsolete in v0.1

- setup.py has been added for installation (e.g. via pip)

- Restructured file structure so that importing teneto has 4 submodules: `plot`, `utils`, `networkmeasures`, `generatenetwork`.

- Functions renamed from camel-case to underscore for python-like code.

- More comments added to plotting functions

- Docs generated and integration with readthedocs.io

- Contact representation field `contacts` is now numpy array instead of tuple.

- `graphlet_stack_plot` plotting function added.

- Examples folder added with several jupyter notebook examples.

- Generatenetwork module added. `rand_binomial` function added which generates a random temporal network.

- `circle_plot.py` created containing `circle_plot` function (previously in `slice_graph.py`).

- variable `vlabs` has been changed to `nlabs`. `dlabs ` has been changed to `tlabs`.

- Field `nlabs` has been added to contact representation.

- `slice_plot` uses information in contact representation when plotting.

- scipy dependency now exists (in graphlet_stack_plot).

- removed unnecessary and unused import of networkx

- restructured the `__init__.py` files for better import of teneto.

## v0.1 - Released Dec 23 2016

- Measures, misc, plot, utils folders added.

- All measures outlined in From static to temporal network theory paper are added.  (temporal efficiency, closeness centrality, bursty coefficient, reachability latency, intercontactitmes, shortest temporal path, fluctuability, volatility)

- Circle_plot and slice_graph added.
