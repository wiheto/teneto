### Upcoming V0.1.5 - develop branch

- Added the function `utils.multiple_contacts_get_values`

- Added the module `teneto.derive` with `teneto.derive.postpro_pipeline`
which inclues: fisher transform, box cox transform and z transform. Can be configured to which ones are used. `teneto.derive.derive_with_weighted_pearson` as main functions which includes sliding window, tapered sliding window, spatial distance.

- Added report generation for derivation. `teneto.derive.report`

- Added possibility to get degree centrality per time point in. `teneto.networkmeasures.temporal_degree_centrality`

- Tested derive functions against matlab code and appears to give same results except for rounding differences in Euclidean distance and Lambda estimation of box cox.

- Added the ability to specify vmax and vmin in `graphlet_stack_plot`

- renamed `clean_subnetwork_indexes` to `clean_community_indexes` (and now included in initializatin)

- added warning message to `derive.postpro_boxcox` if one edge fails to be normal. 

### V0.1.4 - Released  Feb 2 2017

- Changed some types in `rand_binomial ` documentation

- Added some more customizeable options to `graphlets_stack_plot` (border width, grid color, bordercolor etc.)

-  Added possibility to call "subnetwork" option in `volatility`

- Added function, `clean_subnetwork_indexes` which network index assignment to range between 0 and max(NrSubnetwork)-1.

- Updated some of the examples. Also added a ./examples/previous/vx.x.x where previous versions of examples are listed (examples are not created with every new update).  

- Added (uncommented) the `taxicab_distance` in `misc.distance_functions` option. Won't be added to the documentation of distance functions until I check there is no reason why I commented it. But preliminary testing says it gives the correct answer.  

- Functions `misc.correct_pvalues_for_multiple_testing` and `misc.corrcoef_matrix` are added but not yet implemented in any of the main functions.   

### V0.1.3 - Released Jan 26 2017

- Provided clearer documentation in `shortest_temporal_path`

- Add possibility of calculating per time point (or per edge/node - but this takes a tone of time) in `volatility`

- Added possibility of calculating `temporal_efficiency` per node (either "\_from" to "\_to")

- Improved documentation and added references to `rand_binomial`.


### V0.1.2 - Released Jan 6 2017

- Fixed bug in `graphlet_stack_plot` which made white colours have black smudges on them. (Also multiple background colours *should* theoretically be possible now.)

- Added option to remove sharpening filter in `graphlet_stack_plot`

- Added `misc` and `distance_functions` (fixing `volatility`)

- Fixed naming of call to `temporal_shortest_path` in `temporal_efficiency`,`reachability_latency` and `temporal_closeness_centrality`

- Added `process_input` function to cut down on repeating code at beginning of networkmeasures



### v0.1.1 - Released Jan 2 2017

*The changes in v0.1.1 make some functions obsolete in v0.1*

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

- variable `vlabs` has been changed to `nlabs `. `dlabs ` has been changed to `tlabs`.

- Field `nlabs` has been added to contact representation.

- `slice_plot` uses information in contact representation when plotting.   

- scipy dependency now exists (in graphlet_stack_plot).

- removed unnecessary and unused import of networkx

- restructured the `__init__.py` files for better import of teneto.  

### v0.1 - Released Dec 23 2016

- Measures, misc, plot, utils folders added.

- All measures outlined in From static to temporal network theory paper are added.  (temporal efficiency, closeness centrality, bursty coefficient, reachability latency, intercontactitmes, shortest temporal path, fluctuability, volatility)

- Circle_plot and slice_graph added.
