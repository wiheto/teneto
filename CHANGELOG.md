
### v0.1.1 - ( 2016) - Released 2nd Jan 2017

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

### v0.1 - Released 23rd Jan 2016

- Measures, misc, plot, utils folders added.

- All measures outlined in From static to temporal network theory paper are added.  (temporal efficiency, closeness centrality, bursty coefficient, reachability latency, intercontactitmes, shortest temporal path, fluctuability, volatility)

- Circle_plot and slice_graph added.
