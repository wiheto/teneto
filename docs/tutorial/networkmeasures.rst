Tutorial: Temporal network measures
######################################

The module :ref:`teneto.networkmeasures` includes several functions to quantify different properties of temporal networks. Below are four different types of properties which can be calculated for each node. For all these properties you can generally derive a time-averaged version or one value per time-point.

Many of the functions use a calc argument to specify what type of measure you want to quantify. For example *calc='global'* will return the global version of a measure and *calc='communities'* will return the community version of the function.

Centrality measures
*****************************

Centrality measures quantify a value per node. These can be useful for finding important nodes in the network.

-  :py:func:`.temporal_degree_centrality`
-  :py:func:`.temporal_betweenness_centrality`
-  :py:func:`.temporal_closeness_centrality`
-  :py:func:`.topological_overlap`
-  :py:func:`.bursty_coeff`

Community dependent measures
*****************************

Commuinty measure quantify a value per comunity or a value for community interactions. Communities are an important part of network theory, where nodes are grouped into groups.

-  :py:func:`.sid`, when *calc='community_avg'* or *calc='community_pairs'*
-  :py:func:`.bursty_coeff`, when *calc='communities'*
-  :py:func:`.volatility`, when *calc='communities'*

Node measures that are dependent on community vector
=====================================================

-  :py:func:`.temporal_participation_coeff`
-  :py:func:`.temporal_degree_centrality`, when *calc='module_degree_zscore'*


Global measures
*****************************

Global measures try and calculate one value to reflect the entire network.
Examples of global measures:

-  :py:func:`.temporal_efficiency`
-  :py:func:`.reachability_latency`
-  :py:func:`.fluctuability`
-  :py:func:`.volatility`, when *calc='global'*
-  :py:func:`.topological_overlap`, when *calc='global'*
-  :py:func:`.sid`, when *calc='global'*

Edge measures
****************************

Edge measures quantify a property for each edge.

-  :py:func:`.shortest_temporal_paths`
-  :py:func:`.intercontacttimes`
-  :py:func:`.local_variation`
