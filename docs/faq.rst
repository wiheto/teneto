FAQ
==================================

What is the dimension order for dense arrays in *Teneto*?
--------------------------------------------------------------

Inputs/outputs in Teneto can be in both Numpy arrays (time series or temporal works) or Pandas Dataframes (time series).
The default dimension order runs from node to time. This means that if you have a temporal network array in Teneto, than the array should have the dimension order *(node,node,time)*.
If using time series than the dimension order *(node,time)*. This entails that the nodes are the *rows* in a pandas array and the time-points are the *columns*.
Different software can organize their dimension orders differently (e.g. Nilearn uses a time,node dimension order).
