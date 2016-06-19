# tegrato
Temporal Graph Tools - by William Hedley (wiheto)

## What is here? 

At the moment only one plotting tool to create temporal graphs. But more things are coming soon. This will be more complete but at the moment I wrote this text in about 15 minutes and is incomplete. 

## Slice Plots 

Slice plots (pending better name) are useful at showing connections between small number of nodes and time-points. I have made some of these in the past and usually made

Taken from ex1.py. First off some importing of necessary stuff. 

### Example 1: friends

In this example the aim is to show what information temporal graph theory tries to capture and also show it on a plot. 

Scenario: Ashley were friends with Blake in 2014. Ashley becomes friends with in 2015. Blake and Casey meet (possibly through Ashley) and become friends in 2016. In standard graph theory this would be presented as a connectivity matrix with all three people (nodes) connected. However the temporal infomration for this is lost

```
import numpy as np
import slice_graph as sg
import matplotlib.pyplot as plt
```

Then create a 3D matrix of the shape node x node x time. 

```
#Creat the 3d matrix following the scenario given above
A=np.zeros((3,3,3))
A[0,1,:]=1
A[0,2,1:]=1
A[1,2,2]=1
```

Create the edge list. This is simply convert the easier to raw 3d connectivity matrix (usually how the data will be constructed) and makes it a list of tuples of connections. 

```
edgeList = sg.edgeListFromMatrix(A)
```

Then with the edle list we can use sg.plot_slice to plot. 

```
timeLabs=['2014','2015','2016']
nodeLabs=['Ashley','Blake','Casey']
plt.rcParams['image.cmap'] = 'gist_gray' #Gray colour scale (obviously anything is possible)
fig, ax = plt.subplots(1,1)
ax = sg.plot_slice(nodeLabs,timeLabs,ax,edgeList)
ax.set_xlabel('time (years)')
fig.show()
```
This will generate the figure below: 

![](./figures/ex1.png)

Here we see graphs connecting by [Bézier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve) (admittingly I took some of the functions for this from the plot.ly doccumentation).

### Example 2: different dynamics in the temporal graph

![](./figures/ex2.png)

## Circle plots

While not strictly necesarily part of temporal graph theory, a good circle plotting tool is good to have. From my experience they are usually embedded in graph software and not very accessible/customable. At the moment it can only create the node placement and curves. Plotting the labels is upcoming. 

![](./figures/ex3.png)

## This is just some basic plotting. Whats coming up next? 

Metrics for temporal graph theory. They are cool! 