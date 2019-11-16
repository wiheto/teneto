"""TenetoWorkflows are a way of predefining and saving an analysis pipeline using TemporalNetworks or TenetoBIDS."""

import teneto
import numpy as np
import matplotlib.pyplot as plt
import inspect
import pandas as pd
import copy


class TenetoWorkflow():

    def __init__(self, remove_nonterminal_output=True):
        """
        Initialize TenetoWorkflow.

        Parameters:
        -----------
        remove_nonterminal_output : bool
            When running, should the nonterminal output be removed when no longer
            needed (good for RAM).
        """
        self.graph = pd.DataFrame(columns={'i', 'j'})
        self.nodes = {}
        self.classdicts = {}
        self.classdicts['TemporalNetwork'] = dict(inspect.getmembers(
            teneto.TemporalNetwork, predicate=inspect.isfunction))
        self.classdicts['TenetoBIDS'] = dict(inspect.getmembers(
            teneto.TenetoBIDS, predicate=inspect.isfunction))
        self.remove_nonterminal_output = remove_nonterminal_output

    def add_node(self, nodename, func, depends_on=None, params=None):
        """
        Adds a node to the workflow graph.

        Parameters
        ----------
        nodename : str
            Name of the node
        func : str
            The function that is to be called.
            The alternatives here are 'TemporalNetwork' or 'TenetoBIDS',
            or any of the functions that can be called within these classes.
        depends_on : str
            which step the node depends on. If empty, is considered to preceed
            the previous step. If 'isroot' is specified, it is considered an input variable.
        params : dict
            Parameters that are passed into func.

        Note
        ----
        These functions are not run until TenetoWorkflow.run() is called.
        """
        if depends_on is None:
            if func == 'TenetoBIDS' or func == 'TemporalNetwork':
                depends_on = 'isroot'
            else:
                depends_on = 'lastnode'
        if params is None:
            params = {}
        if nodename == 'isroot':
            raise ValueError('isroot cannot be nodename')
        if nodename in self.nodes:
            raise ValueError(
                nodename + ' is already part of workflow graph. \
                Each node must have unique nodename.')
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        if 'isroot' in depends_on:
            if len(depends_on) > 1:
                raise ValueError('Cannot depend on multiple steps and be root')
            elif not (func == 'TenetoBIDS' or func == 'TemporalNetwork'):
                raise ValueError(
                    'root node must be TemporalNetwork or TenetoBIDS')
        if depends_on[0] == 'lastnode':
            depends_on[0] = self.graph.iloc[-1]['j']
        if len(depends_on) > 1:
            raise ValueError(
                'At present, only one dependent per step (multiple steps can \
                share the same depndent).')
            # Needs to add weights to depends_on if multiple inputs to indicate what is primary input
        for step in depends_on:
            self.graph = self.graph.append(
                {'i': step, 'j': nodename}, ignore_index=True).reset_index(drop=True)

        # make sure that the i,j ordering is kept
        self.graph = self.graph.reindex(sorted(self.graph.columns), axis=1)
        self.nodes[nodename] = {'func': func, 'params': params}

    def remove_node(self, nodename):
        """
        Remove a node from the graph.

        Parameters
        ---------
        nodename : str
            Name of node that is to be removed.
        """
        self.nodes.pop(nodename)
        ind = teneto.utils.get_network_when(self.graph, ij=nodename).index
        self.graph = self.graph.drop(ind).reset_index(drop=True)
        # Could add checks to see if network is broken

    def calc_runorder(self):
        """Calculate the run order of the different nodes on the graph."""
        not_run = self.graph['i'].tolist() + self.graph['j'].tolist()
        not_run = list(set(not_run))
        not_run.remove('isroot')
        run = ['isroot']
        levels = 0
        run_level = []
        needed_at = {}
        while len(not_run) > 0:
            candidate_steps = teneto.utils.get_network_when(
                self.graph, i=run, j=not_run, logic='and')['j'].tolist()
            remove_candidate_steps = teneto.utils.get_network_when(
                self.graph, i=not_run, j=candidate_steps, logic='and')['j'].tolist()
            remove_candidate_steps = list(set(remove_candidate_steps))
            _ = [candidate_steps.remove(step)
                 for step in remove_candidate_steps]
            for step in candidate_steps:
                run.append(step)
                not_run.remove(step)
                run_level.append(levels)
                dependencies = teneto.utils.get_network_when(self.graph, j=step)[
                    'i'].tolist()
                for d in dependencies:
                    needed_at[d] = levels
            levels += 1
        run.remove('isroot')
        needed_at.pop('isroot')
        self.dependencyuntil = pd.DataFrame(
            data={'node': list(needed_at.keys()), 'level': list(needed_at.values())})
        self.runorder = pd.DataFrame(data={'node': run, 'level': run_level})

    def run(self):
        """Runs the entire graph."""
        self.output_ = {}
        self.calc_runorder()
        # Can add multiprocess here over levels
        root_funcs = {'TemporalNetwork': teneto.TemporalNetwork,
                      'TenetoBIDS': teneto.TenetoBIDS}
        level = 0
        for i, step in self.runorder.iterrows():
            if i == 0:
                self.output_[step['node']] = root_funcs[self.nodes[step['node']]['func']](
                    **self.nodes[step['node']]['params'])
                self.pipeline = self.nodes[step['node']]['func']

            else:
                dependent_step = teneto.utils.get_network_when(
                    self.graph, j=step['node'], logic='and')['i'].tolist()
                # In future this will isolate the primary and auxillary dependent steps when  multiple dependencies are allowed.
                dependent_step = dependent_step[0]
                self.output_[step['node']] = copy.copy(
                    self.output_[dependent_step])
                out = getattr(self.output_[step['node']], self.nodes[step['node']]['func'])(
                    **self.nodes[step['node']]['params'])
                if out is not None:
                    self.output_[step['node']] = out
            if step['level'] > level and self.remove_nonterminal_output:
                self.delete_output_from_level(level)
                level = step['level']
        if self.remove_nonterminal_output:
            self.delete_output_from_level(level)

    def delete_output_from_level(self, level):
        """Delete the output found after calling TenetoWorkflow.run()."""
        output_todelete = self.dependencyuntil[self.dependencyuntil['level'] == level]['node'].tolist(
        )
        for node in output_todelete:
            self.output_.pop(node)

    def make_workflow_figure(self, fig=None, ax=None):
        """
        Creates a figure depicting the workflow figure.

        Parameters
        ----------
        fig : matplotlib
        ax : matplotlib

        if fig is used as input, ax should be too.

        Returns
        -------
        fig, ax : matplotlib
            matplotlib figure and axis
        """
        self.calc_runorder()
        levelunique = np.unique(self.runorder.level, return_counts=True)[1]
        levelnum = len(levelunique)
        levelmax = levelunique.max()
        self.runorder.level.unique()
        # if ax is None:
        fig, ax = plt.subplots(1, figsize=(levelmax*4, levelnum*2))

        coord = {}
        xmax = 0
        for level in range(levelnum):
            width = 0
            for _, node in enumerate(self.runorder[self.runorder['level'] == level].iterrows()):
                props = dict(boxstyle='round', facecolor='gainsboro', alpha=1)
                p = ax.text(
                    width, levelnum-level, node[1]['node'], fontsize=14, verticalalignment='center', bbox=props)
                midpoint_x = width
                midpoint_y = levelnum - level
                coord[node[1]['node']] = [midpoint_x, midpoint_y]
                width += p.get_bbox_patch().get_extents().width + 1
                if width > xmax:
                    xmax = width

        for _, n in self.graph.iterrows():
            if n['i'] == 'isroot':
                pass
            else:
                ax.plot([coord[n['i']][0], coord[n['j']][0]], [
                        coord[n['i']][1], coord[n['j']][1]], zorder=-10000, color='darkgray')
        ax.axis('off')
        ax.set_ylim([0.5, levelnum])
        ax.set_xlim([0, xmax])
        return fig, ax
