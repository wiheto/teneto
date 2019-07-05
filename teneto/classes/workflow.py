import teneto
import numpy as np
import matplotlib.pyplot as plt
import inspect
import pandas as pd
import copy


class TenetoWorkflow():

    def __init__(self, remove_nonterminal_output=True):
        """

        Parameters:
        -----------
        remove_nonterminal_output : bool
            When running, should the nonterminal output be removed when no longer needed (good for RAM).
        """
        self.graph = pd.DataFrame(columns={'i', 'j'})
        self.nodes = {}
        self.classdicts = {}
        self.classdicts['TemporalNetwork'] = dict(inspect.getmembers(
            teneto.TemporalNetwork, predicate=inspect.isfunction))
        self.classdicts['TenetoBIDS'] = dict(inspect.getmembers(
            teneto.TenetoBIDS, predicate=inspect.isfunction))
        self.remove_nonterminal_output = remove_nonterminal_output

    def add_node(self, name, func, depends_on=None, func_params=None):
        """
        Parameters
        ----------
        name : str
        func : str
        depends_on : str
        func_params :

        """
        if depends_on is None:
            if func == 'TenetoBIDS' or func == 'TemporalNetwork':
                depends_on = 'isroot'
            else:
                depends_on = self.graph.iloc[-1]['j']
        if func_params is None:
            func_params = {}
        if name == 'isroot':
            raise ValueError('root cannot be name of node')
        if name in self.nodes:
            raise ValueError(
                name + ' is already part of workflow graph. Each node must have unique name.')
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        if 'isroot' in depends_on:
            if len(depends_on) > 1:
                raise ValueError('Cannot depend on multiple steps and be root')
            elif not (func == 'TenetoBIDS' or func == 'TemporalNetwork'):
                raise ValueError(
                    'root node must be TemporalNetwork or TenetoBIDS')
        if len(depends_on) > 1:
            raise ValueError(
                'At present, only one dependent per step (multiple steps can share the same depndent).')
            # Needs to add weights to depends_on if multiple inputs to indicate what is primary input

        for step in depends_on:
            self.graph = self.graph.append(
                {'i': step, 'j': name}, ignore_index=True).reset_index(drop=True)

        self.nodes[name] = {'func': func, 'params': func_params}

    def remove_node(self, name):
        """
        Remove a node from the graph
        """
        self.nodes.pop(name)
        ind = teneto.utils.get_network_when(self.graph, ij=name).index
        self.graph = self.graph.drop(ind).reset_index(drop=True)
        # Could add checks to see if network is broken

    def calc_runorder(self):
        """
        Calculate the run order of the different nodes on the graph.
        """
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
            [candidate_steps.remove(step) for step in remove_candidate_steps]
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
        """
        Runs the entire graph.
        """
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
                print(dependent_step)
                # In future this will isolate the primary and auxillary dependent steps when  multiple dependencies are allowed.
                dependent_step = dependent_step[0]
                self.output_[step['node']] = copy.copy(
                    self.output_[dependent_step])
                out = getattr(self.output_[step['node']], self.nodes[step['node']]['func'])(
                    **self.nodes[step['node']]['params'])
                if out is not None:
                    self.output_[step['node']] = out
            if (step['level'] > level or len(self.runorder)-1 == i) and self.remove_nonterminal_output:
                nolonger_needed_nodes = self.dependencyuntil[self.dependencyuntil['level'] == level]['node'].tolist(
                )
                for node in nolonger_needed_nodes:
                    self.delete_output(node)
                level = step['level']

    def delete_output(self, nodename):
        """
        Delete the output found after calling twf.run().
        """
        self.output_.pop(nodename)

    def view(self):
        self.calc_runorder()

        print(self.graph)

    def make_workflow_figure(self, fig=None, ax=None):
        self.calc_runorder()
        levelunique = np.unique(self.runorder.level, return_counts=True)[1]
        levelnum = len(levelunique)
        levelmax = levelunique.max()
        self.runorder.level.unique()
        # if ax is None:
        fig, ax = plt.subplots(1, figsize=(levelmax*4, levelnum*2))

        coord = {}
        xmax = 0
        height = 0
        for level in range(levelmax+1):
            width = 0
            for i, node in enumerate(self.runorder[self.runorder['level'] == level].iterrows()):
                props = dict(boxstyle='round', facecolor='gainsboro', alpha=1)
                p = ax.text(
                    width, levelnum-level, node[1]['node'], fontsize=14, verticalalignment='center', bbox=props)
                midpoint_x = width + p.get_bbox_patch().get_extents().width/2
                midpoint_y = levelnum - level
                coord[node[1]['node']] = [midpoint_x, midpoint_y]
                if width > xmax:
                    xmax = width
                width += p.get_bbox_patch().get_extents().width + 1

        ax.set_ylim([0.5, levelnum])
        ax.set_xlim([0, xmax])

        for i, n in self.graph.iterrows():
            if n['i'] == 'isroot':
                pass
            else:
                ax.plot([coord[n['i']][0], coord[n['j']][0]], [
                        coord[n['i']][1], coord[n['j']][1]], zorder=-10000, color='darkgray')

        ax.axis('off')
        return fig, ax
