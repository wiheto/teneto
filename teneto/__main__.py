import sys
import argparse 
import teneto 
import numpy as np 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="What should be run. Options: derive, networkmeasures", type=str)
    parser.add_argument("-i", help="Path to input file", type=str)
    parser.add_argument("-o", help="Path to output file", type=str)
    parser.add_argument("--method", help="TVC method to run (if r=derive). Options: jackknife, spatialdistance, slidingwindow, taperedslidingwindow, mtd", type=str)
    parser.add_argument("--report", help="Generate pool. yes or no. Default: yes", type=str)
    parser.add_argument("--report_path", help="Where the report is saved. Default is ./report/[analysis_id]/derivation_report.html", type=str)
    parser.add_argument("--analysis_id", help="Add to identify specfic analysis. Generated report will be placed in './report/' + analysis_id + '/derivation_report.html", type=str)
    parser.add_argument("--dimord", help="Dimension order of input date. Either: time,node or node,time (default)", type=str)
    parser.add_argument("--distance", help="Distance metric. If -r=derive and --method=spatialdistance", type=str)
    parser.add_argument("--windowsize", help="Size of window. Only if -r=derive and --method=[slidingwindow,taperedslidingwindow,mtd]", type=int)
    parser.add_argument("--distribution", help="Scipy distribution (e.g. 'norm','expon'). Any distribution here: https://docs.scipy.org/doc/scipy/reference/stats.html. Only if -r=derive and --method=taperedslidingwindow", type=str)
    parser.add_argument("--distribution_params", help="Distribution parameters for specified distribution. Can be called multiple times for multiple parameters (must be in order). Only if -r=derive and --method=taperedslidingwindow", type=str, action='append')
    parser.add_argument("--weight_mean", help="Path to connectivity matrix to weight JC mean. Only if r=derive and --method jackknife", type=str)
    parser.add_argument("--weight_var", help="Path to connectivity matrix to weight JC variance. Only if r=derive and --method jackknife", type=str)
    parser.add_argument("--postpro_standardize", help="Postprocessing if r=derive, standerdize", action='store_true')
    parser.add_argument("--postpro_fisher", help="Postprocessing if r=derive, fisher transform", action='store_true')
    parser.add_argument("--postpro_boxcox", help="Postprocessing if r=derive. box cox transform.", action='store_true')
    args = parser.parse_args()
 
 
    if args.r == 'derive': 
        print(args )
        G = np.load(args.i)
        params = {} 
        params['method'] = args.method
        params['report'] = args.report
        params['report_path'] = args.report_path
        params['analysis_id'] = args.analysis_id
        if params['method'] == 'slidingwindow':
            params['windowsize'] = args.windowsize
        elif params['method'] == 'taperedslidingwindow':
            params['windowsize'] = args.window
            params['distribution'] = args.distribution
            params['distribution_params'] = args.distributions_params
        elif params['method'] == 'jackknife':
            if args.weight_mean: 
                r = np.load(args.weight_mean)
                params['weight_mean'] = r
            if args.weight_var: 
                r = np.load(args.weight_var)
                params['weight_var'] = r
        elif params['method'] == 'slidingwindow':
            params['distance'] = args.distance
        elif params['method'] == 'mtd':
            params['windowsize'] = args.windowsize 
        postpro = ''
        if args.postpro_fisher: 
            postpro += 'fisher'
        if args.postpro_boxcox: 
            if len(postpro) > 0: 
                postpro += '+'
            postpro += 'boxcox'
        if args.postpro_standardize: 
            if len(postpro) > 0: 
                postpro += '+'
            postpro += 'standardize'                                                 
        params['analysis_id'] = args.method
        params['postpro'] = postpro
        dfc = teneto.derive.derive(G,params)
        np.save(args.o,dfc)
    
    if args.r == 'derive-static': 
        data = np.load(args.i)
        if args.dimord: 
            if args.dimord == 'time,node': 
                data = data.transpose()
        R = teneto.misc.corrcoef_matrix(data)[0]
        np.save(args.o,R)

if __name__ == '__main__':
    main()
