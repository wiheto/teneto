import matplotlib.pyplot as plt
import os
import numpy as np
import csv

def gen_report(report,sdir='./'):

    #Create report directory
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    # Add a slash to file directory if not included to avoid DirNameFleName instead of DirName/FileName being creaated
    if sdir[-1] is not '/':
        sdir += '/'

    report_html = '<html><body>'

    if 'method' in report.keys():

        report_html += "<h1>Method: " + report['method'] + "</h1><p>"

        for i in report[report['method']]:

            if i == 'taper_window':

                fig,ax = plt.subplots(1)
                ax.plot(report[report['method']]['taper_window'],report[report['method']]['taper'])
                ax.set_xlabel('Window (time). 0 in middle of window.')
                ax.set_title('Taper from ' + report[report['method']]['distribution'] + ' distribution (PDF).')
                fig.savefig(sdir + 'taper.png')

                report_html += "<img src='./taper.png' width=500>" + "<p>"

            else:

                report_html += "- <b>" + i + "</b>: " + str(report[report['method']][i]) + "<br>"

    if 'postprocess' in report.keys():

        report_html += "<p><h2>Postprocessing:</h2><p>"

        report_html += "<b>Pipeline: </b>"

        for i in report['postprocess']:

            report_html += " " + i + ","


        for i in report['postprocess']:

            report_html += "<p><h3>" + i + "</h3><p>"

            for ii in report[i]:

                if ii == 'lambda':

                    report_html += "- <b>" + ii + "</b>: " + "<br>"

                    l=np.array(report['boxcox']['lambda'])
                    fig,ax = plt.subplots(1)
                    ax.hist(l[:,-1])
                    ax.set_xlabel('lambda')
                    ax.set_ylabel('frequency')
                    ax.set_title('Histogram of lambda parameter')
                    fig.savefig(sdir + 'boxcox_lambda.png')

                    report_html += "<img src='./boxcox_lambda.png' width=500>" + "<p>"
                    report_html += "Data located in " + sdir + "boxcox_lambda.csv <p>"

                    np.savetxt(sdir + "boxcox_lambda.csv", l, delimiter=",")

                else:

                    report_html += "- <b>" + ii + "</b>: " + str(report[i][ii]) + "<br>"

    report_html += '</body></html>'


    with open(sdir + 'derivation_report.html', 'w') as f:
        f.write(report_html)
    f.close()
