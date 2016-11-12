"""This script computes the distances between different rankings recommended 
for each user by different algorithms and plots the histograms of the 
distances for each algorithm."""



import itertools
import re
import glob
import pandas as pd
import scipy.stats
import sys
import argparse
import numpy as np
import logging
import os.path
import multiprocessing as mp
import stats.metrics
import distances as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stats.file_input import rankings_dict






def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+",
                        help="ranking files to be compared.")
    parser.add_argument("-o", "--output", default="distances.csv",
                        help="output csv to be written. (default: %(default)s)")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="number of processes for parallel execution. "
                        "(default: %(default)s)")
    parser.add_argument("-l", "--length", type=int, default=20,
                        help="length of the rankings to be considered")
    return parser.parse_args()

def get_distances(distances_matrix,alg1,alg2):
    
    if alg2 in distances_matrix[alg1].columns:
        #print("DIRECT ORDER")
        return distances_matrix[alg1][alg2]        
    else:
        #print("INVERSE ORDER")
        return distances_matrix[alg2][alg1]





'''
This function receives a dictionary containg dataframes with the distances 
between the rankings recommended to the users by distinc algorithms and plots 
series of histogram for each pair of algorithms

'''
def plot_histograms(values,out_dir='./'):


    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    alg_names = sorted(list(values.keys()))
    log_distances =  open(os.path.join(out_dir,'log_distances'),'w')

    for alg1 in alg_names:
        #creates a set of plots where the histograms of the distances of 
        #each algortihm will be ploted
        log_distances.write(alg1+'\n')
        f, axxr = plt.subplots(len(alg_names)-1, sharey=True)
        f.set_figheight(10)
        idx = 0
        ymax_value = 0
        quartiles_mean = []
        for alg2 in alg_names:
            if alg1 != alg2:
                usrs_distances = np.array(get_distances(values,alg1, alg2))
                val_bars,bins,patch = axxr[idx].hist(usrs_distances,
                                        bins=np.linspace(0,1,50),normed=True)
                axxr[idx].set_title(alg2)                
                ymax_value = max(ymax_value,max(val_bars))
                quartiles_mean.append([np.percentile(usrs_distances,x) for x in [25,50,75]])    
                #vlines = [np.percentile(usrs_distances,x) for x in [25,50,75]]
                quartiles_mean[-1].append(usrs_distances.mean())

                log_distances.write(alg2+',%.2f,%.2f,%.2f,%.2f\n' %tuple(quartiles_mean[-1])) 
                idx += 1
        log_distances.write("\n")
        #ploting vertical lines to mark the quartiles and the mean
        #We compare each algorithm with len(alg_names)-1 other algs
        for i in range(len(alg_names)-1):    
            vcolors = ['r','g','black','c']
            labels=['Q1','Q2','Q3','Avg']
            [axxr[i].vlines(val,ymin=0, ymax=ymax_value+1, color=c, 
            linestyle='dashed',label=l) for val,c,l in zip(quartiles_mean[i],vcolors,labels)]               
                         
    
        axxr[0].legend(bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(out_dir,alg1+'_distances.pdf'))
        plt.close()


if __name__ == '__main__':


    args = parse_args()

    alg_files = sorted(glob.glob('../../../datasets/ml-100k/u1*.out'))
    algs = dist.load_algs(args.files,args.length)
    dist_values = dist.distance_matrix_users(algs,dist.kendall,1)
    plot_histograms(dist_values)


'''    for i in range(len(alg_files)):
        #for j in range(i+1,len(alg_files)):
        for j in range(len(alg_files)):
    
            if i != j:
                print(dist.get_name_from_path(alg_files[i]),"->",dist.get_name_from_path(alg_files[j]))
                test = get_distances(dist_values,dist.get_name_from_path(alg_files[i]), dist.get_name_from_path(alg_files[j]))
                f = plt.figure()
                plt.hist(test,bins=np.linspace(0,1,50),normed=True)
                plt.savefig(dist.get_name_from_path(alg_files[i])+dist.get_name_from_path(alg_files[j])+'.png')
                plt.close()
'''    

