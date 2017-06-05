import os
import sys
import matplotlib as mpl
mpl.use("PDF")
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import distances as dist
import stats.metrics
#import stats.aux
import glob
import argparse
import calc_metrics
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir",type=str,
                        help="ranking files to be compared.")

    parser.add_argument("--agg_files", type=str,
                        help="ranking files to be compared.")

    parser.add_argument("-c",'--config',type=str,default="config.json",
                        help="Config file")    
    parser.add_argument("-o", "--output", default="./",
                        help="output dir to plot the histograms)")
    parser.add_argument("-p","--part", default="u1",
                        help="Partition to compute the distances")
    parser.add_argument("-n", "--num_processes", type=int, default=1,
                        help="number of processes for parallel execution. "
                        "(default: %(default)s)")
    parser.add_argument("-l", "--length", type=int, default=20,
                        help="length of the rankings to be considered")
    
    parser.add_argument("-d","--dist_func",type=str,default="kendall",
                        help="Distance function to be used when comparing the rankings")
    return parser.parse_args()


COLORS = ["b","r","g","p"]
SYMBOLS = ["o","v","s","^"]

if __name__ == "__main__":

    args = parse_args()

    print("1111")
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    print("2222")
    symbol_iter = iter(SYMBOLS)
    color_iter = iter(COLORS) #CHANGE TO A COLORMAP

    files = sorted(glob.glob(os.path.join(args.basedir,args.part+"*.out")))
    test_file = os.path.dirname(files[0])
    test_file = calc_metrics.read_test(os.path.join(test_file,args.part+".test"))

    algs = dist.load_algs(files,10)
    sorted_algs_recomm = sorted(algs.keys())
    distance_matrix_recomm = dist.distance_matrix(algs,dist.kendall_samuel,num_processes=4)

    mean_distance_recomm = [np.average(distance_matrix_recomm[i]) for i in range(len(distance_matrix_recomm))]
    map_values_recomm = [calc_metrics.MAP(algs[alg],test_file) for alg in sorted_algs_recomm]

    x_max = max(mean_distance_recomm)
    y_min = min(map_values_recomm)
 

    fig, ax = plt.subplots()
    #ax.set_title('Pairwise distances between recommendation algorithms',loc='botton')
    # Format
    fig = plt.gcf()
    #fig.set_size_inches(13, 13)
    ax.set_axis_bgcolor('white')
    ax.set_axis_on()
    #ax.grid(True)
    
    plt.axhline(0, color='black')
    plt.axvline(0.4, color='black')
    plt.axhline(0.4, color='black')
    plt.axvline(1, color='black')

    #ax.grid(color='gray',which='major')
    #ax.tick_params(color='b')

    recomm_algs_plt = ax.scatter(mean_distance_recomm, map_values_recomm, color=next(color_iter), marker=next(symbol_iter),label="Recomm. Alg")

    ax.legend(scatterpoints=1,prop={'size':12})
    ax.set_xlabel("Ranking average distance",fontsize=12)
    ax.set_ylabel("MAP",fontsize=12)
    ax.set_xlim(0.4,1)
    ax.set_ylim(0,0.4)	
    plt.savefig(os.path.join(args.output,"dist_metrics_scatter_recomm.pdf"),bbox='tight')
    #plt.close()

    





    #--------------------------------------------------------------------------
    agg_plots = []
    folder_names = []
    nested_folders = []

    if args.agg_files:
        #recomm_algs_plt = plt.scatter(mean_distance_recomm, map_values_recomm, color=next(color_iter), marker=next(symbol_iter))
        nested_folders = glob.glob(os.path.join(args.agg_files,"*/"))
    
        if len(nested_folders) == 0:
            nested_folders = glob.glob(args.agg_files)

        nested_folders = sorted(nested_folders)

        for folder in nested_folders:
            algs_to_compare = glob.glob(os.path.join(folder,args.part+"*.out"))

            from_algs = dist.load_algs(algs_to_compare,10)
            sorted_from_algs = sorted(from_algs.keys())
            distance_matrix_agg = dist.distance_matrix(algs,dist.kendall_samuel,num_processes=4,from_algs=from_algs)
            mean_distance_agg = [np.average(distance_matrix_agg[i]) for i in range(len(distance_matrix_agg))]
            #TODO alterar
            map_values_agg = [calc_metrics.MAP(from_algs[alg],test_file) for alg in sorted_from_algs]
     
    
            #y_max = max(map_values_agg)

            agg_plots.append(ax.scatter(mean_distance_agg, map_values_agg,color = next(color_iter), marker = next(symbol_iter)))
            folder_names.append(folder.strip().split("/")[-2].replace("_",". "))


        
        

        folder_names.append("Recomm. Algs")
        agg_plots.append(recomm_algs_plt)

        ax.legend(tuple(agg_plots),tuple(folder_names),scatterpoints=1,prop={'size':14})
        ax.set_xlabel("Ranking average distance",fontsize=14)
        ax.set_ylabel("MAP",fontsize=14)
        
        ax.set_xlim(0.4,1)
        ax.set_ylim(0,0.4)
        ax.tick_params(axis='both', which='major', labelsize=12, colors='black')
        plt.savefig(os.path.join(args.output,"dist_metrics_from_agg.pdf"))
        plt.close()
    


    


