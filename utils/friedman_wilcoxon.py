from scipy import stats
import sys
import numpy as np



if __name__ == '__main__':

    fname = sys.argv[1]
    flines = open(fname)
    header = flines.readline().strip().split(";")
    num_metrics = len(header) - 1 

    algorithms = {}

    for l in flines:
        if l.strip(): #check if teh line is empty
            tokens = l.strip().split(';')
            if not algorithms.has_key(tokens[0]):
                algorithms[tokens[0]] = [[float(x)] for x in tokens[1:]]
            else:
                i = 1
                for metric_vet in algorithms[tokens[0]]:
                    metric_vet.append(float(tokens[i]))
                    i += 1


    for metric in range(num_metrics):
        print "Metric " + str(metric)        
        res_friedman = stats.friedmanchisquare(*(np.array(algorithms[alg][metric]) for alg in algorithms.keys()))
        print res_friedman
        
        




    



