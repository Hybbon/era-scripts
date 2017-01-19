
import os
import sys
import time
import argparse
import convert_to_letor_format as conversion
import create_ranking_from_scores as rank_creator





RANKER_NAMES = {0: "MART", 
                1: "RankNet", 
                2: "RankBoost", 
                3: "AdaRank", 
                4: "CoordinateAscent", 
                6: "LambdaMART", 
                7: "ListNet", 
                8: "RandomForests"}


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("-base_dir",type=str,default="",
        help = "Folder containing the datasets used in the learning process")
    p.add_argument("-o","--out_dir",type=str,
        help="Folder where the output rankings will be saved")
    p.add_argument("-size",type=int,default=10,
        help="Size of the output rank that will be saved")
    p.add_argument("-ranklib",type=str,default="./RankLib-2.1.jar")
    p.add_argument("-p","--part",type=str,default="u1")
    p.add_argument("-ranker", type=int, default = 6,
        help = "Ranker thar will be used in the l2r 0: MART 1: RankNet 2: RankBoost 3: AdaRank 4: Coordinate Ascent 6: LambdaMART 7: ListNet 8: Random Forests")
    p.add_argument("-ranklib_tmp",type=str,default="ranklib_tmp/")
    p.add_argument("-converted",action="store_true",
            help="When set indicates that the conversion to the LETOR format is already done")

    p.add_argument("-Xmx",type=int,default=10)

    return p.parse_args()




def convert_datasets(args):

    #construct the datasets
    partitions = ["u"+str(i) for i in range(1,6)]
    for part in partitions:
        cmd = "python convert_to_letor_format.py -data_folder " +args.base_dir+"/classif/" +" -p "+part
        os.system(cmd)
        cmd_reeval = "python convert_to_letor_format.py -data_folder " +args.base_dir+"reeval/" +" -p "+part + " -test"
        os.system(cmd_reeval)


def run_ranklib(args):
    
    if not os.path.isdir(os.path.join(args.out_dir,args.ranklib_tmp)):
        os.mkdir(os.path.join(args.out_dir,args.ranklib_tmp))


    timestamp = open(os.path.join(args.out_dir,args.ranklib_tmp,"timestamp"),'w')    

    partitions = ["u"+str(i) for i in range(1,6)]
    for part in partitions:
        args.part = part
        init = time.time()
        cmd = "java -Xmx{Xmx}g -jar {ranklib} -train {base_dir}/classif/{part}.train.letor -ranker {ranker} -metric2t MAP -save {out_dir}{ranklib_tmp}{part}-{0}_model.txt".format(RANKER_NAMES[args.ranker],**args.__dict__)
        os.system(cmd)
        cmd_reeval = "java -Xmx{Xmx}g -jar {ranklib} -load {out_dir}{ranklib_tmp}{part}-{0}_model.txt -rank {base_dir}/reeval/{part}.train.letor -score {out_dir}{ranklib_tmp}{part}-{0}.scores".format(RANKER_NAMES[args.ranker],**args.__dict__)
        os.system(cmd_reeval)
        create_ranking(args)
        total_time = time.time() - init

        timestamp.write(part + ":" + str(total_time)+"\n")
        timestamp.flush()

    timestamp.close()



def create_ranking(args):    

    cmd = "python create_ranking_from_scores.py -map {base_dir}/reeval/{part}.train.map -scores {out_dir}{ranklib_tmp}{part}-{0}.scores -o {out_dir}{part}-{0}.out".format(RANKER_NAMES[args.ranker],**args.__dict__)
    print(cmd)
    os.system(cmd)



if __name__ == "__main__":

    args = parse_args()
    if not args.converted:
        convert_datasets(args)
    run_ranklib(args)
    




