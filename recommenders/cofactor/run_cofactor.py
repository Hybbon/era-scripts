import CoF_pre_process as pre_CoF
import Cofactorization as CoF
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("data", type=str,
            help="path to the input files")
    p.add_argument("-o'", "--output_dir", type=str, default="",
            help="output directory (default: same as input file)")
    p.add_argument("-p", "--part", type=str, default="u1",
            help="partition used in the training")
    p.add_argument("--proc_folder", type=str, default="pro",
            help="folder containing the processed files used by CoFactor")
    p.add_argument("-v", "--no_validation", action='store_true',
            help="if specified, no validation folds are generated")
    p.add_argument("-c", "--config", type=str, default="",
            help="name of an override config file. (default: none)")


    parsed_p = p.parse_args()

    if parsed_p.output_dir == "":
        parsed_p.output_dir = parsed_p.data

    return parsed_p



if __name__ == '__main__':

    args = parse_args()
    print "-----------------------------"

    if 'reeval' in args.data:
        print(args)
        args.no_validation = True

    pre_CoF.run(args) 
    CoF.run(args) #pass args
    os.system('rm -r {0}*'.format(os.path.join(args.data,args.proc_folder,args.part)))
