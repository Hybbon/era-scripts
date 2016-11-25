#!/usr/bin/env python3

"""
run_recommenders.py
===================
Recommender algorithm multiprocess batch runner
-----------------------------------------------

This script is used for automating the execution of recommenders with
datasets in the MyMediaLite rating format. As one would expect, the
folds generated by *pre_process.py* are in such format.

The script supports all item recommenders supported by MyMediaLite_,
as well as LibRec_'s (which are generally slower, but still supported),
PureSVD (as implemented at `alibezz`_'s BayesianDiversity_ repo) and
CofiRank_.

.. _MyMediaLite: http://www.mymedialite.net/
.. _LibRec: http://www.librec.net/
.. _alibezz: https://github.com/alibezz
.. _BayesianDiversity: https://github.com/alibezz/BayesianDiversity

This script runs according to RankingStats config files. (Read more
about them in the README file at the top directory of this repo.)

The algorithms to be executed are specified in the config files.
They're defined by their names. If a name isn't present in the
``non_mml_algs`` dictionary, it is assumed to be a MyMediaLite
item recommender (support for rating predictors is intended but still
not fully implemented).

Multiprocess execution is supported, albeit a warning must be made
regarding memory consumption, since the script does not manage such
resources automatically. For such settings and more, the command-line
help flag (``-h``) is the main source of information.


Requirements for the recommenders
---------------------------------

For MyMediaLite, the only strict requirement is mono. When working
with large datasets, it must be compiled with the option
``--with-large-heap=yes`` in order to avoid heap space issues.

Regarding LibRec, Java version 1.7+ is required.

In order to run PureSVD, it is required to specify via ``-p`` the path
to a *Python 2.7* installation with the following packages available:

- numpy
- scipy
- cython
- sparsesvd

All of them can be installed via *pip*. One should note that, on the
other hand, all RankingStats scripts should be run with *Python 3*.

In order to run CofiRank, one must compile it manually for their
platform before running this script. Instructions for doing so are
available at ``recommenders/cofirank/README.md``. This script uses
the **deployment** executable.
"""

import os
import argparse
import stats.aux
from random import randint
import pandas as pd
import re
from operator import itemgetter
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue
import logging
import time
import glob
import tempfile
import numpy as np
import heapq

from sklearn.model_selection import train_test_split


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    p = argparse.ArgumentParser()
    p.add_argument("data", type=str,
        help="folder containing the config file and the"
        " ranking data to be evaluated.")
    p.add_argument("-o", "--output_dir", type=str, default=".",
        help="folder where the output files should be saved, "
        "relative to data folder. (default: %(default)s)")
    p.add_argument("-c", "--config", type=str, default="",
        help="name of the config file. (default: %(default)s)")
    p.add_argument("-r", "--results", type=str, default="resultados",
        help="name of the results log file."
        " (default: %(default)s)")
    p.add_argument("-M", "--mymedialite", type=str,
        default="recommenders/mml", help="path to "
        "MyMediaLite's item recommendation binary, relative to"
        " the program's run path. (default: %(default)s)")
    p.add_argument("-C", "--cofirank", type=str,
        default="recommenders/cofirank/", help="path to"
        " CofiRank's item recommendation binary, relative to"
        " the program's run path. (default: %(default)s)")
    p.add_argument("-L", "--librec", type=str,
        default="recommenders/librec/", help="path to"
        " LibRec's directory, relative to"
        " the program's run path. (default: %(default)s)")
    try:
        py2_default = os.path.join(glob.glob(os.environ['HOME']+'/anaconda*')[0], 'bin/python2')
    except:
        py2_default = "python2"

    p.add_argument("-p", "--python2", type=str,
        default=py2_default, help="absolute path to the Python 2 binaries. Required"
        " packages: numpy, scipy, cython, sparsesvd.")
    p.add_argument("-s", "--puresvd", type=str,
        default="recommenders/bayesiandiversity/recommender.py")
    p.add_argument("-n", "--num_processes", type=int, default=1)
    p.add_argument("--Xmx", type=int, default=5,
                help="Size of the maximum heap used by the jvm when running librec")
    p.add_argument("-w", "--overwrite", action="store_true",
            help="if specified, any previous results are generated once again, "
            "instead of being kept whenever possible.")
    p.add_argument("--cofactor", type=str,
        default="recommenders/cofactor/run_cofactor.py")
    p.add_argument("--libfm", type=str,
        default="recommenders/libfm/libfm_ranking.py")
    p.add_argument("--poisson", default="recommenders/poisson/hgaprec")

    return p.parse_args()


# MML STUFF

mml_cmd = ("mono {mml_binary} "
           "--training-file={base} "
           "--test-file={test} "
           "--recommender={alg} "
           "--prediction-file={pred} "
           "--predict-items-number={num_items} "
           "--measures='AUC,prec@5,prec@10,MAP,NDCG' "
           "--random-seed={seed}")

mml_res = ("mono {mml_binary} "
           "--training-file={base} "
           "--test-file={test} "
           "--recommender={alg} "
           "--predict-items-number={num_items} "
           "--measures='AUC,prec@5,prec@10,MAP,NDCG' "
           "--random-seed={seed} "
           ">> {results}")


def mml_item_ranking(kwargs):
    command = mml_cmd.format(**kwargs)
    print(command)
    os.system(command)

    res_command = mml_res.format(**kwargs)
    print(res_command)
    os.system(res_command)


mml_rp_cmd = ("mono {mml_binary_rp} "
              "--training-file={base} "
              "--test-file={test} "
              "--recommender={alg} "
              "--prediction-file={pred_rp_tmp} "
              "--measures='AUC,prec@5,prec@10,MAP,NDCG' "
              "--random-seed={seed}")


def mml_rating_prediction(kwargs):
    tmp_filename = "{}-{}-{}.tmp.out".format(kwargs['p'], kwargs['alg'], randint(0, 999999))
    tmp_file_path = os.path.join(stats.aux.TMP_DIR, tmp_filename)
    kwargs['pred_rp_tmp'] = tmp_file_path
    command = mml_rp_cmd.format(**kwargs)
    print(command)
    os.system(command)
    # rp_to_ranking(tmp_file_path, kwargs['pred'])


mml_rating_based = ["SVDPlusPlus"]

def mml_run(kwargs):
    if kwargs['alg'] in mml_rating_based:
        mml_rating_prediction(kwargs)
    else:
        mml_item_ranking(kwargs)



# PURESVD STUFF

puresvd_cmd = ("{python2} {puresvd} {base} {pred} -n {num_items}")


def puresvd_run(kwargs):
    command = puresvd_cmd.format(**kwargs)
    print(command)
    os.system(command)


# CoFactor STUFF
CoFactor_cmd = ("{python2} {cofactor} {data_folder} -p {p}")


def cofactor_run(kwargs):
    command = CoFactor_cmd.format(**kwargs)
    print(command)
    os.system(command)




# Libfm STUFF
libfm_cmd = ("{python2} {libfm} {data_folder} -p {p} -n {num_items}")


def libfm_run(kwargs):
    command = libfm_cmd.format(**kwargs)
    print(command)
    os.system(command)


def mml_to_frame(addr):
    input_headers = ("user_id", "item_id", "rating")
    return pd.read_csv(addr, "\t", names=input_headers)


def ranking_frame_to_mml(frame, out_addr):
    with open(out_addr, "w") as out_file:
        for uid, user_frame in frame.groupby("user_id"):
            items_and_scores = []
            for row_id, row in user_frame.iterrows():
                s = "{0}:{1}".format(int(row.item_id), row.rating)
                items_and_scores.append(s)
            print("{}\t[{}]".format(uid, ",".join(items_and_scores)),
                  file=out_file)


# COFIRANK STUFF

def cr_convert_ratings(in_addr, out_addr, uids=[]):
    frame = mml_to_frame(in_addr)
    if len(uids):
        frame = frame[frame['user_id'].isin(uids)]
    else:
        uids = frame['user_id'].unique()

    with open(out_addr, "w") as out_file:
        for uid, user_frame in frame.groupby("user_id"):
            str_pairs = []
            for row_id, row in user_frame.iterrows():
                str_pairs.append("{0}:{1}".format(row['item_id'],
                                                  row['rating']))
            out_file.write(" ".join(str_pairs) + "\n")
    return uids


float_regex = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:h[eE][+-]?\d+)?'


def cr_convert_output(in_addr, out_addr, uids, num_items):
    with open(in_addr, "r") as in_file, open(out_addr, "w") as out_file:
        for uid, scores_line in zip(uids, in_file):
            score_regex = re.compile("(\d+):({0})".format(float_regex))
            score_tuples = score_regex.findall(scores_line)
            score_tuples.sort(key=itemgetter(1), reverse=True)
            trimmed = score_tuples[:num_items]
            str_list = ["{0}:{1}".format(item_id, score)
                        for item_id, score in trimmed]
            out_file.write("{0}\t[{1}]\n".format(uid, ",".join(str_list)))


cr_cmd = ("{cr_binary} {cr_cfg}")


def prepare_cr_settings(p, cr_dir):

    config_dir = os.path.join(cr_dir, "config")
    default_config = os.path.join(config_dir, "default.cfg")
    new_config = os.path.join(config_dir, "config_{}.cfg".format(p))

    def copy(src, dest):
        cmd = "cp {src} {dest}"
        os.system(cmd.format(src=src, dest=dest))

    copy(default_config, new_config)

    def sed(old, new, path):
        cmd = "sed -i 's:{old}:{new}:' {path}"
        os.system(cmd.format(old=old, new=new, path=path))


    data_dir = os.path.join(cr_dir, "data")

    sed("cofirank/data/dummytrain", os.path.join(data_dir, p + ".base"), new_config)
    sed("cofirank/data/dummytest", os.path.join(data_dir, p + ".test"), new_config)

    cr_out_dir = os.path.join(cr_dir, "out_" + p)
    if not os.path.isdir(cr_out_dir):
        os.mkdir(cr_out_dir)

    sed("out/", cr_dir[:-1] + "/out_"+p+"/", new_config)

    return new_config


def cr_run(kwargs):
    print("{0} -> {1}".format(kwargs['base'], kwargs['cr_base']))
    uids = cr_convert_ratings(kwargs['base'], kwargs['cr_base'])
    print("{0} -> {1}".format(kwargs['test'], kwargs['cr_test']))
    cr_convert_ratings(kwargs['test'], kwargs['cr_test'], uids=uids)
    command = cr_cmd.format(**kwargs)
    print(command)
    os.system(command)
    print("{0} -> {1}".format(kwargs['cr_out'], kwargs['pred']))
    cr_convert_output(kwargs['cr_out'], kwargs['pred'], uids,
                      kwargs['num_items'])


# LIBREC STUFF


def librec_make_config(alg, kwargs, cfg_path, out_path):
    template_path = os.path.join(kwargs['librec_template_dir'], alg + ".conf")



    with open(template_path) as template, open(cfg_path, "w") as out:
        for line in template:
            out.write(line + "\n")

        base_in = "dataset.ratings=./{0}".format(kwargs['base'])
        test_in = "evaluation.setup=test-set -f ./{0} --rand-seed 1 --test-view all".format(kwargs['test'])
        output_setup = "output.setup=on -dir ./{0}/".format(out_path)
        ranking_setup = "item.ranking=on -topN {0}".format(kwargs['num_items'])

        for line in [base_in, test_in, output_setup, ranking_setup]:
            out.write(line + "\n")

librec_cmd = "java -Xmx{Xmx}g -jar {librec_binary} -c {librec_cfg}"

def librec_convert_output(from_path, to_path):

    line_regex = re.compile(r"(\d+): (.*)")
    item_regex = re.compile(r"(\d+)\*?, ({0})".format(float_regex))

    #SAMUEL - Algumas vezes nome do arquivo de saida eh diferente do nome do
    #algoritmo (Ex.: FISM -> FISMauc) o try catch abaixo tem como objetivo
    #resolver este problema

    in_file = None

    try:
        in_file = open(from_path)
    except:
        out_dir = os.path.join(*from_path.split("/")[:-1])
        #pega o caminha do arquivo que realmente tem a saida
        actual_file = glob.glob(os.path.join(out_dir,'*items*'))[-1]
        in_file = open(actual_file)

    with open(to_path, "w") as out_file:
        next(in_file)  # skip comment header
        for line in in_file:
            user_id, rest = line_regex.match(line).groups()
            items = item_regex.findall(rest)
            s = "{0}\t[{1}]\n".format(user_id,
                                    ",".join((uid + ":" + rating
                                              for uid, rating in items)))
            out_file.write(s)

    in_file.close()


def librec_run(alg, kwargs):
    dir_name = "librec-{0}-{1}-{2}".format(kwargs['p'], alg, randint(0, 999999))
    run_dir = os.path.join(stats.aux.TMP_DIR, dir_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    cfg_path = os.path.join(run_dir, "run.conf")
    librec_make_config(alg, kwargs, cfg_path, run_dir)

    out_file = "{0}-top-{1}-items.txt".format(alg, kwargs['num_items'])
    out_path = os.path.join(run_dir, out_file)

    print(librec_cmd.format(librec_cfg=cfg_path, **kwargs))

    os.system(librec_cmd.format(librec_cfg=cfg_path, **kwargs))
    print("File to write -----> "+out_file)
    librec_convert_output(out_path, kwargs['pred'])
    print("Converte out do LibRec " + alg + ": " + out_path + " -> " + kwargs['pred'])


def librec_closure(alg):
    def func(kwargs):
        return librec_run(alg, kwargs)
    return func


# POISSON STUFF

def load_hga_matrix(path):
    rows = []
    uids = []
    with open(path) as f:
        for line in f:
            i, uid, *row = line[:-1].split()
            uids.append(int(uid))
            rows.append([float(s) for s in row])
    return uids, np.array(rows)


def subdirectories(dir):
    subdirs = []
    for item in os.listdir(dir):
        path = os.path.join(dir, item)
        if os.path.isdir(path):
            subdirs.append(path)
    return subdirs


def save_poisson_files(dir, train_frame, valid_frame, test_frame):
    for frame, filename in ((train_frame, 'train.tsv'),
                            (valid_frame, 'validation.tsv'),
                            (test_frame, 'test.tsv')):
        frame = frame[['user_id', 'item_id']]
        frame = frame.drop_duplicates()
        frame = frame.sort_values(['user_id', 'item_id'])
        frame['rating'] = [1] * len(frame)
        # frame['timestamp'] = [1241459650] * len(frame)
        frame['timestamp'] = range(1241459650, 1241459650 - len(frame), -1)

        path = os.path.join(dir, filename)
        frame.to_csv(path, sep='\t', header=None, index=None)
    test_users_path = os.path.join(dir, "test_users.tsv")
    test_users = pd.Series(sorted(test_frame.user_id.unique()))
    test_users.to_csv(test_users_path, index=False)
    print(len(test_users))


def compute_rating_matrix(dir):
    iids, beta = load_hga_matrix(os.path.join(dir, "beta.tsv"))
    uids, theta = load_hga_matrix(os.path.join(dir, "theta.tsv"))
    ratings = beta @ theta.T
    return iids, uids, ratings


def build_ranking_frame(dir, train_frame, valid_frame, n=100):
    iids, uids, ratings = compute_rating_matrix(dir)
    tuples = []
    for uid, user_ratings in zip(uids, ratings.T):
        def seen(frame):
            return set(frame[frame.user_id == uid].item_id)
        seen_items = seen(train_frame) | seen(valid_frame)

        unseen = [(rating, iid)
                  for (rating, iid) in zip(user_ratings, iids)
                  if iid not in seen_items]
        for rating, iid in heapq.nlargest(n, unseen):
            tuples.append((uid, iid, rating))
    columns = ("user_id", "item_id", "rating")
    frame = pd.DataFrame.from_records(tuples, columns=columns)
    return frame


def poisson_split(kwargs):
    train_file, test_file = kwargs['base'], kwargs['test']
    train_valid_frame = mml_to_frame(train_file)

    train_frame = pd.DataFrame()
    valid_frame = pd.DataFrame()

    for uid, user_frame in train_valid_frame.groupby('user_id'):
        train_ratings, valid_ratings = train_test_split(user_frame)
        train_frame = train_frame.append(train_ratings)
        valid_frame = valid_frame.append(valid_ratings)

    # Fix items being in validation but not in training
    not_in_train = valid_frame.item_id.apply(lambda x: x in train_frame.item_id)
    train_frame = train_frame.append(valid_frame[not_in_train])
    valid_frame = valid_frame[~not_in_train]
    test_frame = mml_to_frame(test_file)

    return train_frame, valid_frame, test_frame


poisson_cmd = "{poisson_binary} -n {users} -m {items} -dir {poisson_dir} -k {poisson_factors}"

def poisson_run(kwargs):
    with tempfile.TemporaryDirectory() as run_dir:
        train_frame, valid_frame, test_frame = poisson_split(kwargs)

        save_poisson_files(run_dir, train_frame, valid_frame, test_frame)
        kwargs = kwargs.copy()
        kwargs['users'] = test_frame.user_id.max() + 1
        kwargs['items'] = test_frame.item_id.max() + 1
        kwargs['poisson_dir'] = run_dir
        kwargs['poisson_factors'] = 5
        prev_cwd = os.getcwd()
        # Change directory to the temporary directory, so that hgaprec
        # creates the result directory inside it.
        os.chdir(run_dir)
        cmd = poisson_cmd.format(**kwargs)
        print(">>", cmd)
        print("at", os.getcwd())
        os.system(cmd)
        # Return to the previous current working directory
        os.chdir(prev_cwd)
        subdirs = subdirectories(run_dir)
        out_dir = subdirs[0]
        ranking_frame = build_ranking_frame(out_dir, train_frame, test_frame)
        ranking_frame_to_mml(ranking_frame, kwargs['pred'])


def arg_set_for_run(p, alg, args, conf):
    base_filename = conf['base_form'].format(p)
    hits_filename = conf['hits_form'].format(p)
    out_filename = conf['alg_form'].format(p, alg)

    num_items = conf['len_ranking']

    base_str = os.path.join(args.data, base_filename)
    hits_str = os.path.join(args.data, hits_filename)

    out_dir = os.path.join(args.data, args.output_dir)
    out_str = os.path.join(out_dir, out_filename)
    results_str = os.path.join(out_dir, args.results)

    mml_binary_str = os.path.join(args.mymedialite, "item_recommendation.exe")
    mml_binary_rp_str = os.path.join(args.mymedialite, "rating_prediction.exe")

    cr_dir = args.cofirank
    cr_binary_str = os.path.join(cr_dir, "dist/cofirank-deploy")

    cr_cfg_str = prepare_cr_settings(p, cr_dir)

    cr_base_str = os.path.join(cr_dir, "data/"+str(p)+".base.lsvm")
    cr_test_str = os.path.join(cr_dir, "data/"+str(p)+".test.lsvm")
    cr_out_str= os.path.join(cr_dir, "out_"+str(p)+"/F.lsvm")

    python2_binary_str = args.python2
    puresvd_script_str = args.puresvd
    cofactor_script_str = args.cofactor #SAMUEL
    libfm_script_str = args.libfm #SAMUEL

    librec_binary_str = os.path.join(args.librec, "librec.jar")
    librec_template_dir_str = os.path.join(args.librec, "conf")

    data_folder = args.data #SAMUEL

    poisson_binary_str = os.path.abspath(args.poisson)

    kwargs = {
        'python2': python2_binary_str,
        'puresvd': puresvd_script_str,
        'cofactor': cofactor_script_str,
        'libfm': libfm_script_str,
        'mml_binary': mml_binary_str,
        'mml_binary_rp': mml_binary_rp_str,
        'cr_binary': cr_binary_str,
        'cr_cfg': cr_cfg_str,
        'cr_base': cr_base_str,
        'cr_test': cr_test_str,
        'cr_out': cr_out_str,
        'librec_binary': librec_binary_str,
        'librec_template_dir': librec_template_dir_str,
        'base': base_str,
        'test': hits_str,
        'alg': alg,
        'p': p,
        'Xmx' : args.Xmx,
        'data_folder' : data_folder, #SAMUEL
        'pred': out_str,
        'num_items': num_items,
        'seed': 123,
        'results': results_str,
        "poisson_binary": poisson_binary_str
    }

    return kwargs

non_mml_algs = {
    "pureSVD": puresvd_run,
    "CofiRank": cr_run,
    "WRMF_librec": librec_closure("WRMF"),
    "ItemKNN_librec": librec_closure("ItemKNN"),
    #SAMUEL - adicionando novos algoritmos
    "NMF_librec" : librec_closure("NMF"),
    "LDA_librec" : librec_closure("LDA"),
    "RankALS_librec" : librec_closure("RankALS"),
    "AoBPR_librec" : librec_closure("AoBPR"),
    "BHfree_librec" : librec_closure("BHfree"),
    "SLIM_librec" : librec_closure("SLIM"),
    "SVD++_librec" : librec_closure("SVD++"),
    "GBPR_librec" : librec_closure("GBPR"),
    "LRMF_librec" : librec_closure("LRMF"),
    "BUCM_librec" : librec_closure("BUCM"),
    "Hybrid_librec" : librec_closure("Hybrid"),
    "PRankD_librec" : librec_closure("PRankD"),
    "FISM_librec" : librec_closure("FISM"),
    "CoFactor" : cofactor_run,
    "libfm" :   libfm_run,
    "Poisson": poisson_run
}




def run(Q):
#def run(arg_set):
    print(Q)
    print(Q.qsize())

    while not Q.empty():

        arg_set = Q.get()


        alg = arg_set['alg']
        p = arg_set['p']
        start_time = time.time()
        logging.warning("Begin execution of {p}-{alg}".format(p=p, alg=alg))
        print("Begin execution of {p}-{alg}".format(p=p, alg=alg))
        try:
            if alg in non_mml_algs:
                non_mml_algs[alg](arg_set)
            else:
                mml_run(arg_set)
        except Exception as inst:
            logging.error("Error when running "+alg +"- "+p)
            err_time = time.time() - start_time
            hours, rem = divmod(err_time, 3600)
            minutes, seconds = divmod(err_time, 60)
            delta = "{h}h{m}m{s}s".format(h=hours, m=minutes, s=seconds)

            logging.error("{p}-{alg} - Total time until error {delta}".format(
                        p=p,alg=alg,delta=delta))
            logging.error("{p} - {alg} ".format(p=p,alg=alg) + str(inst))
            logging.error("{p} - {alg} ".format(p=p,alg=alg) + str(type(inst)))

            print("Error when running "+alg +"- "+p)
            print(inst)
            print(type(inst))
            logging.warning("Try to continue tread with the next job in the queue")
            continue

        duration = time.time() - start_time
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(duration, 60)
        delta = "{h}h{m}m{s}s".format(h=hours, m=minutes, s=seconds)
        logging.warning("Finish execution of {p}-{alg} em {delta}".format(p=p, alg=alg, delta=delta))
        print("Finish execution of {p}-{alg} em {delta}".format(p=p, alg=alg, delta=delta))



def main():
    args = parse_args()


    log_filename = args.config.replace('json','log')
    log_path = os.path.join(args.output_dir,log_filename)

    logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='[%(levelname)s] - %(asctime)s: %(message)s')



    conf = stats.aux.load_configs(stats.aux.CONF_DEFAULT,
                            os.path.join(args.data, stats.aux.BASE_CONF),
                            args.config)



    logging.info("*******************STARTING*********************")

    logging.info("Running for the following recommenders: " +
            ",".join(conf['algs']))

    logging.info("Running using %d processes" %(args.num_processes))

    base_conf = conf['base']


    Q = Queue()
    parallel_arg_sets = []
    for alg in conf['algs']:
        print(alg + " pre-processing")
        consumers = []
        for p in conf['parts']:
            arg_set = arg_set_for_run(p, alg, args, base_conf)
            if args.overwrite or not os.path.isfile(arg_set['pred']):
                Q.put(arg_set)
                parallel_arg_sets.append(arg_set)
                logging.info("%s - %s added to processing queue" %(p,alg))
                print(p + " added to processing queue")
            else:
                print(p + "-" + alg + " has already been processed")




    consumers = []
    for i in range(args.num_processes):
        try:
            p = Process(target=run,args=(Q,))
            consumers.append(p)
            p.start()
        except Exception as e:
            print("Exception in the main")
            print(e)


    for pp in consumers:
        try:
            #print("waiting to join")
            print(pp)
            pp.join()
        except Exception as e:
            print("Exception waiting join")
            print(e)



    # = mp.Pool(processes=args.num_processes)
    #pool.map(run, parallel_arg_sets)

if __name__ == "__main__":
    main()

















