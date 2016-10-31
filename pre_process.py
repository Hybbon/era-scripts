#!/usr/bin/env python3

"""
pre_process.py
==============
Dataset pre-processor and cross-validation fold generator
---------------------------------------------------------

This script is used to create cross-validation folds from a dataset's
raw ratings and perform any preliminary filtering of the ratings,
according to a few criteria, such as number of ratings per user, per
item or setting a minimum relevance threshold for ratings.

Currently supported are the following dataset formats:

- mml (MyMediaLite, default)
- jester (Jester Dataset)
- bx (Book-Crossing dataset)
- yelp (Yelp Challenge Dataset)

This script runs according to RankingStats config files. (Read more
about them in the README file at the top directory of this repo.)

Running this script can be assimple as specifying the source folder,
which contains the rating file to be filtered and grouped. However,
many different settings are available, and can be checked through the
``-h`` command-line flag.

Also, in order to specify the type of dataset, the name of the source
file in the data folder and the separator character (if the dataset's
ratings are contained in a csv-format file), one must create a
``config.json`` file inside the data folder or specify an override
file through command-line parameters. This procedure is best
explained in the main README file.

The default pre-processing is done as defined below:

1. Ratings under 4 stars are pruned.
2. Items rated by less than 5% of the users are removed.
3. Users which rated less than 4 items are filtered out.
4. Folds are generated.

By default, five folds (u1 to u5) are generated at the data folder.
These folds are split between training (60%), validation (20%) and
testing (20%) ratings. Inside the new "reeval" directory, there are
folds for reevaluation, in which training files contain both
training and validation ratings.

Examples::

    python pre_process.py bases/ml100k/

    python pre_process.py -c jester_override.json bases/jester/ -v

The ``-v`` flag specifies no three-way (train-validation-test) folds
should be generated, only the regular train-test format.
"""




'''
1) O config.json deve estar dentro da pasta da base
2) alterei a atribuicao da variavel de saida


'''

import argparse
import pandas as pd
import numpy as np
import os
import math
import json
import re
from collections import namedtuple
from stats import aux
import time
import matplotlib.pyplot as plt


input_headers = ("user_id", "item_id", "rating")#, "timestamp")
types = {'user_id':np.int32,'item_id':np.int32,'rating':np.float64}#,'timestamp':np.int32}

def read_ratings_file(addr, sep="\t",default_headers=input_headers,default_types=types):
    
    frame = pd.read_csv(addr, sep,header=None)
    #remove the timestamp column
    if len(frame.columns) == 4:
        del frame[3]
    frame.columns = input_headers            
    return frame

    '''except:
        reduced_headers = ("user_id", "item_id", "rating")
        reduced_types = {'user_id':np.int32,'item_id':np.int32,'rating':np.float64}
        frame = pd.read_csv(addr, sep, names=reduced_headers,dtype=reduced_types)
        return frame
    else:
        print("Unexpected Error, pleach check read_ratings_file function and or your input files")'''


def remap_series_to_ints(col):
    """Maps a series of unique ids to integer indexes starting from 1"""
    mappings = {old_id: index for index, old_id
                in enumerate(col.unique(), start=1)}
    return col.apply(lambda old_id: mappings[old_id])


jester_headers = ("user_id", "item_id", "rating")


def read_jester_file(addr, sep="\t"):
    extension = os.path.splitext(addr)[1]
    if extension == '.xls':
        matrix = pd.read_excel(addr, header=None)
    else:
        matrix = pd.read_csv(addr, sep=sep, header=None)
    del matrix[0]  # Number of items rated by the user, irrelevant
    tuples = []
    for user_id, t in enumerate(matrix.itertuples(index=False)):
        for item_id, rating in enumerate(t, 1):
            tuples.append((user_id, item_id, rating))
    frame = pd.DataFrame(data=tuples, columns=jester_headers)

    #EDIT - Samuel: Retirei a transformacao linear para os ratings

    #def linear_transform(x):
    #    return 99 if x == 99 else math.ceil((x + 10) / 4)

    #frame['rating'] = frame['rating'].apply(linear_transform)
    frame = frame[frame['rating'] != 99]

    return frame


bx_headers = ("user_id", "isbn", "rating")


def read_bx_file(addr, sep="\t"):
    frame = pd.read_csv(addr, sep=sep, names=bx_headers, header=0)
    frame['item_id'] = remap_series_to_ints(frame['isbn'])
    #frame['rating'] = frame['rating'].apply(lambda x: math.ceil(x / 2))

    return frame

hash_regex_str = r'[0-9A-Za-z_-]{22}'
yelp_regex_str = (r'.*"user_id": "(?P<user_hash>{hash})"'
                  '.*"stars": (?P<rating>\d)'
                  '.*"business_id": "(?P<item_hash>{hash})"'.format(
                      hash=hash_regex_str
                      ))
yelp_regex = re.compile(yelp_regex_str)
yelp_headers = ("user_hash", "rating", "item_hash")


def read_yelp_file(addr, **kwargs):
    """Reads reviews from the Yelp Challenge dataset to a pd.DataFrame.

    Yelp review files have one json object per line. The desired properties
    are 'user_id', 'business_id' and 'stars'. The former two are string unique
    identifiers. The latter is a integer star rating from 1 to 5 (no rescaling
    is needed!).
    """
    tuples = []
    with open(addr) as f:
        for json_str in f:
            m = yelp_regex.match(json_str)
            t = (m.group('user_hash'), int(m.group('rating')),
                 m.group('item_hash'))
            tuples.append(t)
    frame = pd.DataFrame.from_records(tuples, columns=yelp_headers)
    frame['user_id'] = remap_series_to_ints(frame['user_hash'])
    frame['item_id'] = remap_series_to_ints(frame['item_hash'])
    return frame


base_settings = {
    'default': {
        'read_function': read_ratings_file
    }, 'jester': {
        'read_function': read_jester_file
    }, 'bx': {
        'read_function': read_bx_file,
        'use_abs_rating_counts': True
    },
    'yelp': {
        'read_function': read_yelp_file,
        'use_abs_rating_counts': True
    }
}


out_cols = ("user_id", "item_id", "rating")


def save_ratings_file(addr, ratings):
    print("Saving new file at {0}".format(addr))
    ratings.to_csv(addr, header=False, index=False, sep="\t", columns=out_cols)


def rating_frequency_dict(ratings):
    counts = ratings.groupby('item_id')['user_id'].count()
    num_users = ratings['user_id'].nunique()
    return {item_id: count/num_users for item_id, count in counts.items()}


def rating_count_dict(ratings):
    counts = ratings.groupby('item_id')['user_id'].count()
    return {item_id: count for item_id, count in counts.items()}


def user_count_dict(ratings):
    counts = ratings.groupby('user_id')['item_id'].count()
    return {user_id: count for user_id, count in counts.items()}




'''
Retorna duas matrizes de rating, a primeira contem somente os ratings relevantes
e a segunda contem os ratings considerados nao relevantes. A união das duas 
matrizes forma a matriz original

1)Alterar para passar a metrica desejada, atualmente estou usando mediana

'''
def personalized_rating_normalization(ratings,metric='median'):

    uniq_users = ratings.user_id.unique()
    ratings['to_remove'] = pd.Series(np.zeros(len(ratings)),index=ratings.index)

    '''medians = {}

    for row in ratings.index:
        usr = ratings.iloc[row].user_id
        if usr in medians:
            medians[usr].append(ratings.loc[row].rating)
        else:
            medians[usr] = [ratings.loc[row].rating]


    for key in medians.keys():
        medians[key] = np.median(medians[key])

    #ratings.sort_values(by='user_id')
    ini1 = time.time()
    user_medians = ratings.groupby(['user_id'])['rating'].median()
    print('1st' + str(time.time()-ini1))

    ratings[ratings]

    ini2 = time.time()
    for row in ratings.index:
        usr = int(ratings.iloc[row].user_id)
        if ratings.loc[row].rating < medians[usr]:
            ratings.loc[row,'to_remove'] = 1
    

    print('2st' + str(time.time()-ini2))'''

    ini1 = time.time()
    user_medians = ratings.groupby(['user_id'])['rating'].median()
    for user in uniq_users:
        ratings.loc[(ratings.user_id == user) & (ratings.rating < user_medians[user]),'to_remove'] = 1
    print('1st' + str(time.time()-ini1))




    '''ini2 = time.time()
    
    for user in uniq_users:
        if metric == 'median':
            med_rating = ratings[ratings['user_id']==user]['rating'].median()
        elif metric == 'mean':
            med_rating = ratings[ratings['user_id']==user]['rating'].mean()
        else:
            med_rating = 4

        ratings.loc[(ratings.user_id == user) & (ratings.rating < med_rating),'to_remove'] = 1
        #to_replace = list(range(int(med_rating)))
        #default_val = [-999 for _ in to_replace]
        #ratings[ratings.user_id ==user]['rating'].replace(to_replace=to_replace,value=default_val,inplace=True)                
        #primeiramente da um drop em todos os registros para o usuario 'usr'
        #depois concatena com o bkp que foi armazenado na linha anterior
        #ratings = pd.concat([ratings[ratings.user_id != user],bkp])
    
    print('2st' + str(time.time()-ini2))'''
    return ratings[ratings['to_remove'] != 1], ratings[ratings['to_remove'] == 1]





'''def personalized_rating_normalization2(ratings):

    #1) sort the ratings dataframe
     2) use ratings.iter_rows() to iterate in the dataframe


    uniq_users = ratings.user_id.unique()

    ratings['to_remove'] = pd.Series(np.zeros(len(ratings)),index=ratings.index)
    ratings = ratings.sort(columns='user_id')

    
    for i in range(1,len(ratings)):
        
        tokens = ratings.irow(i)
        if tokens.user_id == ratings.irow(i-1):
            med_ratings += tokens.rating
            num_user_ratings += 1
        else:
                

    for user in uniq_users:
        med_rating = ratings[ratings['user_id']==user]['rating'].median()        
        ratings.loc[(ratings.user_id == user) & (ratings.rating < med_rating),'to_remove'] = 1
        #to_replace = list(range(int(med_rating)))
        #default_val = [-999 for _ in to_replace]
        #ratings[ratings.user_id ==user]['rating'].replace(to_replace=to_replace,value=default_val,inplace=True)                
        #primeiramente da um drop em todos os registros para o usuario 'usr'
        #depois concatena com o bkp que foi armazenado na linha anterior
        #ratings = pd.concat([ratings[ratings.user_id != user],bkp])
  

    return ratings[ratings['to_remove'] != 1], ratings[ratings['to_remove'] == 1]
'''








#SAMUEL
'''
This funtion ensures that the same users and itens used in the train files 
will be present in the validation (TODO check if it is necessary, or even possible
to do the same thing with the test files)
TODO check if I need to do the same thing for users
'''
def ensure_same_users_items(train_data,val_data,test_data):

    ini = time.time()    

    print("item train {0} items val {1}".format(len(train_data.item_id.unique()),len(val_data.item_id.unique())))
    
    unique_items_train = set(train_data.item_id.unique())
    '''unique_users = [val_data.user_id.unique(),test_data.user_id.unique()]'''
    
    #removing users that are not in all files
    #TODO check if it is the best way to do this
    #union_users = set(train_data.user_id.unique())
    #TODO remove
    '''intersection_users = set(train_data.user_id.unique())
    for x in unique_users:
        union_users = union_users.union(x)
        intersection_users = intersection_users.intersection(x)
    '''

    '''users_not_in_all = union_users - intersection_users

    print (users_not_in_all)
    print("Users {0} in union {1} in intersection".format(len(union_users),len(intersection_users)))
    
    users_not_in_all_aux = train_data['user_id'].isin(users_not_in_all)
    train_data = train_data[~users_not_in_all_aux]
    users_not_in_all_aux = val_data['user_id'].isin(users_not_in_all)
    val_data = val_data[~users_not_in_all_aux]
    users_not_in_all_aux = test_data['user_id'].isin(users_not_in_all)
    test_data = test_data[~users_not_in_all_aux]
    print("Users {0} in union {1} in intersection".format(len(union_users),len(intersection_users)))
    #------------------------------------------------------------------
    '''

    #Change the files in a way that the training file contains all possible items
    unique_items = unique_items_train.union(val_data.item_id.unique())
    left_sid = list(unique_items - unique_items_train)
    
    #TODO remove
    '''for x in val_data.item_id.unique():
        unique_items.add(x)    

    left_sid = list()
    for i, sid in enumerate(unique_items):
        if sid not in train_data.item_id.unique():
            left_sid.append(sid)
    '''
        
    items_not_in_train = val_data['item_id'].isin(left_sid)

    tr_data = train_data.append(val_data[items_not_in_train])
    val_data = val_data[~items_not_in_train]

    test_data = test_data[test_data['item_id'].isin(unique_items)]

    print("unique items "+ str(time.time()-ini))

def filter_ratings(ratings, min_freq=0.05, min_rating=4, min_cnt=10,
                   min_ratings=10, use_abs_rating_counts=False, **kwargs):
    """Removes non-frequent movies or ratings under a certain score."""


    print("Argumentos")
    print(kwargs)
    process_stats = []
    users_countings = []
    items_countings = []
    # Converts all ratings to integers
    ratings['rating'] = ratings['rating'].apply(lambda x: int(x))
    print("Initial length: {0}".format(len(ratings)))
    
    uniq_users = ratings.user_id.unique()
    process_stats.append(datailed_dataset_stats(ratings))
    
    users_countings.append([len(ratings[ratings.user_id == x]) for x in uniq_users])   
    '''plt.hist(num_ratings_per_user,bins=75)
    print(a)   
    plt.savefig('step1.pdf')
    plt.close()'''

    #1) Remove ratings lower than the median of the user's ratings
    ratings, irrel_ratings = personalized_rating_normalization(ratings,kwargs['fmetric'])
    
    #ratings = ratings[ratings['rating'] >= min_rating]
    print("After removal of ratings under {1}: {0}".format(len(ratings),
                                                           min_rating))
    process_stats.append(datailed_dataset_stats(ratings))
    print_dataset_stats(ratings)

    users_countings.append([len(ratings[ratings.user_id == x]) for x in uniq_users])
    '''uniq_users = ratings.user_id.unique()
    num_ratings_per_user = [len(ratings[ratings.user_id == x]) for x in uniq_users]
    plt.hist(num_ratings_per_user,bins=75)
    plt.savefig('step2.pdf')
    plt.close()'''

    #2) Remove non-frequent items - 
    #Non-frequent can be defined in absolute or relative values (5% or 10 items, respectively)

    if use_abs_rating_counts:
        rating_counts = rating_count_dict(ratings)

        def get_rcounts(x):
            return rating_counts[x]
        ratings = ratings[ratings['item_id'].apply(get_rcounts) >= min_ratings]
    else:
        freqs = rating_frequency_dict(ratings)
        
        def get_freq(x):
            #print(x)
            return freqs[x]       
        ratings = ratings[ratings['item_id'].apply(get_freq) >= min_freq]

    
    users_countings.append([len(ratings[ratings.user_id == x]) for x in uniq_users])
    

    ##TODO Salvar as matrizes com os relevantes e os irrelevantes       
    print("After removal of non-frequent items: {0}".format(len(ratings)))
    print_dataset_stats(ratings)
    process_stats.append(datailed_dataset_stats(ratings))

    user_counts = user_count_dict(ratings)

    def get_user_count(x):
        return user_counts[x]
    ratings = ratings[ratings['user_id'].apply(get_user_count) >= min_cnt]
    print("After removal of users with few ratings: {0}".format(len(ratings)))
    print_dataset_stats(ratings)
    process_stats.append(datailed_dataset_stats(ratings))

    users_countings.append([len(ratings[ratings.user_id == x]) for x in uniq_users])
    

    #Saving statistics
    
    
    stats_file = open(os.path.join(kwargs['output_dir'],'pre_process.stats'),'w')
    stats_file.write('Use abs ratings counts: %s\n' %(use_abs_rating_counts))

    if use_abs_rating_counts:
        stats_file.write('Item min ratings: %d\n' %(min_ratings))
    else:
        stats_file.write('Item min ratings perc: %d\n' %(min_freq))
        stats_file.write('Item min ratings total: %d\n' %(min_freq*len(uniq_users)))


    stats_file.write('Users min ratings: %d\n' %(min_cnt))
    

    stats_keys = ['num_users','num_items','sparsity','avg_ratings_per_user',
                    'max_ratings_per_user','min_ratings_per_user',
                    'avg_ratings_per_item','max_ratings_per_item',
                    'min_ratings_per_item']

        
    stats_file.write(';'.join(stats_keys)+'\n')
    for stat in process_stats:
        stats_file.write(';'.join([str(stat[x]) for x in stats_keys])+'\n')

    stats_file.close()


    plot_histograms(2,2,users_countings,**kwargs)



    return ratings



def plot_histograms(num_x_axis,num_y_axis,countings,normalize=False,**kwargs):
    
    f, axarr = plt.subplots(num_x_axis,num_y_axis)

    curr = 0    

    for i in range(num_x_axis):
        for j in range(num_y_axis):
            axarr[i,j].hist(countings[curr],bins=100,normed=normalize)
            axarr[i,j].set_title("Step %s" %(curr))
            curr += 1
    

    plt.savefig(os.path.join(kwargs['output_dir'],'pre_proc_hist_%s.pdf' %(normalize)))
    plt.close()

    if not normalize:
        plot_histograms(num_x_axis,num_y_axis,countings,normalize=True,**kwargs)
    
    



def shuffle_df(df):
    return df.reindex(np.random.permutation(df.index))


Fold = namedtuple('Fold', ['test', 'base'])
ValidationFold = namedtuple('ValidationFold', ['test', 'base', 'validation',
                                               'baseval'])


def make_folds(ratings, num_folds=5, validation=False,seed_value=123):

    
    np.random.seed(seed_value)

    grouped = ratings.groupby('user_id')
    split_groups = {user_id: np.array_split(group, min(num_folds, len(group)))
                    for user_id, group in grouped}

    folds = []
    for i in range(num_folds):
        test = []
        base = []
        valid = []
        baseval = []
        for split_group in split_groups.values():
            for j, split in enumerate(split_group):
                if i == j:
                    test.append(split)
                elif validation and i == (j + 1) % num_folds:
                    baseval.append(split)
                    valid.append(split)
                else:
                    baseval.append(split)
                    base.append(split)

        if validation:
            f = ValidationFold(pd.concat(test), pd.concat(base),
                               pd.concat(valid), pd.concat(baseval))
        else:
            f = Fold(pd.concat(test), pd.concat(base))
        folds.append(f)

    return folds



'''
Save the folds passed in the structure fold into train, test and validation 
files.

folds: The instances of each partition  (fold.train,fold.test,fold.validation)
dir_path: The path to save the files
validation: A boolean indication if the validation file will be used or not
seed_number: In the case of generanting the cross validation with more than one
seed the files are saved in different folders inside dir_path. Namely
dir_path/0
dir_path/1
dir_path/2
'''
def save_folds(folds, dir_path, validation,use_enum=False,seed_number=0):

    for i, fold in enumerate(folds):
        #SAMUEL------------------------------------------------------
        if use_enum:
            path = os.path.join(dir_path,str(seed_number), "u{0}.{1}")
            if not os.path.isdir(os.path.join(dir_path,str(seed_number))):
                os.makedirs(os.path.join(dir_path,str(seed_number)))

            reeval_dir = os.path.join(dir_path,str(seed_number), "reeval")  
            if not os.path.exists(reeval_dir):
                os.makedirs(reeval_dir)

        else:
            path = os.path.join(dir_path, "u{0}.{1}")
            reeval_dir = os.path.join(dir_path, "reeval")  
            if not os.path.exists(reeval_dir):
                os.makedirs(reeval_dir)
 


        if validation:
            ensure_same_users_items(fold.base,fold.validation,fold.test)
        #-------------------------------------------------------------        

        save_ratings_file(path.format(i + 1, "test"), fold.test)
        save_ratings_file(path.format(i + 1, "base"), fold.base)
        if validation:
            reeval_path = os.path.join(reeval_dir, "u{0}.{1}")
            save_ratings_file(path.format(i + 1, "validation"),
                              fold.validation)
            save_ratings_file(reeval_path.format(i + 1, "base"),
                              fold.baseval)
            save_ratings_file(reeval_path.format(i + 1, "test"),
                              fold.test)


def datailed_dataset_stats(ratings):
    """Returns the number of users, items and the sparsity of the dataset."""
    
    stats = {}
    stats['num_users'] = ratings.user_id.nunique()
    stats['num_items'] = ratings.item_id.nunique()
    max_possible_ratings = stats['num_users'] * stats['num_items']
    num_ratings = len(ratings.index)
    stats['sparsity'] = num_ratings / max_possible_ratings

    uniq_users = ratings.user_id.unique()
    num_ratings_per_user = [len(ratings[ratings.user_id == x]) for x in uniq_users]
    stats['avg_ratings_per_user'] = np.mean(num_ratings_per_user)
    stats['min_ratings_per_user'] = np.min(num_ratings_per_user)
    stats['max_ratings_per_user'] = np.max(num_ratings_per_user)

    uniq_items = ratings.item_id.unique()
    num_ratings_per_item = [len(ratings[ratings.item_id == x]) for x in uniq_items]

    stats['avg_ratings_per_item'] = np.mean(num_ratings_per_item)
    stats['min_ratings_per_item'] = np.min(num_ratings_per_item)
    stats['max_ratings_per_item'] = np.max(num_ratings_per_item)

    return stats


def dataset_stats(ratings):
    """Returns the number of users, items and the sparsity of the dataset."""
    num_users = ratings.user_id.nunique()
    num_items = ratings.item_id.nunique()
    max_possible_ratings = num_users * num_items
    num_ratings = len(ratings.index)
    sparsity = num_ratings / max_possible_ratings
    return num_users, num_items, sparsity


def print_dataset_stats(ratings):
    num_users, num_items, sparsity = dataset_stats(ratings)
    print("Number of users:", num_users)
    print("Number of items:", num_items)
    print("Rating matrix sparsity:", sparsity)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("data", type=str,
            help="path to the input, raw ratings file")
    p.add_argument("-o", "--output_dir", type=str, default="",
            help="output directory (default: same as input file)")
    p.add_argument("-p", "--proc_file", type=str, default="u.proc.data",
            help="name of the output file with all ratings after filtering")
    p.add_argument("-r", "--raw_file", type=str, default="u.raw.data",
            help="name of the raw file, an output file with all ratings, "
            "before any filtering (it is created because some use cases "
            "require such a file in default mml rating file format)")
    p.add_argument("-R", "--raw_only", action="store_true", default=False,
            help="if specified, only the raw file is generated (no "
            "pre-processing is performed)")
    p.add_argument("-f", "--folds", type=int, default=5,
            help="number of folds to be crea ted")
    p.add_argument("-n", "--no_validation", action='store_false',
            help="if specified, no validation folds are generated")
    p.add_argument("-c", "--config", type=str, default="",
            help="name of an override config file. (default: none)")
    #SAMUEl
    p.add_argument('--fmetric', type=str, default='median',
            help='metric used to filter ratings as relevant or not')
    p.add_argument('--seeds', type=str, default=None,
            help='file containg the seed that will be used to create the cross'   
             'validation files')
    return p.parse_args()


def main():
    args = parse_args()



    if args.output_dir == "":
        out_dir = args.data
    else:
        out_dir = args.output_dir

    

    conf = aux.load_configs(aux.CONF_DEFAULT, os.path.join(args.data,
                            aux.BASE_CONF), args.config)

    pp_conf = conf["pre_process"]

    this_settings = base_settings[pp_conf["type"]]
    read_function = this_settings['read_function']

    src_file = os.path.join(args.data, pp_conf["source"])

    #read files using the the specified reading function
    #there is a distinct function for each data type
    ratings = read_function(src_file, sep=pp_conf["separator"])


    #ratings eh um dataframe do pandas, nao eh necessario mostrar os stats
    print("Before filtering:")
    print_dataset_stats(ratings)

    if len(ratings) == 0:
        raise RuntimeError("Ratings matrix has zero ratings - likely read"
                           " error")

    if len(args.raw_file):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        save_ratings_file(os.path.join(out_dir, args.raw_file), ratings)
        if (args.raw_only):
            return

    #EDIT Samuel
    #Alterei os parametros passados para filter_ratings
    #ao inves de this parameters passo args pois vou usar as pastas de saida
    #para salvar estatisticas
    ratings = filter_ratings(ratings,use_abs_rating_counts=True, **vars(args))
    ratings = shuffle_df(ratings)

    print("After filtering:")
    print_dataset_stats(ratings)

    if len(args.proc_file):
        save_ratings_file(os.path.join(out_dir, args.proc_file), ratings)


    seeds = []
    if args.seeds != None:
        with open(args.seeds) as s:
            for line in s:
                seeds.append(int(line))

        print(seeds)
        for i,seed in enumerate(seeds):
            folds = make_folds(ratings, args.folds, args.no_validation,seed_value=seed)
            save_folds(folds, out_dir, args.no_validation,use_enum=True,seed_number=i)

    else:
        folds = make_folds(ratings, args.folds, args.no_validation)
        save_folds(folds, out_dir, args.no_validation)

   


if __name__ == "__main__":
    main()
