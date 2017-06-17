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
import ipdb

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
e a segunda contem os ratings considerados nao relevantes. A uniao das duas 
matrizes forma a matriz original

1)Alterar para passar a metrica desejada, atualmente estou usando mediana

'''
def personalized_rating_normalization(ratings,metric='median'):

    uniq_users = ratings.user_id.unique()
    ratings['to_remove'] = pd.Series(np.zeros(len(ratings)),index=ratings.index)

    ini1 = time.time()

    if metric == 'median':
        user_medians = ratings.groupby(['user_id'])['rating'].median()
        for user in uniq_users:
            ratings.loc[(ratings.user_id == user) & ((ratings.rating < user_medians[user]) | (ratings.rating <= 0)),'to_remove'] = 1
        print('median time:' + str(time.time()-ini1))
        return ratings[ratings['to_remove'] != 1], ratings[ratings['to_remove'] == 1]
    elif 'fixed':
        print('fixed time' + str(time.time()-ini1))
        return ratings[ratings['rating'] >= 4], ratings
       
'''
This funtion ensures that the same users and itens used in the train files 
will be present in the validation and test.
The changes are not done inplace. The resultant dataframes are returned to the 
caller, that is responsible to update the corresponding data

Output: 
train, validation and test dataframes
boolean indicating if some modification was done in the DFs

'''
def ensure_same_users_items(train_data,val_data,test_data):

    ini = time.time()    

    unique_items_train = set(train_data.item_id.unique())

    if  val_data is not None:
        print("item train {0} items val {1}".format(len(train_data.item_id.unique()),len(val_data.item_id.unique())))
                
        #Change the files in a way that the training file contains all possible items
        unique_items = unique_items_train.union(val_data.item_id.unique())
        left_sid = list(unique_items - unique_items_train)
        print(left_sid)
        items_not_in_train = val_data['item_id'].isin(left_sid)        
        train_data = train_data.append(val_data[items_not_in_train])
        val_data = val_data[~items_not_in_train]

        print("len items test : "+str(len(test_data.item_id.unique())))
        test_data = test_data[test_data['item_id'].isin(unique_items)]
        print("len items test : "+str(len(test_data.item_id.unique())))


        print("unique items time: "+ str(time.time()-ini))

    else:

        unique_items = unique_items_train.union(test_data.item_id.unique())
        left_sid = list(unique_items - unique_items_train)
        print(left_sid)
        items_not_in_train = test_data['item_id'].isin(left_sid)        
        train_data = train_data.append(test_data[items_not_in_train])
        print("len items test : "+str(len(test_data.item_id.unique())))
        test_data = test_data[~items_not_in_train]
        print("len items test : "+str(len(test_data.item_id.unique())))

        '''print("len items test : "+str(len(test_data.item_id.unique())))
        test_data = test_data[test_data['item_id'].isin(unique_items)]
        print("len items test : "+str(len(test_data.item_id.unique())))


        print("unique items time: "+ str(time.time()-ini))'''



    if len(left_sid) > 0:
        return train_data,val_data,test_data,True
    else:
        return train_data,val_data,test_data,False





def filter_ratings(ratings, min_freq=0.05, min_rating=4, min_cnt=10,
                   min_ratings=10, use_abs_rating_counts=False, **kwargs):
    """Removes non-frequent movies or ratings under a certain score."""


    process_stats = []
    users_countings = []
    items_countings = []

    stats_file = open(os.path.join(kwargs['output_dir'],'pre_process.stats'),'w')
    stats_file.write('Users min ratings: %d\n' %(min_cnt))
    

    # Converts all ratings to integers
    ratings['rating'] = ratings['rating'].apply(lambda x: int(x))
    print("Initial length: {0}".format(len(ratings)))
    
    uniq_users = ratings.user_id.unique()
    if not kwargs['silent']:
        process_stats.append(detailed_dataset_stats(ratings))
    else:
        stats_file.write('Num initial users: %d\n' %(len(uniq_users)))
        stats_file.write('Num initial items: %d\n' %(len(ratings.item_id.unique())))



    users_countings.append([len(ratings[ratings.user_id == x]) for x in uniq_users])   
    '''plt.hist(num_ratings_per_user,bins=75)
    print(a)   
    plt.savefig('step1.pdf')
    plt.close()'''
    
    #when this argument is set we do not remove the ratings lower than a 
    #threshold. We use this when the pre processed dataset will be used to 
    #rating prediction instead of ranking prediction
    if not kwargs['rating_pred']:
        #1) Remove ratings lower than the median of the user's ratings
        ratings, irrel_ratings = personalized_rating_normalization(ratings,kwargs['fmetric'])
        
        #ratings = ratings[ratings['rating'] >= min_rating]
        print("After removal of ratings under {1}: {0}".format(len(ratings),
                                                               min_rating))
        process_stats.append(detailed_dataset_stats(ratings))
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
    if not kwargs['silent']:
        print_dataset_stats(ratings)
        process_stats.append(detailed_dataset_stats(ratings))

    user_counts = user_count_dict(ratings)

    def get_user_count(x):
        return user_counts[x]
    ratings = ratings[ratings['user_id'].apply(get_user_count) >= min_cnt]
    print("After removal of users with few ratings: {0}".format(len(ratings)))
    print_dataset_stats(ratings)
    process_stats.append(detailed_dataset_stats(ratings))

    users_countings.append([len(ratings[ratings.user_id == x]) for x in uniq_users])
    

    #Saving statistics
    
    stats_file.write('Use abs ratings counts: %s\n' %(use_abs_rating_counts))

    if use_abs_rating_counts:
        stats_file.write('Item min ratings: %d\n' %(min_ratings))
    else:
        stats_file.write('Item min ratings perc: %d\n' %(min_freq))
        stats_file.write('Item min ratings total: %d\n' %(min_freq*len(uniq_users)))

    
    
    
    if not kwargs['silent']:
        stats_keys = ['num_users','num_items','sparsity','avg_ratings_per_user',
                        'max_ratings_per_user','min_ratings_per_user',
                        'avg_ratings_per_item','max_ratings_per_item',
                        'min_ratings_per_item']

            
        stats_file.write(';'.join(stats_keys)+'\n')
        for stat in process_stats:
            stats_file.write(';'.join([str(stat[x]) for x in stats_keys])+'\n')
    else:
        num_users, num_items, sparsity = dataset_stats(ratings)
        stats_file.write("Number of users: %d \n" %(num_users))
        stats_file.write("Number of items: %d \n" %(num_items))
        stats_file.write("Rating matrix sparsity: %f \n" %(sparsity))        

    stats_file.close()


    #plot_histograms(2,2,users_countings,**kwargs)



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
            #TODO Samuel - Testar o que é a saida do groupby e pd.concat
            basex,validx,testx,need_save = ensure_same_users_items(pd.concat(base),pd.concat(valid),pd.concat(test))
            basevalx,_,testx,need_save = ensure_same_users_items(pd.concat(baseval),None,testx)
            f = ValidationFold(testx, basex,
                               validx, basevalx)

            #f = ValidationFold(pd.concat(test), pd.concat(base),
            #                   pd.concat(valid), pd.concat(baseval))
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
 


        '''if validation:
            #fold.base,fold.validation,fold.test, need_save = ensure_same_users_items(fold.base,fold.validation,fold.test)
            xbase,xvalidation,xtest, need_save = ensure_same_users_items(fold.base,fold.validation,fold.test)
        else:
            fold.base,fold.validation,fold.test, need_save = ensure_same_users_items(fold.base,None,fold.test)'''

        #-------------------------------------------------------------        

        save_ratings_file(path.format(i + 1, "test"), fold.test)
        save_ratings_file(path.format(i + 1, "base"), fold.base)
        #ipdb.set_trace()
        inflate_andsave_test_for_prediction(fold.base,[],path.format(i + 1, "test_inflated"))

        if validation:
            reeval_path = os.path.join(reeval_dir, "u{0}.{1}")
            save_ratings_file(path.format(i + 1, "validation"),
                              fold.validation)
            save_ratings_file(reeval_path.format(i + 1, "base"),
                              fold.baseval)
            save_ratings_file(reeval_path.format(i + 1, "test"),
                              fold.test)
            inflate_andsave_test_for_prediction(fold.baseval,[],
                reeval_path.format(i + 1, "test_inflated"))



#*****************************RATING PREDICTION*********************************






'''
Creates a test file containing all items for each user (inflates the test file
with the items in the train file). This new test will be used to construct a 
ranking for the users.
This version uses a list instead of a dict to save the new_test
'''
def inflate_andsave_test_for_prediction(train_data,item_list,filename):    

    #new_test = []
    out = open(filename,'w')
    #ipdb.set_trace()
    users_ids = sorted(train_data.user_id.unique())    
    item_list = train_data.item_id.unique()
    #users_ids = sorted(train_data.keys())
    
    for usr_idx,user in enumerate(users_ids):
        #new_test.append([])
        items_to_save = ''
        for item in item_list:
            #ipdb.set_trace()
            if len(train_data[(train_data.user_id==user) & 
                ( train_data.item_id==item)]) == 0:
                items_to_save += '{0}\t{1}\t{2}\n'.format(user,item,0)
                #new_test[-1].append((item,rating))
    
        out.write(items_to_save)
    #return new_test,users_ids


'''
Given a dataset (a rating matrix in movielens format) returns
'''
def get_item_list(dataset):
    itemset = set()
    for user in dataset.keys():
        user_items = dataset[user]
        for item,rating in user_items:            
            itemset.add((item,0)) #add all items with rating 0

    #just convert to a list, allowing indexing
    item_list = [x for x in itemset]

    return item_list





#**************************END RATING PREDICTION********************************







def detailed_dataset_stats(ratings):
    """Returns the number of users, items and the sparsity of the dataset."""
    
    stats = {}
    stats['num_users'] = ratings.user_id.nunique()
    stats['num_items'] = ratings.item_id.nunique()
    max_possible_ratings = stats['num_users'] * stats['num_items']
    num_ratings = len(ratings.index)
    stats['sparsity'] = num_ratings / max_possible_ratings

    uniq_users = ratings.user_id.unique()
    num_ratings_per_user = [len(ratings[ratings.user_id == x]) for x in uniq_users]
    stats['quartiles_user'] = [np.percentile(num_ratings_per_user,x) for x in [25,50,75]]
    stats['std_ratings_per_user'] = np.std(num_ratings_per_user)
    stats['avg_ratings_per_user'] = np.mean(num_ratings_per_user)
    stats['min_ratings_per_user'] = np.min(num_ratings_per_user)
    stats['max_ratings_per_user'] = np.max(num_ratings_per_user)

    uniq_items = ratings.item_id.unique()
    num_ratings_per_item = [len(ratings[ratings.item_id == x]) for x in uniq_items]

    stats['quartiles_item'] = [np.percentile(num_ratings_per_item,x) for x in [25,50,75]]
    stats['std_ratings_per_item'] = np.std(num_ratings_per_item)
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
    p.add_argument('-s','--silent', action='store_true',
            help = 'Do not generate statistics')
    p.add_argument('--rating_pred',action='store_true',
            help='Set this parameter when you want to construct datasets for ' 
                  'rating predction. This pre-process do not remove ratings lower '
                  'than a specified threshold')
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
