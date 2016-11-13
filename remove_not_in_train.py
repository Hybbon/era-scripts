import sys
import os
import argparse
import time
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("data", type=str,
            help="path to the input, raw ratings file")
    p.add_argument("-o", "--output_dir", type=str, default="",
            help="output directory (default: same as input file)")
    return p.parse_args()





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

    if len(left_sid) > 0:
        return train_data,val_data,test_data,True
    else:
        return train_data,val_data,test_data,False



if __name__ == "__main__":
    
    args = parse_args()
    
    header = ("user_id","item_id","rating")
    types = {"user_id": np.int32, "item_id" : np.int32, "rating" : np.int32}
    for p in range(1,2):
        part = 'u'+str(p)
        train = pd.read_csv(os.path.join(args.data,part+'.base'),sep="\t",names=header, dtype=types)
        test = pd.read_csv(os.path.join(args.data,part+'.test'),sep="\t",names=header, dtype=types)
        
        validation = pd.read_csv(os.path.join(args.data,part+'.validation'),sep="\t",names=header, dtype=types)

        train,validation,test,need_save =  ensure_same_users_items(train,validation,test)


        if need_save:
            print("Need to save the changes")
            train.to_csv(os.path.join(args.data,part+'.base'), header=False, index=False, sep="\t", columns=header)
            validation.to_csv(os.path.join(args.data,part+'.validation'), header=False, index=False, sep="\t", columns=header)
            test.to_csv(os.path.join(args.data,part+'.test'), header=False, index=False, sep="\t", columns=header)    
        


