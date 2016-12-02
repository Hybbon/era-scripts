import sys
import os
import argparse
import random


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("data", type=str,
        help="folder containing the config file and the"
        " ranking data to be evaluated.")
    p.add_argument("-p", "--part", type=str,
        help="Partition")
    p.add_argument('-n','--num_items',type=int, default=100,
        help='number of items that will be recommended')
    p.add_argument("--train", type=str,
            help="path to the training file, movielens format")
    p.add_argument("--val", type=str,default='',
            help="path to the validation file, movielens format")
    p.add_argument("--test", type=str,
            help="path to the test file, movielens format")    

    return p.parse_args()

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


'''
Sample items that are not rated by the user as negative examples and 
inflate the training data with them
'''
def inflate_train(train_data,item_list):

    for user in train_data.keys():
        num_items_user = len(train_data[user])
        for i in range(num_items_user):
            to_insert = item_list[random.randint(0,len(item_list)-1)]
            #garante que o item escolhido nao esta entre os items ja presentes
            #no treinamento        
            while to_insert in train_data[user]:
                to_insert = item_list[random.randint(0,len(item_list)-1)]
            train_data[user].append(to_insert)



'''
Creates a test file containing all items for each user. This new test will be
used to construct a ranking for the users.

DEPRECATED
'''
#TODO remover o parametro test_data, uma vez que o novo test vai conter todos os
#items, exceto os que estao no treino, nao eh necessario ter acesso ao teste 
#original
def inflate_test_for_prediction(train_data,item_list):
    
    new_test = {}
    for user in train_data.keys():
        new_test[user] = []
        for item,rating in item_list:
            is_in_train = len([(item_u,rating_u) for item_u,rating_u in train_data[user] if item_u == item])
            if is_in_train == 0:
                new_test[user].append((item,rating))
    
    return new_test

'''
Creates a test file containing all items for each user. This new test will be
used to construct a ranking for the users.

This version uses a list instead of a dict to save the new_test
'''
def inflate_test_for_prediction_savemem(train_data,item_list):
    
    new_test = []
    users_ids = sorted(train_data.keys())
    for usr_idx,user in enumerate(users_ids):
        new_test.append([])
        for item,rating in item_list:
            is_in_train = len([(item_u,rating_u) for item_u,rating_u in train_data[user] if item_u == item])
            if is_in_train == 0:
                new_test[-1].append((item,rating))
    
    return new_test,users_ids


    

def create_rankings(test_data,predictions_f,outfolder,out_name,usrs_ids,rank_size=10):


    pred_f = open(predictions_f,'r')

    rankings_f = open(os.path.join(outfolder,out_name),'w') 

    

    for usr_idx,user in enumerate(usrs_ids):#test_data.keys():
        num_items_test = len(test_data[usr_idx])
        preds = []
        
        for item_id,_ in test_data[usr_idx]:
            
            pred_value = pred_f.readline().strip()
            pred_value = float(pred_value)
            preds.append((item_id,pred_value))

        preds.sort(key=lambda tup : tup[1], reverse=True)
        
        s = '{0}\t['.format(user)
        s += ','.join([str(x)+':'+str(y) for x,y in preds[:rank_size]])
        s += ']\n'


        rankings_f.write(s)

    pred_f.close()
    rankings_f.close()
    

def read_movilens_format(rating_file,sep='\t'):


    data = open(rating_file,'r')
    
    tokens = data.readline().strip().split(sep)
    past_usr,past_mov,past_rat = tokens[:3]
    line_usr = past_mov

    users = {int(past_usr):[(int(past_mov),float(past_rat))]}


    nusers = 1;
    for line in data:
        tokens = line.strip().split(sep)
        usr,mov,rat = tokens[:3]
        
        if usr == past_usr:

            users[int(usr)].append((int(mov),float(rat)))
            #line_usr += ' '+mov
            past_usr,past_mov = usr,mov
        else:
            users[int(usr)] = [(int(mov),float(rat))]
            nusers += 1
            past_usr,past_mov = usr,mov

    return users


def save_ratings_movielens_format(dataset,filename,usrs_ids=[]):
    
    out = open(filename,'w')

    using_dict = False
    if len(usrs_ids) == 0:
        usrs_ids = dataset.keys()
        using_dict = True        

    for usr_idx,user in enumerate(usrs_ids):
        if using_dict:
            for item,rating in dataset[user]:
                s = '{0}\t{1}\t{2}\n'.format(user,item,rating)
                out.write(s)
        else:
            for item,rating in dataset[usr_idx]:
                s = '{0}\t{1}\t{2}\n'.format(user,item,rating)
                out.write(s)


    out.close()

if __name__ == '__main__':
    args = parse_args()    
        
    datadir = args.data

    train_name = args.part+'.base'
    test_name = args.part+'.test'

    train = os.path.join(datadir,train_name)
    test = os.path.join(datadir,test_name)           
    #test_data = read_movilens_format(test) #args.test
    train_data = read_movilens_format(train) #args.test
    item_list = get_item_list(train_data)

    new_test,usrs_ids = inflate_test_for_prediction_savemem(train_data,item_list)
    inflate_train(train_data,item_list)

    #train_libfm = args.train.split('/')[-1] + '_inflated'
    #test_libfm = args.test.split('/')[-1] + '_inflated'
    if not os.path.isdir(os.path.join(datadir,'tmp_libfm')):
        os.mkdir(os.path.join(datadir,'tmp_libfm'))

    train_libfm = os.path.join(datadir,'tmp_libfm',train_name + '_inflated')
    test_libfm = os.path.join(datadir,'tmp_libfm',test_name + '_inflated')
    save_ratings_movielens_format(train_data,train_libfm)
    save_ratings_movielens_format(new_test,test_libfm,usrs_ids)
    os.system('./recommenders/libfm/scripts/triple_format_to_libfm.pl -in {0},{1} -target 2 '.format(train_libfm,test_libfm)+
                 '-separator "\t"')

    output_f = os.path.join(datadir,'tmp_libfm',args.part+'_predictions.dat')

    os.system("./recommenders/libfm/bin/libFM -task r -train {0}.libfm -test {1}.libfm -dim '1,1,8' ".format(train_libfm,test_libfm) +
                "-out {0}".format(output_f))

    

    create_rankings(new_test,output_f,datadir,args.part+'-libfm.out',usrs_ids = usrs_ids, rank_size=args.num_items)

    os.system('rm '+datadir+'tmp_libfm/'+args.part+'*')





