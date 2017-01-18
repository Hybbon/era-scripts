import sys
import os
import argparse
import random
import multiprocessing as mp
#deal with the Attribute error : __exit__ when using mp.Pool  as pool
from contextlib import closing 


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
    p.add_argument("--binarize", action="store_true",
            help="Use this argument when there is a need to use binary data in the libfm input. Possible this option is useful when you want libfm to read data faster or if the dataset does not fit in memory. In the last case you also need to set the parameter --cache_size")
    p.add_argument("--cache_size",type=int, default=10000000000,
            help="The max size used by each of the input files in the binary format. This is the size for the train, test and transposed data. The size is given in Bytes and the default corresponds to 10GB (10000000000)")

    p.add_argument("--method",type=str, default="mcmc", 
            help="model to be used in libfm, options mcmc,als,sgd")

    p.add_argument("--save_model",action="store_true")


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

Since the libfm works as a Rating Prediction, or even a regression, algorithm
it needs examples of all 'classes'.
Therefore, these items are inserted in the training to be used as negative examples (we 
attribute them a rating 0). 
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



def inflate_test_for_prediction_savemem(train_data,item_list,filename):    

    new_test = []
    users_ids = sorted(train_data.keys())
    for usr_idx,user in enumerate(users_ids):
        new_test.append([])
        for item,rating in item_list:
            is_in_train = len([(item_u,rating_u) for item_u,rating_u in train_data[user] if item_u == item])
            if is_in_train == 0:
                new_test[-1].append((item,rating))
    
    return new_test,users_ids




'''
Creates a test file containing all items for each user. This new test will be
used to construct a ranking for the users.

This version uses a list instead of a dict to save the new_test
'''
def inflate_andsave_test_for_prediction(train_data,item_list,filename):    

    #new_test = []
    out = open(filename,'w')    
    users_ids = sorted(train_data.keys())
    for usr_idx,user in enumerate(users_ids):
        #new_test.append([])
        items_to_save = ''
        for item,rating in item_list:
            is_in_train = len([(item_u,rating_u) for item_u,rating_u in train_data[user] if item_u == item])
            if is_in_train == 0:
                items_to_save += '{0}\t{1}\t{2}\n'.format(user,item,rating)
                #new_test[-1].append((item,rating))
    
        out.write(items_to_save)
    #return new_test,users_ids
    

'''
Create rankings without using the test file inflated in the past steps
This function just uses the list of all items and the items already rated by 
the users in the training file

'''
def create_rankings(original_train,item_list,predictions_f,outfolder,out_name,usrs_ids,rank_size=10):
    
    #load file with libfm predictions
    pred_f = open(predictions_f,'r')
    rankings_f = open(os.path.join(outfolder,out_name),'w')     
    
    for usr_idx,user in enumerate(usrs_ids):#test_data.keys():

        preds = []
        #load items that was already rated by the user in the training file
        #this loading do not consider the items that where inflated in the 
        #training files (items rated with 0)
        items_already_rated = [x for x,y in original_train[user] if y != 0]

        for item_idx,item_rat in enumerate(item_list):        
        #for item_id,_ in test_data[user]:
            item_id = item_rat[0]
            #verify if the item was not ratted in the training and therefore
            #loads the rating
            if not item_id in items_already_rated:                
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

'''def create_rankings_deprecated(test_data,predictions_f,outfolder,out_name,usrs_ids,rank_size=10):


    pred_f = open(predictions_f,'r')
    rankings_f = open(os.path.join(outfolder,out_name),'w') 

   
    for usr_idx,user in enumerate(usrs_ids):#test_data.keys():
        num_items_test = len(test_data[user])
        preds = []
        
        for item_id,_ in test_data[user]:            
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
'''    

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

    print args

    #TODO 
    #TODO Deveria pegar o arquivo de validacao e juntar com o de teste
    #TODO
    train_name = args.part+'.base'
    test_name = args.part+'.test'


    run_dir = "./recommenders/libfm/"
    #run_dir = "./"
    train = os.path.join(datadir,train_name)
    test = os.path.join(datadir,test_name)           
    #test_data = read_movilens_format(test) #args.test
    train_data = read_movilens_format(train) #args.test
    item_list = get_item_list(train_data)

    if not os.path.isdir(os.path.join(datadir,'tmp_libfm')):
        os.mkdir(os.path.join(datadir,'tmp_libfm'))


    test_libfm = os.path.join(datadir,'tmp_libfm',test_name + '_inflated')
    test_libfm_uninflated = os.path.join(datadir,'tmp_libfm',test_name)
    inflate_andsave_test_for_prediction(train_data,item_list,test_libfm)
    inflate_train(train_data,item_list)
   
    train_libfm = os.path.join(datadir,'tmp_libfm',train_name + '_inflated')

    save_ratings_movielens_format(train_data,train_libfm)        


    os.system("cp "+os.path.join(datadir,test_name)+" "+ test_libfm_uninflated)


    if not (os.path.isfile(train_libfm+'.libfm') and
       os.path.isfile(test_libfm+'.libfm') and 
       os.path.isfile(test_libfm_uninflated+'.libfm')):
    
        os.system(run_dir+'scripts/triple_format_to_libfm.pl -in {0},{1} -target 2 '.format(train_libfm,test_libfm_uninflated)+
             '-separator "\t"')
        os.system(run_dir+'scripts/triple_format_to_libfm.pl -in {0},{1} -target 2 '.format(train_libfm,test_libfm)+
             '-separator "\t"')



    train_libfm += ".libfm"
    test_libfm += ".libfm"
    test_libfm_uninflated += '.libfm'

    if args.binarize:        
        #convertendo os arquivos para o formato binario do libfm
        print "Binarizando"

        conv_cmd = run_dir+"bin/convert --ifile {0} --ofilex {1}_bin.x --ofiley {2}_bin.y"
        all_conv_cmd = [conv_cmd.format(train_libfm,train_libfm,train_libfm),
                        conv_cmd.format(test_libfm,test_libfm,test_libfm)]
        with closing(mp.Pool(processes=2)) as pool:
            pool.map(os.system,all_conv_cmd)
            pool.terminate()
        #os.system(run_dir+"bin/convert --ifile {0} --ofilex {1}_bin.x --ofiley {2}_bin.y".format(train_libfm,train_libfm,train_libfm))
        #os.system(run_dir+"bin/convert --ifile {0} --ofilex {1}_bin.x --ofiley {2}_bin.y".format(test_libfm,test_libfm,test_libfm))        
        train_libfm += "_bin"
        test_libfm += "_bin"

        print "Transpondo"
        #quando utilizamos o formato binario eh necessario realizar a transposicao do arquivos com os artibutos (.x)

        transp_cmd = run_dir+"bin/transpose --ifile {0}.x --ofile {1}.xt"
        all_transp_cmd = [transp_cmd.format(train_libfm,train_libfm,train_libfm),
                        transp_cmd.format(test_libfm,test_libfm,test_libfm)]

        with closing(mp.Pool(processes=2)) as pool:
            pool.map(os.system,all_transp_cmd)
            pool.terminate()

        #os.system(run_dir+"bin/transpose --ifile {0}.x --ofile {1}.xt".format(train_libfm,train_libfm))
        #os.system(run_dir+"bin/transpose --ifile {0}.x --ofile {1}.xt".format(test_libfm,test_libfm))


    print "TES1"
    output_f = os.path.join(datadir,'tmp_libfm',args.part+'_predictions.dat')

    run_cmd = run_dir+"bin/libFM -task r -iter 50 -train "
    if args.save_model and not 'mcmc' in args.method:

        model_path = os.path.join(datadir,'tmp_libfm',args.part+"-model_"+args.method)

        run_cmd += "{0} -test {1} -dim '1,1,8' -method {2} -regular '1,1,1' -save_model {3} ".format(train_libfm,test_libfm_uninflated,args.method,model_path)
        run_cmd += "-out {0}.temp".format(output_f)
    else:
        run_cmd += "{0} -test {1} -dim '1,1,8' ".format(train_libfm,test_libfm)
        run_cmd += "-out {0}".format(output_f)

    if args.binarize:
        run_cmd += " --cache_size {0}".format(args.cache_size)

    print run_cmd

    os.system(run_cmd)
        

    if args.save_model and not 'mcmc' in args.method:
        run_cmd = run_dir+"bin/libFM -task r -iter 1 -train "
        run_cmd += "{0} -test {1} -dim '1,1,8' -method {2} -regular '1,1,1' -load_model {3} ".format(train_libfm,test_libfm,args.method,model_path)
        run_cmd += "-out {0}".format(output_f)
    
        os.system(run_cmd)


    print "TES2"
    #new_test = read_movilens_format(test_libfm)
    usrs_ids = sorted(train_data.keys())


    #create_rankings(new_test,output_f,datadir,args.part+'-libfm.out',usrs_ids = usrs_ids, rank_size=args.num_items)
    create_rankings(train_data,item_list,output_f,datadir,args.part+'-libfm.out',usrs_ids = usrs_ids, rank_size=args.num_items)
    #os.system('rm '+datadir+'tmp_libfm/'+args.part+'*')





