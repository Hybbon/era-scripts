"""" This script receives two input files
1) A file in a classification dataset format 
<att0;att1;att2;....attk,class> 
where each line represents the efatures extracted from an item recommended to a 
user and the relevance of this item to the user considered.
2) A map file mapping where each line represents an user and the items 
recommended to this user in the following format:

user_id:(item0_pos,item0_id);(item1_pos,item1_id);....(itemN_pos,itemN_id)

ie.: Each pair(itemK_pos,itemK_id) indicates that the item itemK_id indicated 
to the user <user_id> is placed in the <itemK_pos> line of the classification 
datased 
"""


import os
import sys
import argparse



def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("-data_folder",type=str,default="",
        help="Classfication dataset used to construct the dataset in LETOR format")
    p.add_argument("-o","--out_dir",type=str,default="",
        help="The folder where the output datasets will be saved")
    p.add_argument("-p","--part",type=str,default="u1")
    p.add_argument("-test",action="store_true",
        help="Construct the test dataset")
    return p.parse_args()


def do_conversion(args):


    map_path = os.path.join(args.data_folder,args.part+".train.map")
    
    if not args.test:
        data_path = os.path.join(args.data_folder,args.part+".train_logit")
        out_path = os.path.join(args.data_folder,args.part+".train.letor") 
    else:
        data_path = os.path.join(args.data_folder,args.part+".plain.train")        
        out_path = os.path.join(args.data_folder,args.part+".train.letor") 


    with open(map_path,'r') as map_file, open(out_path,'w') as out_file:
        #with data_file as open(data_path,'r'):
        data_file = open(data_path,'r')
        for map_line in map_file:
            user_id,map_items = map_line.strip().split(':') #get user id

            map_items = map_items.replace("(","").replace(")","").split(";") #break the line in the item mapping

            for item in map_items: #each pair (pos,item) are correlated to one line of data_file 

                _,item_id = item.split(',')#gets the item_id, the position is irrelevant in this case 
                data_line = data_file.readline() #reads the line which represents the item
                data_line = data_line.strip()
                
                #Verify if we are constructing train or test datasets
                #in case of test datasets, they are constructed using the data in the reeval folders
                if not args.test:
                    #get the item's attributes 
                    atts = data_line.split(';')[:-1] 
                    #gets the class (relevant or irrelavant), that in the original data_file is in the end of the line
                    new_line = data_line[-1]+" qid:"+user_id + " "
                else:
                    atts = data_line.split(';')
                    #gets the class (relevant or irrelavant), that in the original data_file is in the end of the line
                    new_line = "0"+" qid:"+user_id + " "

                #put the attributes in the LETOR format
                for i,att in enumerate(atts):
                    new_line += str(i+1)+":"+att+" "                    
                new_line += "#docid = "+item_id+" inc = 0.0 prob = 0.0\n"
                out_file.write(new_line)
                #print(new_line)
                #a = input()
        
    data_file.close()


if __name__ == "__main__":

    args = parse_args()
    do_conversion(args)





