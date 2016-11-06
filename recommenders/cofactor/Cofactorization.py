
# coding: utf-8

# # Fit CoFactor model to the binarized ML20M

# In[87]:

import itertools
import glob
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


import bottleneck as bn
import pandas as pd
from scipy import sparse
import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')


# In[88]:

import cofacto
import rec_eval


# ### Construct the positive pairwise mutual information (PPMI) matrix

# Change this to wherever you saved the pre-processed data following [this notebook](./preprocess_ML20M.ipynb).

# In[89]:


def load_data(csv_file, n_users, n_items):
    shape=(n_users, n_items)
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['uid']), np.array(tp['sid'])
    relevance = np.ones((rows.size, 1), dtype='int')                      
    seq = np.concatenate((rows[:, None], cols[:, None],relevance ), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq



def _coord_batch(lo, hi, train_data,DATA_DIR,kwargs):
    rows = []
    cols = []
    for u in xrange(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
    np.save(os.path.join(DATA_DIR, kwargs.part+'-coo_%d_%d.npy' % (lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
    pass


def get_row(Y, i):
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return lo, hi, Y.data[lo:hi], Y.indices[lo:hi]



def save_predictions(train_data,test_data, Et, Eb, user_idx, unique_uid, unique_sid,DATA_DIR, k=100,
                        output_f = '-CoFactor.out', mu=None, vad_data=None):

    batch_users = user_idx.stop - user_idx.start
    X_pred = rec_eval._make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=mu,
                              vad_data=vad_data)
    
    #faz uma ordenacao parcial, ou seja, somente retorna os k maiores sem que 
    #eles estejam necessariamente ordenados entre si
    if '1.1.0' in bn.__version__:
        idx_topk_part = bn.argpartsort(-X_pred, k, axis=1) 
    else:
        idx_topk_part = bn.argpartition(-X_pred, k, axis=1) 
    #pega os valores dos indices retornados anteriormente
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]] 
    

    #retorna os indices ordenados a partir da pre selecao feita nas duas linhas anteriores
    idx_part = np.argsort(-topk_part, axis=1) 
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part] #armazena o indice real dos items


    #TODO salvar o valor das predicoes de cada item
    #TODO converter para os indices reais

    print output_f
    f_out = open(output_f,'w')

    aps = np.zeros(batch_users)
    for i, idx in enumerate(xrange(user_idx.start, user_idx.stop)):

        user_predictions = unique_uid[idx] + "\t["
        actual = test_data[idx].nonzero()[1]

        if len(actual) > 0:
            predicted = idx_topk[i]
            x = ','.join([unique_sid[x]+':'+str(X_pred[idx,x]) for x in predicted])
            user_predictions += x + ']\n'
            f_out.write(user_predictions)

    f_out.close()




def run(kwargs):

    
    DATA_DIR = kwargs.data
    partition = kwargs.part
    #DATA_DIR = './ml100k/pro/'
    #partition = 'u1'


    # In[90]:

    unique_uid = list()
    with open(os.path.join(DATA_DIR,kwargs.proc_folder, partition+'-unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
        
    unique_sid = list()
    with open(os.path.join(DATA_DIR,kwargs.proc_folder, partition+'-unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())


    # In[91]:

    n_items = len(unique_sid)
    n_users = len(unique_uid)

    print n_users, n_items

    train_data, train_raw = load_data(os.path.join(DATA_DIR,kwargs.proc_folder, partition+'-train.csv'),n_users,n_items)

    watches_per_movie = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
    print("The mean (median) watches per movie is %d (%d)" % (watches_per_movie.mean(), np.median(watches_per_movie)))


    user_activity = np.asarray(train_data.sum(axis=1)).ravel()

    print("The mean (median) movies each user wathced is %d (%d)" % (user_activity.mean(), np.median(user_activity)))

    if kwargs.no_validation:
        vad_data = None
    else:
        vad_data, vad_raw = load_data(os.path.join(DATA_DIR,'pro', partition+'-validation.csv'),n_users,n_items)



    plt.semilogx(1 + np.arange(n_users), -np.sort(-user_activity), 'o')
    plt.ylabel('Number of items that this user clicked on')
    plt.xlabel('User rank by number of consumed items')
    pass

    plt.semilogx(1 + np.arange(n_items), -np.sort(-watches_per_movie), 'o')
    plt.ylabel('Number of users who watched this movie')
    plt.xlabel('Movie rank by number of watches')
    pass


    from joblib import Parallel, delayed

    batch_size = 5000

    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]

    Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data,DATA_DIR,kwargs) for lo, hi in zip(start_idx, end_idx))
    pass


    X = sparse.csr_matrix((n_items, n_items), dtype='float32')

    for lo, hi in zip(start_idx, end_idx):
        coords = np.load(os.path.join(DATA_DIR, kwargs.part+'-coo_%d_%d.npy' % (lo, hi)))
        
        rows = coords[:, 0]
        cols = coords[:, 1]
        
        tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_items, n_items), dtype='float32').tocsr()
        X = X + tmp
        
        print("User %d to %d finished" % (lo, hi))
        sys.stdout.flush()

    np.save(os.path.join(DATA_DIR, kwargs.part+'-coordinate_co_binary_data.npy'), X.data)
    np.save(os.path.join(DATA_DIR, kwargs.part+'-coordinate_co_binary_indices.npy'), X.indices)
    np.save(os.path.join(DATA_DIR, kwargs.part+'-coordinate_co_binary_indptr.npy'), X.indptr)


    float(X.nnz) / np.prod(X.shape)


    # ### Or load the pre-saved co-occurrence matrix
    # or co-occurrence matrix from the entire user history
    dir_predix = DATA_DIR

    data = np.load(os.path.join(dir_predix, kwargs.part+'-coordinate_co_binary_data.npy'))
    indices = np.load(os.path.join(dir_predix, kwargs.part+'-coordinate_co_binary_indices.npy'))
    indptr = np.load(os.path.join(dir_predix, kwargs.part+'-coordinate_co_binary_indptr.npy'))

    X = sparse.csr_matrix((data, indices, indptr), shape=(n_items, n_items))

    float(X.nnz) / np.prod(X.shape)

    count = np.asarray(X.sum(axis=1)).ravel()

    n_pairs = X.data.sum()


    # ### Construct the SPPMI matrix

    M = X.copy()

    for i in xrange(n_items):
        lo, hi, d, idx = get_row(M, i)
        M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))

    M.data[M.data < 0] = 0
    M.eliminate_zeros()
    print float(M.nnz) / np.prod(M.shape)


    # Now $M$ is the PPMI matrix. Depending on the number of negative examples $k$, we can obtain the shifted PPMI matrix as $\max(M_{wc} - \log k, 0)$

    # number of negative samples
    k_ns = 1

    M_ns = M.copy()

    if k_ns > 1:
        offset = np.log(k_ns)
    else:
        offset = 0.
        
    M_ns.data -= offset
    M_ns.data[M_ns.data < 0] = 0
    M_ns.eliminate_zeros()

    plt.hist(M_ns.data, bins=50)
    plt.yscale('log')
    pass

    float(M_ns.nnz) / np.prod(M_ns.shape)


    # ### Train the model
    scale = 0.03

    n_components = 100
    max_iter = 20
    n_jobs = 8
    lam_theta = lam_beta = 1e-5 * scale
    lam_gamma = 1e-5
    c0 = 1. * scale
    c1 = 10. * scale

    save_dir = os.path.join(DATA_DIR,kwargs.proc_folder, partition+'-_ns%d_scale%1.2E' % (k_ns, scale))

    reload(cofacto)
    coder = cofacto.CoFacto(n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=n_jobs, 
                            random_state=98765, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True, 
                            lam_theta=lam_theta, lam_beta=lam_beta, lam_gamma=lam_gamma, c0=c0, c1=c1)


    coder.fit(train_data, M_ns, vad_data=vad_data, batch_users=5000, k=100)

    test_data, _ = load_data(os.path.join(DATA_DIR,'pro', partition+'-test.csv'),n_users,n_items)
    test_data.data = np.ones_like(test_data.data)

    n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))

    params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
    U, V = params['U'], params['V']


    
    print 'Test Recall@20: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=20, vad_data=vad_data)
    print 'Test Recall@50: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=50, vad_data=vad_data)
    print 'Test NDCG@100: %.4f' % rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data)
    print 'Test MAP@100: %.4f' % rec_eval.map_at_k(train_data, test_data, U, V, k=10, vad_data=vad_data)

    save_predictions(train_data,test_data, U, V.T, slice(0,len(unique_uid),None),unique_uid, unique_sid,output_f = os.path.join(kwargs.output_dir,kwargs.part+'-CoFactor.out'),DATA_DIR=DATA_DIR)
    cmd = "rm {0}*.npy".format(os.path.join(dir_predix,kwargs.part))
    os.system(cmd)

    # In[134]:

    #np.savez('CoFactor_K100_ML20M.npz', U=U, V=V)
    

if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    partition = sys.argv[2]
    run(DATA_DIR,partition)

