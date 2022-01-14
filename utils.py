import random
import os
import numpy as np
import torch
import torch.utils.data as data
import dataset
import sklearn.preprocessing as pp
import logging

# from sklearn.cluster import KMeans
# from torch.autograd import Variable


def get_dataset_name(path):                                                                                                
    l,r = os.path.split(path)                                                                                              
    l,r = os.path.splitext(r)                                                                                              
    return l               


def zero2eps(x):
    x[x == 0] = 1
    return x


def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty


def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def eval_cos(data_1: np.ndarray, data_2: np.ndarray)->np.ndarray:

    pro_data_1 = pp.normalize(data_1, norm='l2') # to calculate the cos-similarity
    pro_data_2 = pp.normalize(data_2, norm='l2')

    affinity = np.mean(np.sum(np.multiply(pro_data_1, pro_data_2), axis=1))
    return affinity


def eval_cheby(te_labels, prid_labels):
    # print('CHEBY');
    if 0 in te_labels or 0 in prid_labels:
        print('input error: containt 0')
        return

    sub = np.abs(te_labels - prid_labels)

    m = np.max(sub, 1)
    res = np.mean(m)
    return res


def eval_clark(te_labels, prid_labels):
    # print('CLARK');
    if 0 in te_labels or 0 in prid_labels:
        print('input error: containt 0')
        return

    sub = np.power(te_labels - prid_labels, 2)
    plus = np.power(te_labels + prid_labels, 2)
    if te_labels.ndim == 1 and prid_labels.ndim == 1:
        res = np.sqrt(np.sum(sub / plus))
    else:
        m = np.sqrt(np.sum(sub / plus, 1))
        res = np.mean(m)
        
    return res


def eval_canberra(te_labels, prid_labels):
    # print('CANBERRA');
    if 0 in te_labels or 0 in prid_labels:
        print('input error: containt 0')
        return
    sub = np.abs(te_labels - prid_labels)
    plus = te_labels + prid_labels

    m = np.sum(sub / plus, 1)
    res = np.mean(m)
    return res


def eval_kl(te_labels, prid_labels):
    # print('KL:');
    if 0 in te_labels or 0 in prid_labels:
        print('input error: containt 0')
        return
    #m = te_labels * np.log(te_labels / prid_labels)
    #res = np.mean(m)
    m = np.sum(te_labels * np.log(te_labels / prid_labels), 1)
    res = np.mean(m)
    return res


def eval_inters(te_labels, prid_labels):
    # print('INTERS:');
    m_chann = np.stack((te_labels, prid_labels), axis=-1)
    m = np.min(m_chann, 2)

    res = np.mean(np.sum(m, 1))
    return res


def affinity_cos(data_1: np.ndarray, data_2: np.ndarray)->np.ndarray:

    data_1 = pp.normalize(data_1, norm='l2')
    data_2 = pp.normalize(data_2, norm='l2')

    affinity = np.matmul(data_1, data_2.T) # [0, 1] max->more similar
    return affinity


def affinity_cos_gpu(data_1: torch.Tensor, data_2: torch.Tensor):

    data_1 = torch.nn.functional.normalize(data_1, p=2, dim=1)  # to calculate the cos-similarity
    data_2 = torch.nn.functional.normalize(data_2, p=2, dim=1)

    affinity = torch.matmul(data_1, data_2.T)
    return affinity


def affinity_clark(label1, label2):
    n = label1.shape[0]
    m = label2.shape[0]
    sim = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            sim[i, j] = np.exp(-eval_clark(label1[i], label2[j]))
    return sim


def calc_eucl_dis_gpu(x, y):
    x2 = torch.sum(torch.pow(x, 2), 1)
    y2 = torch.sum(torch.pow(y, 2), 1)
    m = x.mm(y.t())

    res = x2 + y2.t() - 2 * m 
    res = res - torch.min(res) + 1e-8
    
    res = torch.sqrt(res)
    return res

    
def affinity_euclidean(data_1: torch.Tensor, data_2: torch.Tensor):
    affinity = torch.nn.functional.pairwise_distance(data_1, data_2)
    return 1/affinity


def calc_hammingDist_cpu(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.dot(B2.transpose(1, 0)))
    return distH


def affinity_hamm_gpu(data_1: torch.Tensor, data_2: torch.Tensor):
    dis = calc_hammingDist(torch.sign(data_1), torch.sign(data_1))
    return torch.exp(-dis)


def _softmax(k):
    #ones = np.ones((n, 3))
    x = np.array(range(k))
    x_exp = 0.2 + np.exp(-x)
    #x_exp = 0.1 + np.exp(-x)
    tot = np.sum(x_exp)

    w = x_exp / tot
    w = np.expand_dims(w, 1)

    #res = np.multiply(ones, w)
    #print('res:', res)
    #s = np.sum(res, 0)
    #print(s)
    return w


def eval_ldl(test_label, testL):
    logging.info('--------------------')
    # Chebyshev -
    metric_Chebyshev = eval_cheby(test_label, testL)
    logging.info('| Chebyshev: %.4f' % metric_Chebyshev)

    # Clark -
    metric_Clark = eval_clark(test_label, testL)
    logging.info('| Clark:     %.4f' % metric_Clark)

    # Canberra -
    metric_Canberra = eval_canberra(test_label, testL)
    logging.info('| Canberra:  %.4f' % metric_Canberra)

    # KL -
    metric_KL = eval_kl(test_label, testL)
    logging.info('| KL:        %.4f' % metric_KL)

    # Cosine +
    metric_Cosine = eval_cos(test_label, testL)
    logging.info('| Cosine:    %.4f' % metric_Cosine)
    # logging.info('| Cosine:    \033[1;31;31m%.4f\033[0m' % metric)

    # Intersection +
    metric_Intersection = eval_inters(test_label, testL)
    logging.info('| Intersec:  %.4f' % metric_Intersection)
    logging.info('--------------------')

    return [metric_Chebyshev, metric_Clark, metric_Canberra, metric_KL, metric_Cosine, metric_Intersection]

def euclidean_distances(x, y, squared=True):
    """Compute pairwise (squared) Euclidean distances.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x*x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y*y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances


def select_hard_sample(S_feat, S_label):                                                                                   
    n = S_feat.shape[0]                                                                                                    
    mid = (n - 1) // 2                                                                                                     
    all_hard_pos_idx = []                                                                                                  
    all_hard_neg_idx = []                                                                                                  
    K = 10                                                                                                                 
                                                                                                                           
    for i in range(n):                                                                                                     
        order = np.argsort(S_label[i,:])[:-2]                                                                              
        neg_idx = order[:K]                                                                                                
        pos_idx = order[K:]                                                                                                
                                                                                                                           
        _hard_neg_idx = np.argmax(S_feat[i, neg_idx])                                                                      
        hard_neg_idx = neg_idx[_hard_neg_idx]                                                                              
        all_hard_neg_idx.append(hard_neg_idx)                                                                              
                                                                                                                           
        _hard_pos_idx = np.argmin(S_feat[i, pos_idx])                                                                      
        hard_pos_idx = pos_idx[_hard_pos_idx]                                                                              
        all_hard_pos_idx.append(hard_pos_idx)                                                                              
                                                                                                                           
    return np.asarray(all_hard_pos_idx), np.asarray(all_hard_neg_idx)               

def select_hard_sample_gpu(S_feat, S_label):                                                                                   
    n = S_feat.shape[0]                                                                                                    
    mid = (n - 1) // 2                                                                                                     
    all_hard_pos_idx = []                                                                                                  
    all_hard_neg_idx = []                                                                                                  
    # K = 10                                                                                                                 
                                                                                                                           
    for i in range(n):                                                                                                     
        order = torch.argsort(S_label[i,:])[:-2]                                                                              
        neg_idx = order[:mid]                                                                                                
        pos_idx = order[mid:]                                                                                                
                                                                                                                           
        _hard_neg_idx = torch.argmax(S_feat[i, neg_idx])                                                                      
        hard_neg_idx = neg_idx[_hard_neg_idx]                                                                              
        all_hard_neg_idx.append(hard_neg_idx)                                                                              
                                                                                                                           
        _hard_pos_idx = torch.argmin(S_feat[i, pos_idx])                                                                      
        hard_pos_idx = pos_idx[_hard_pos_idx]                                                                              
        all_hard_pos_idx.append(hard_pos_idx)                                                                              
                                                                                                                           
    return torch.tensor(all_hard_pos_idx), torch.tensor(all_hard_neg_idx)     


def eval_hash_code(B1, B2, label1, label2, topk):
    n = B1.shape[0]

    hamm_dis = calc_hammingDist_cpu(B1, B2)
    hamm_idx = np.argsort(hamm_dis, axis=1)

    cos_sim = affinity_cos(label1, label2)
    cos_idx = np.argsort(-cos_sim, axis=1)

    res = np.zeros((n, topk))

    for i in range(n):
        for j in range(topk):
            idx = np.where(cos_idx[i]==hamm_idx[i,j])
            res[i,j] = idx[0]
    tot = np.sum(res) / n

    logging.info('rank eval :{:,d}'.format(int(tot)))


def write_result(args, result, best_func_epoch, path):
    if not os.path.exists(path):
        mode = 'w'
    else:
        mode = 'a'

    with open(path, mode) as f:
        f.write('%5s %5d %10d %10d %8d %11.5f %11.5f %14.5f %5.5f %5.5f %5.5f %5.5f ----- %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n' %\
             (args.seq, args.epoch, args.epoch_code, best_func_epoch, args.max_iter, args.lr_code_net, args.lr_func_net, args.triplet_margin, \
                 args.eta, args.alpha, args.beta, args.sigma, \
                     result[0], result[1], result[2], result[3], result[4], result[5])) 

#def eval_hash_code(B1, B2, label1, label2, topk):
#    n = B1.shape[0]
#
#    hamm_dis = calc_hammingDist_cpu(B1, B2)
#    hamm_idx = np.argsort(hamm_dis, axis=1)
#
#    cos_dis = affinity_cos(label1, label2)
#    cos_idx = np.argsort(-cos_dis, axis=1)
#
#    res = np.zeros((n, topk))
#
#    for i in range(n):
#        for j in range(topk):
#            idx = np.where(hamm_idx[i]==cos_idx[i,j])
#            res[i,j] = idx[0]
#    tot = np.sum(res) / n
#
#    print('rank eval :{:,d}'.format(int(tot)))

if __name__ == '__main__':
    import scipy.io as scio
    dataFile = 'dbc_ldl_eval_hash.mat'
    data = scio.loadmat(dataFile)

    B1 = data['B_test'][:]
    B2 = data['B_train'][:]
    D1 = data['test_labels'][:]
    D2 = data['train_labels'][:]

    #B1 = np.sign(np.random.randn(10, 8))
    #B2 = B1
    #
    #L1 = np.random.randn(10, 4)
    #L2 = L1

    #K = 5
    
    eval_hash_code(B1, B2, D1, D2, 30)



