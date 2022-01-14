import argparse
import numpy as np
import tqdm
import logging
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
from scipy import io as scio
# from numpy.core.arrayprint import _void_scalar_repr
# from scipy.linalg import hadamard as hadamard

import utils
import dataset
from model import HashCodeNet
from model import HashFuncNet

loss_l1 = torch.nn.L1Loss()
loss_l2 = torch.nn.MSELoss()
crossentropyloss = nn.CrossEntropyLoss()
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
aggregate_file = 'aggregate.txt'

def ldl_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_dir', type=str, default='output')       # 当前参数的目录，此进程运行一次的使用一个目录
    parser.add_argument('--seq', type=str, default=0, help='seq')

    parser.add_argument('--dataset', type=str, default='Flickr_LDL.mat')
    parser.add_argument('--ROOT', type=str, default='/data/s2020020937/LDL/dataset/partition/')

    parser.add_argument('--bit', type=int, default=128)
    parser.add_argument('--TOPK', type=int, default=30)
    
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--epoch_code', type=int, default=2)
    parser.add_argument('--epoch_func', type=int, default=200)
    parser.add_argument('--epoch_func_test_inter', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=3)  #离散求解迭代伦次
    parser.add_argument('--lr_code_net', type=float, default=0.0001)
    parser.add_argument('--lr_func_net', type=float, default=0.0001)
    parser.add_argument('--triplet_margin', type=float, default=0.2, help='triplet_margin')
    parser.add_argument('--eta', type=float, default=0.5, help='code loss, eta * loss_S + (1 - eta) * loss_B')
    parser.add_argument('--alpha', type=float, default=0.5, help='For calc R')
    parser.add_argument('--beta', type=float, default=0.5, help='For calc R')
    # parser.add_argument('--gama', type=float, default=0.5, help='For calc R')
    parser.add_argument('--sigma', type=float, default=0.5, help='func loss, sigma * loss_S + (1 - sigma) * loss_H')
    # parser.add_argument('--S', type=float, default=10, help='loss_S')
    args = parser.parse_args()

    return args

def log_set(args):
    record_dir = os.path.join(os.getcwd(), args.record_dir, args.seq)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    #logging.basicConfig(filename=os.path.join(record_dir, 'print.log'), filemode='w', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def ldl_test(hash_func_net, R, train_feature, train_label, test_feature, test_label, args):
    hash_func_net.eval()
    train_dataloader = data.DataLoader(dataset.LdlDatasetCode(train_feature, train_label), 
                                       batch_size=train_feature.shape[0], 
                                       shuffle=False)

    test_dataloader = data.DataLoader(dataset.LdlDatasetCode(test_feature, test_label), 
                                      batch_size=test_feature.shape[0], 
                                      shuffle=False)
    
    db = []
    db_label = train_label
    for i, (feat, _) in enumerate(train_dataloader):
        feat = Variable(feat).cuda()
        hash = hash_func_net(feat) @ R
        db.append(hash.data.cpu().numpy())

    db_H = np.concatenate(db)
    db_code = np.sign(db_H)

    test = []
    testL = np.zeros(test_label.shape) 
    for i, (feat, _) in enumerate(test_dataloader):
        feat = Variable(feat).cuda()
        hash = hash_func_net(feat) @ R
        test.append(hash.data.cpu().numpy())

    test_H = np.concatenate(test)
    test_code = np.sign(test_H)

    hamm_dist = test_code @ db_code.T
    hamm_idx = np.argsort(-hamm_dist)[:, :args.TOPK]

    w = utils._softmax(args.TOPK)
    for i in range(test_code.shape[0]):
        tL = np.mean(db_label[hamm_idx[i]], 0)
        testL[i] = tL

    logging.info('test to db: ')
    utils.eval_hash_code(test_code, db_code, test_label, db_label, args.TOPK)
    return utils.eval_ldl(test_label, testL)
    

def train_code_net(hash_code_net, train_feature, train_label, args):
    train_dataloader = data.DataLoader(dataset.LdlDatasetCode(train_feature, train_label), 
                                       batch_size=train_feature.shape[0], 
                                       shuffle=True)
    opt = torch.optim.Adam(hash_code_net.parameters(), lr=args.lr_code_net)
    feat_final, label_final, H_final = None, None, None
    triplet_loss = nn.TripletMarginLoss(margin=args.triplet_margin, p=2)

    hash_code_net.cuda().train()
    
    for epoch in range(1, args.epoch_code + 1):
        # logging.info('epoch_code [%d/%d]' % (epoch, args.epoch_code))
        for i, (feat, label) in enumerate(train_dataloader):
            feat_final = feat.clone()
            label_final = label.clone()
            opt.zero_grad()

            feat = Variable(feat).cuda()
            label = Variable(label).cuda()
            S_feat = utils.affinity_cos_gpu(feat, feat)
            S_label = utils.affinity_cos_gpu(label, label)

            S_label_limit = S_label.clone()
            S_label_limit[S_label_limit < 0.5] = 0

            H = hash_code_net(feat, S_label_limit)
            # H_norm = F.normalize(H)
            B = torch.sign(H)
            H_final = H.detach().cpu().clone()

            loss_qua = loss_l2(H, B)

            # loss_S = loss_l2(H_norm.mm(H_norm.t()), S_label) * 10
            pos_idx, neg_idx = utils.select_hard_sample_gpu(S_feat, S_label)
            loss_rank = triplet_loss(H, H[pos_idx], H[neg_idx])

            loss = args.eta * loss_rank + (1 - args.eta) * loss_qua

            loss.backward()
            opt.step()
            # logging.info('loss code = %.4f' % (loss.item()))

    return feat_final, H_final, label_final

def cal_rotate(H, L, args):
    max_iter = args.max_iter                                                                                          
    _, l = L.shape                                                                                                         
    bit = args.bit                                                                                                         
                                                                                                                           
    alpha = args.alpha   

    beta = args.beta
    gama = 1 - args.beta                                                                                           
                                                                                                                           
    B = np.sign(H)                                                                                                         
    P = np.random.randn(l, bit)                                                                                           
    R = np.random.randn(bit, bit)                                                                                          
    R, _, _ = np.linalg.svd(R)                                                                                             
    R = R[:, :bit]                                                                                                         
    S = (utils.affinity_cos(L, L) * 2 - 1) * bit # TODO init                                                               
    M = B                                                                                                                  
                                                                                                                           
    for i in range(1, max_iter + 1):                                                                                       
        P = np.linalg.pinv(L) @ B                                                                                          
        left, _, right = np.linalg.svd(B.T @ H)                                                                            
        R = right.T @ left.T                                                                                               
                                                                                                                           
        B = np.sign((alpha * S @ M + beta * H @ R + gama * L @ P) @ np.linalg.pinv(alpha * M.T @ M + (beta + gama) * np.eye(bit)))             
        M = B                                                                                                              
                                                                                                                           
        # term1 = alpha * np.sum(np.power(S - B @ M.T, 2))                                                                       
        # term2 = beta * np.sum(np.power(B - H @ R, 2))                                                                         
        # term3 = gama * np.sum(np.power(B - L @ P, 2))                                                                                                                                                      
        # loss = term1 + term2 + term3                                                                                       
                                                                                                                           
        # logging.info('iter[%2d/%2d] tot = %.3f, [%.3f] [%.3f] [%.3f]' % (i, max_iter, loss, term1, term2, term3))                 
                                                                                                                           
    return R                                                                               

def train_func_net(hash_func_net, R, _X, _H, _L, train_feature, train_label, test_feature, test_label, args):
    train_dataloader = data.DataLoader(dataset.LdlDatasetFunc(_X, _H, _L), batch_size=_X.shape[0], shuffle=True)
    opt = torch.optim.Adam(hash_func_net.parameters(), lr=args.lr_func_net)
    hash_func_net.cuda()
    hash_func_net.train()
    inter = args.epoch_func_test_inter
    score_result = np.zeros((int(args.epoch_func/inter), 6))

    for epoch in range(args.epoch_func):
        logging.info('epoch_func[%3d/%3d]' % (epoch, args.epoch_func))
        for X, H_tea, L in train_dataloader:
            opt.zero_grad()

            X = Variable(X).cuda()
            H_tea = Variable(H_tea).cuda()
            L = Variable(L).cuda()

            S_label = utils.affinity_cos_gpu(L, L)
            S_label_limit = S_label.clone()
            S_label_limit[S_label_limit < 0.5] = 0

            H = hash_func_net(X)
            H_norm = F.normalize(H)
            B = torch.sign(H)

            # loss_B = loss_l2(H, B) + loss_l2(H, H_tea) * args.loss_B
            loss_H = loss_l2(H, H_tea)
            loss_S = loss_l2(H_norm.mm(H_norm.t()), S_label_limit)
            loss = args.sigma * loss_S + (1 - args.sigma) * loss_H

            loss.backward()
            opt.step()
            # logging.info('func loss: %.3f' % (loss.item()))
        if epoch % inter == 0:
            res = ldl_test(hash_func_net, R, train_feature, train_label, test_feature, test_label, args)
            res = np.asarray(res)
            score_result[int(epoch/inter), :] = res

    # scio.savemat(os.path.join(os.getcwd(), args.record_dir, args.seq, 'result.mat'), {'result': score_result})    
    np.savetxt(os.path.join(os.getcwd(), args.record_dir, args.seq, 'result.txt'), score_result, fmt='%7.4f')
    best_idx = np.argmax(score_result[:, 4]) 
    best_score = score_result[best_idx, :]
    path = os.path.join(os.getcwd(), args.record_dir, aggregate_file)
    utils.write_result(args, best_score, best_idx * inter, path)


def ldl_run(hash_code_net, hash_func_net, train_feature, train_label, test_feature, test_label, args):
    
    for epoch in range(1, args.epoch + 1):
        logging.info('epoch [%3d/%3d]' % (epoch, args.epoch))
        X, H, L = train_code_net(hash_code_net, train_feature, train_label, args)

        R = cal_rotate(H.detach().cpu().numpy(), L.cpu().numpy(), args)
        R = torch.from_numpy(R).float().cuda()
  
        train_func_net(hash_func_net, R, X, H, L, train_feature, train_label, test_feature, test_label, args)
        
    


def main():
    utils.seed_setting()
    args = ldl_parse_args()
    log_set(args)
    logging.info('=' * 120)
    logging.info(args)
    train_feature, train_label, test_feature, test_label = dataset.load_dataset(os.path.join(args.ROOT, args.dataset))
    feature_dim = train_feature.shape[1]
    
    hash_code_net = HashCodeNet(feature_dim, args.bit)
    hash_func_net = HashFuncNet(feature_dim, args.bit)
    
    ldl_run(hash_code_net, hash_func_net, train_feature, train_label, test_feature, test_label, args)

if __name__ == '__main__':
    main()

