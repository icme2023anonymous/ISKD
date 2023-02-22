import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import random, pdb, math, copy
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import datetime
import torch.nn.functional as F
##########################import self###############
import utils
from loss import CrossEntropyLabelSmooth
from data_list import ImageList, ImageList_idx
import network
import loss

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def gmm(all_fea, pi, mu, all_output):    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + 1e-6 * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += 1e-6 * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if netB is None:
                outputs = netC(netF(inputs))
            else:
                outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item() / np.log(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, mean_ent
    else:
        return accuracy*100, mean_ent
def evaluation_tea(loader, netF, netB, netC, args, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in range(len(loader)):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1].cuda()
        inputs = inputs.cuda()
        if netB == None:
            feas = netF(inputs)
        else:
            feas = netB(netF(inputs))
        outputs = netC(feas)
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
            
    _, predict = torch.max(all_output, 1)
    accuracy_return = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).data.item()

    if args.dset=='VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_fea = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + 1e-6), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea),args.class_num)/args.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)
            
    aff = gamma
    
    acc = (pred_label==all_label).float().mean()
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'
    log_str += 'GMM pred_label accuracy  : {:.2f}%\n'.format(acc*100)
    if args.dset=='VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'
        log_str += 'pred_label accuracy  : {:.2f}%\n'.format(acc*100)
    print(log_str)
    nor_mu = F.normalize(mu,p=2,dim=1)
    similarity_matrix = torch.matmul(nor_mu, nor_mu.t())
    rec_weight = torch.matmul(gamma, similarity_matrix)
    weight1 = 1+torch.sum(torch.log(all_output)*all_output,dim=1)/torch.log(torch.tensor([args.class_num]).cuda())
    print(weight1)
    cs =weight1.detach()
    #rec_weight = torch.clamp(rec_weight,min=0)
    if args.dset=='VISDA-C':
        return aff, aacc/100,rec_weight,cs
    return aff, max(accuracy,acc),rec_weight,cs
def evaluation(loader, netF, netB, netC, args, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in range(len(loader)):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1].cuda()
        inputs = inputs.cuda()
        if netB == None:
            feas = netF(inputs)
        else:
            feas = netB(netF(inputs))
        outputs = netC(feas)
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
            
    _, predict = torch.max(all_output, 1)
    accuracy_return = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).data.item()

    if args.dset=='VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + 1e-6), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea),args.class_num)/args.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)
            
    aff = gamma
    
    acc = (pred_label==all_label).float().mean()
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'
    log_str += 'GMM pred_label accuracy  : {:.2f}%\n'.format(acc*100)
    if args.dset=='VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'
        log_str += 'pred_label accuracy  : {:.2f}%\n'.format(acc*100)
    print(log_str)
    nor_mu = F.normalize(mu,p=2,dim=1)
    similarity_matrix = torch.matmul(nor_mu, nor_mu.t())
    rec_weight = torch.matmul(gamma, similarity_matrix)
    weight1 = 1+torch.sum(torch.log(all_output)*all_output,dim=1)/torch.log(torch.tensor([args.class_num]).cuda())
    print(weight1)
    cs =weight1.detach()
    if args.dset=='VISDA-C':
        return aff, aacc/100,rec_weight,cs
    return aff, max(accuracy,acc),rec_weight,cs


def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    count = np.zeros(args.class_num)
    tr_txt = []
    te_txt = []
    for i in range(len(txt_src)):
        line = txt_src[i]
        reci = line.strip().split(' ')
        if count[int(reci[1])] < 3:
            count[int(reci[1])] += 1
            te_txt.append(line)
        else:
            tr_txt.append(line)

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def train_source_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    param_group = []
    learning_rate = args.lr_src
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netF(inputs_source))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, None, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netC

def test_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num = args.class_num, feat_dim=netF.in_features).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, None, netC, False)
    log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    print(log_str + '\n')

def copy_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()   
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    source_model = nn.Sequential(netF, netC).cuda()
    source_model.eval()
    ##################################################
    netF.eval()
    netC.eval()
    with torch.no_grad():
        cnt =0
        soft_pseudo_label, accuracy,rec_weight,cs = evaluation_tea(
            dset_loaders["target_te"], netF, None, netC, args, cnt
        )
        if args.dset == 'office':
            rec_weight = torch.ones(rec_weight.shape).detach().cuda()
    #####################################################
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, pretrain=True).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    ent_best = 1.0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.iter
    iter_num = 0

    model = nn.Sequential(netF, netB, netC).cuda()
    model.eval()
    
    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["target_te"])
        for i in range(len(dset_loaders["target_te"])):
            data = iter_test.next()
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()
            outputs = source_model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            _, src_idx = torch.sort(outputs, 1, descending=True)
            if args.topk > 0:
                topk = np.min([args.topk, args.class_num])
                for i in range(outputs.size()[0]):
                    outputs[i, src_idx[i, topk:]] = (1.0 - outputs[i, src_idx[i, :topk]].sum())/ (outputs.size()[1] - topk)

            if start_test:
                all_output = outputs.float()
                all_label = labels
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels), 0)
        mem_P = all_output.detach()

    model.train()
    maxacc = 0
    while iter_num < max_iter:
 
        if args.ema < 1.0 and iter_num > 0 and iter_num % interval_iter == 0:
            model.eval()
            start_test = True
            with torch.no_grad():
                iter_test = iter(dset_loaders["target_te"])
                for i in range(len(dset_loaders["target_te"])):
                    data = iter_test.next()
                    inputs = data[0]
                    inputs = inputs.cuda()
                    outputs = model(inputs)
                    outputs = nn.Softmax(dim=1)(outputs)
                    if start_test:
                        all_output = outputs.float()
                        start_test = False
                    else:
                        all_output = torch.cat((all_output, outputs.float()), 0)
                mem_P = mem_P * args.ema + all_output.detach() * (1 - args.ema)
            model.train()

        try:
            inputs_target, y, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, y, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=1.5)
        inputs_target = inputs_target.cuda()
        with torch.no_grad():
            outputs_target_by_source = mem_P[tar_idx, :]*rec_weight[tar_idx, :]
            _, src_idx = torch.sort(outputs_target_by_source, 1, descending=True)
        outputs_target = model(inputs_target)
        outputs_target = torch.nn.Softmax(dim=1)(outputs_target)
        if iter_num!=1:
            classifier_loss = nn.KLDivLoss(reduction='none')(outputs_target.log(),  outputs_target_by_source)
            classifier_loss = torch.sum(torch.sum(classifier_loss,dim=1))/classifier_loss.shape[0]
        else:
            classifier_loss = nn.KLDivLoss(reduction='batchmean')(outputs_target.log(), outputs_target_by_source)
        optimizer.zero_grad()

        entropy_loss = torch.mean(loss.Entropy(outputs_target))
        msoftmax = outputs_target.mean(dim=0)
        gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        classifier_loss += args.ent*entropy_loss

        classifier_loss.backward()

        if args.mix > 0:
            alpha = 0.3
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(inputs_target.size()[0]).cuda()
            mixed_input = lam * inputs_target + (1 - lam) * inputs_target[index, :]
            mixed_output = (lam * outputs_target + (1 - lam) * outputs_target[index, :]).detach()

            update_batch_stats(model, False)
            outputs_target_m = model(mixed_input)
            update_batch_stats(model, True)
            outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
            classifier_loss = args.mix*nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
            classifier_loss.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc_s_te, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)

            # ##################################################
            netF.eval()
            netB.eval()
            netC.eval()
            with torch.no_grad():
                cnt =0
                soft_pseudo_label, accuracy,rec_weight,cs = evaluation(
                    dset_loaders["target_te"], netF, netB, netC, args, cnt
                )
            if maxacc<accuracy:
                maxacc = accuracy
                torch.save(netF.state_dict(), osp.join(args.output_dir, "source_F.pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "source_B.pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "source_C.pt"))
            # #####################################################

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Ent = {:.4f}'.format(args.name, iter_num, max_iter, acc_s_te, mean_ent)
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            print(log_str+'\n')
            model.train()

    return maxacc

   

def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISKD")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'image-clef', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--lr_src', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net_src', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output_src', type=str, default='san')  
    parser.add_argument('--iter', type=int, default=10)  
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--topk', type=int, default=1)

    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--ema', type=float, default=0.6)
    parser.add_argument('--mix', type=float, default=1.0)
    parser.add_argument('--ent', type=float, default=1.0)
    args = parser.parse_args()
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DomainNet':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = ''
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
    #######################
    nowTime = datetime.datetime.now().strftime('%m-%d-%H-%M')
    logfilename = "./log/train_log_file-" + nowTime + ".txt"
    sys.stdout = utils.LoggerTxt(logfilename)
    print(args)
    ####################
    args.output_dir_src = osp.join(args.output_src, args.net_src, str(args.seed), args.da, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
        
    maxaccs=[]
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            
        test_target_simp(args)
        maxaccs.append(copy_target_simp(args))
    print(maxaccs)
