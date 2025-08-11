import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import sklearn.metrics

from utils.dataset_loader import WTVDataset
from models.model import *
from utils.Loss import cal_predictor_loss
from utils.util import *
from utils.reward import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
rec_length = 20

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--gamma', type=float, default=1, help='Weight of individual- and population-level rewards')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='path to check pioint directory')
parser.add_argument('--data_dir', type=str, default='/path/to/data', help='path to data directory')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
opt = parser.parse_args()



def collect_active_sensing_result(active_sensing_res, pd_prob_np, label_np, action_reward_np, mask_np, num_record):
    for ib in range(action_reward_np.shape[0]):
        label = label_np[ib]
        n_record = num_record[ib]
        for modal in ['walking', 'tapping', 'voice', 'multi']:
            if modal == 'multi':
                n_rec, iw, it, iv = 0, 0, 0, 0
                key = 'active_sensing_' + modal
                while n_rec < rec_length:
                    ma = mask_np[ib, iw, it, iv].copy()
                    ar = action_reward_np[ib, iw, it, iv].copy()
                    ar[ma == 0] = -1
                    if ma[:3].max() <= 0:
                        break
                    if sum(ma) > 1:
                        idx = np.argmax(ar)
                        if ar[idx] <= -1:
                            break
                        elif idx == 0:
                            iw += 1
                            if iw >= pd_prob_np.shape[1]:
                                iw -= 1
                                break
                        elif idx == 1:
                            it += 1
                            if it >= pd_prob_np.shape[2]:
                                it -= 1
                                break
                        elif idx == 2:
                            iv += 1
                            if iv >= pd_prob_np.shape[3]:
                                iv -= 1
                                break
                        else:
                            break
                        n_rec = iw + it + iv
                        if n_rec < len(active_sensing_res[key]):
                            active_sensing_res[key][n_rec]['pred'].append(pd_prob_np[ib, iw, it, iv])
                            active_sensing_res[key][n_rec]['label'].append(label)
                            active_sensing_res[key][n_rec]['total_num'] += n_rec
                while n_rec < len(active_sensing_res[key]) and n_rec > 0:
                    active_sensing_res[key][n_rec]['pred'].append(pd_prob_np[ib, iw, it, iv])
                    active_sensing_res[key][n_rec]['label'].append(label)
                    active_sensing_res[key][n_rec]['total_num'] += iw + it + iv
                    n_rec += 1
            else:
                if modal == 'walking':
                    action_reward = action_reward_np[ib, :, 0, 0, :]
                    pd_prob = pd_prob_np[ib, :, 0, 0, :]
                    mask = mask_np[ib, :, 0, 0, :]
                    idx_modal = 0
                elif modal == 'tapping':
                    action_reward = action_reward_np[ib, 0, :, 0, :]
                    mask = mask_np[ib, 0, :, 0, :]
                    pd_prob = pd_prob_np[ib, 0, :, 0, :]
                    idx_modal = 1
                elif modal == 'voice':
                    action_reward = action_reward_np[ib, 0, 0, :, :]
                    mask = mask_np[ib, 0, 0, :, :]
                    pd_prob = pd_prob_np[ib, 0, 0, :, :]
                    idx_modal = 2
                pass
                key = 'active_sensing_' + modal
                max_n_rec = -1
                for n_rec in range(len(action_reward) - 1):
                    ma = mask[n_rec + 1, idx_modal]
                    if ma <= 0:
                        break
                    if action_reward[n_rec, idx_modal] < action_reward[n_rec, 3]:
                        break
                    if n_rec >= len(active_sensing_res[key]):
                        break
                    active_sensing_res[key][n_rec]['pred'].append(pd_prob[n_rec + 1])
                    active_sensing_res[key][n_rec]['label'].append(label)
                    active_sensing_res[key][n_rec]['total_num'] += n_rec + 1
                    max_n_rec = n_rec
                if max_n_rec > -1:
                    for n_rec in range(max_n_rec + 1, len(action_reward) - 1):
                        if n_rec >= len(active_sensing_res[key]):
                            break
                        active_sensing_res[key][n_rec]['pred'].append(pd_prob[max_n_rec + 1])
                        active_sensing_res[key][n_rec]['label'].append(label)
                        active_sensing_res[key][n_rec]['total_num'] += max_n_rec + 1


def train_eval(train_loader, device, model, optimizer, p_dict, phase='train'):
    if phase == 'train':
        for k in model:
            model[k].train()
    else:
        for k in model:
            model[k].eval()
    pred_list, label_list, critic_correct_list = [], [], []
    loss_dict = {
        'predictor_loss': [],
        'critic_loss': [],
        'actor_loss': [],
        'loss': []
    }
    reward_res = [[[] for r in range(12)] for _ in range(rec_length)]
    active_sensing_res = {
        'active_sensing_multi': [{'pred': [], 'label': [], 'total_num': 0} for _ in range(rec_length)],
        'active_sensing_walking': [{'pred': [], 'label': [], 'total_num': 0} for _ in range(rec_length)],
        'active_sensing_tapping': [{'pred': [], 'label': [], 'total_num': 0} for _ in range(rec_length)],
        'active_sensing_voice': [{'pred': [], 'label': [], 'total_num': 0} for _ in range(rec_length)],
    }

    for batch in tqdm(train_loader):
        pid = batch[0]
        label, walking_data, tapping_data, tapping_accel_data, voice_data = [x.to(device) for x in batch[1:6]]
        n_rec = batch[6]
        loss = 0
        embedding, pd_prob = model['predictor'](walking_data, tapping_data, tapping_accel_data, voice_data, False)
        embedding = embedding.detach()
        action_reward = model['critic'](embedding.detach())
        reward_gt, mask_next, mask_current, p_dict = get_reward(p_dict, pd_prob, n_rec, label, phase)
        if p_dict['train_predictor']:
            predictor_loss = cal_predictor_loss(pd_prob, label, mask_current, mask_next, device, phase)
            loss_dict['predictor_loss'].append(predictor_loss.to('cpu').data.numpy())
            loss += predictor_loss
        idx_nonzero = mask_next != 0
        if p_dict['train_critic']:
            critic_loss = nn.MSELoss()(action_reward[idx_nonzero], reward_gt[idx_nonzero])
            loss += critic_loss
            loss_dict['critic_loss'].append(critic_loss.to('cpu').data.numpy())
        action_reward_np = action_reward.cpu().data.numpy()
        mask_np = mask_next.cpu().data.numpy()
        reward_gt_np = reward_gt.cpu().data.numpy()
        pd_prob_np = pd_prob.cpu().data.numpy()
        label_np = label.cpu().data.numpy()
        collect_active_sensing_result(active_sensing_res, pd_prob_np, label_np, action_reward_np, mask_np,
                                      n_rec.cpu().data.numpy())
        for ib in range(action_reward_np.shape[0]):
            for iw in range(action_reward_np.shape[1]):
                for it in range(action_reward_np.shape[2]):
                    for iv in range(action_reward_np.shape[3]):
                        ma = mask_np[ib, iw, it, iv]
                        ar = action_reward_np[ib, iw, it, iv]
                        rg = reward_gt_np[ib, iw, it, iv]
                        assert len(ma) == len(ar) == len(rg) == 4
                        arm = [[a, r, m, p] for a, r, m, p in zip(ar, rg, ma, range(4))]
                        arm = sorted(arm, key=lambda x: x[0], reverse=True)
                        if ma.sum() > 2:
                            for i_rank in range(4):
                                a, r, m, p = arm[i_rank]
                                if m > 0:
                                    n_records = iw + it + iv
                                    if n_records < len(reward_res):
                                        reward_res[n_records][i_rank].append(r)
                                        reward_res[n_records][4 + p].append(r)
                                        reward_res[n_records][8 + p].append(a)
                            arm = [x for x in arm if x[2] > 0]
                            if arm[0][1] == max([x[1] for x in arm]):
                                critic_correct_list.append(1)
                            else:
                                critic_correct_list.append(0)
        if phase == 'train':
            if p_dict['train_predictor']:
                optimizer['predictor'].zero_grad()
            if p_dict['train_actor']:
                optimizer['actor'].zero_grad()
            if p_dict['train_critic']:
                optimizer['critic'].zero_grad()
            loss.backward()
            if p_dict['train_predictor']:
                optimizer['predictor'].step()
            if p_dict['train_actor']:
                optimizer['actor'].step()
            if p_dict['train_critic']:
                optimizer['critic'].step()
        label_value = label.to('cpu').detach().numpy().reshape(-1)
        pred_value = pd_prob.to('cpu').detach().numpy().reshape(pd_prob.size(0), -1)
        assert len(label_value) == len(pred_value)
        assert pred_value.max() <= 1
        assert pred_value.min() >= 0
        for ib in range(len(label_value)):
            p = pred_value[ib]
            l = label_value[ib]
            pred_list += list(p)
            label_list += [l for _ in p]
    auc = compute_auc(label_list, pred_list)
    print(f'Epoch: {p_dict["epoch"]}  Phase: {phase}  Predictor AUC: {auc * 100:5.2f}%')
    if p_dict['train_predictor'] and phase == 'train':
        predictor_loss_val = np.mean(loss_dict['predictor_loss'])
        print(f'Predictor Loss: {predictor_loss_val:5.5f}')
    if p_dict['train_critic']:
        critic_loss_val = np.mean(loss_dict['critic_loss'])
        critic_accuracy = np.mean(critic_correct_list)
        print(f'Critic Loss: {critic_loss_val:5.5f}, Correct Rate: {critic_accuracy * 100:5.2f}%')
    print('--' * 45 + phase if phase != 'train' else '' + '--' * 45)
    performance_res = {}
    for key in active_sensing_res:
        mauc_list = []
        avg_record_list = []
        local_ratios = []
        for i in range(rec_length):
            record = active_sensing_res[key][i]
            if len(record['label']) > 1:
                mauc = compute_auc(record['label'], record['pred'])
                auc_percent = mauc * 100.0
                avg_record = record["total_num"] / len(record["label"])
                mauc_list.append(auc_percent)
                avg_record_list.append(avg_record)
                if key == 'active_sensing_multi' and phase == 'test':
                    local_ratios.append(auc_percent / avg_record)
            else:
                mauc_list.append(None)
                avg_record_list.append(None)
        if phase != 'train':
            performance_res[key] = {'auc': mauc_list, 'avg_record': avg_record_list}
            formatted_mauc = ["{:5.2f}".format(val) if val is not None else "    " for val in mauc_list]
            formatted_avg_record = ["{:5.2f}".format(val) if val is not None else "    " for val in avg_record_list]
            print(f'{key:25s}: {" ".join(formatted_mauc)}')
            print(f'{key:25s}: {" ".join(formatted_avg_record)}')
    if phase == 'valid':
        if auc > p_dict.get('best_auc', 0):
            p_dict['best_predictor_auc'] = auc
            p_dict['best_predictor_epoch'] = p_dict['epoch']
            if p_dict['train_predictor']:
                save_model(p_dict, model, optimizer, 'predictor')
        print(f'Best Predictor Epoch: {p_dict["best_predictor_epoch"]}  AUC: {p_dict["best_predictor_auc"] * 100:5.2f}%')
        if p_dict['train_critic']:
            if critic_accuracy > p_dict.get('best_critic_accuracy', 0):
                p_dict['best_critic_accuracy'] = critic_accuracy
                p_dict['best_critic_loss'] = critic_loss_val
                p_dict['best_critic_epoch'] = p_dict['epoch']
                save_model(p_dict, model, optimizer, 'critic')
            print(
                f'Best Critic Epoch: {p_dict["best_critic_epoch"]}  Loss: {p_dict["best_critic_loss"]:5.5f}  Accuracy: {p_dict["best_critic_accuracy"] * 100:5.2f}%')
        print('--' * 100)
    return p_dict


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    with open(os.path.join(opt.data_dir, 'data_split.pkl'), 'rb') as f:
        data_split = pickle.load(f)
    pid_set = set(os.listdir(os.path.join(opt.data_dir, 'raw_data')))
    pid_rec_file = os.path.join(opt.data_dir, 'pid_rec_dict.json')
    if not os.path.exists(pid_rec_file):
        pid_rec_dict = {}
        print('writing pid_rec_dict.json')
        for pid in tqdm(pid_set):
            fi = os.path.join(opt.data_dir, 'raw_data', pid, 'pid.json')
            d = json.load(open(fi))
            pid_rec_dict[pid] = {}
            for modality, md in d.items():
                if type(md) is dict:
                    for k, v in md.items():
                        if k.startswith('num_record'):
                            pid_rec_dict[pid][k] = v
    else:
        pid_rec_dict = json.load(open(pid_rec_file, 'r'))
    len_valid = len(data_split['val'])
    data_val = data_split['test'][-len_valid:]
    data_test = data_split['val'] + data_split['test'][:-len_valid]
    healthCode_train = {pid: pid_rec_dict[pid] for pid in data_split['train'] if
                        pid in pid_set and pid in pid_rec_dict}
    healthCode_val = {pid: pid_rec_dict[pid] for pid in data_val if pid in pid_set and pid in pid_rec_dict}
    healthCode_test = {pid: pid_rec_dict[pid] for pid in data_test if pid in pid_set and pid in pid_rec_dict}
    print(f'There are {len(healthCode_train)} Patients in training set')
    print(f'There are {len(healthCode_val)} Patients in validation set')
    print(f'There are {len(healthCode_test)} Patients in test set')
    train_data = WTVDataset(opt, healthCode_train, phase='train')
    valid_data = WTVDataset(opt, healthCode_val, phase='valid')
    test_data = WTVDataset(opt, healthCode_test, phase='test')
    train_loader = DataLoader(dataset=train_data, batch_size=16, num_workers=4, shuffle=True, pin_memory=True,
                              prefetch_factor=10)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, num_workers=4, shuffle=False, pin_memory=True,
                              prefetch_factor=10)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4, shuffle=False, pin_memory=True,
                             prefetch_factor=10)
    
    
    # Model and optimizer
    num_filters = 64
    model = {'predictor': Predictor(num_filters).to(device),
             'critic': Critic(num_filters).to(device)
             }
    optimizer = {
        'predictor': torch.optim.Adam(model['predictor'].parameters(), opt.lr),
        'critic': torch.optim.Adam(model['critic'].parameters(), opt.lr)
    }
    p_dict = {}
    start_epoch = 0
    p_dict['train_predictor'] = True
    p_dict['train_actor'] = False
    p_dict['train_critic'] = True
    p_dict['device'] = device
    p_dict['loss_mat'] = None
    p_dict['w_reward_pop'] = 1/len(train_loader)
    p_dict['model_folder'] = opt.checkpoint_dir
    if not os.path.exists(p_dict['model_folder']):
        os.makedirs(p_dict['model_folder'])
    p_dict['epoch'] = 0
    p_dict['gamma'] = opt.gamma

    # Training and validation
    for epoch_i in range(start_epoch, opt.epoch):
        p_dict['epoch'] = epoch_i
        train_eval(train_loader, device, model, optimizer, p_dict, phase='train')
        train_eval(valid_loader, device, model, optimizer, p_dict, phase='valid')

    # Evaluation on the test set    
    test_model = {'predictor': Predictor(num_filters).to(device),
                  'critic': Critic(num_filters).to(device)
                  }
    load_model(p_dict, test_model, optimizer, key='predictor')
    load_model(p_dict, test_model, optimizer, key='critic')
    train_eval(test_loader, device, test_model, optimizer, p_dict, phase='test')


if __name__ == '__main__':
    main()