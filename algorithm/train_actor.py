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

from utils.dataset_loader import WTVDataset
from models.model import *
from utils.Loss import cal_actor_loss
from utils.reward import *
from utils.util import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--gamma', type=float, default=1, help='Weight of individual- and population-level rewards')
parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint', help='path to check pioint directory')
parser.add_argument('--data_dir', type=str, default='/path/to/data', help='path to data directory')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

rec_length = 20

def get_action(p_dict, reward, mask):
    reward = reward.to('cpu').detach().numpy()
    mask = mask.to('cpu').detach().numpy()
    action_score = np.zeros_like(reward, dtype=np.float32)
    action_score[mask == 0] = -np.inf
    action_indices = np.argmax(action_score, axis=-1)
    return torch.from_numpy(action_indices).long().to(p_dict['device'])


def action_result(active_sensing_res, pd_prob_np, label_np, action_np, mask_np, num_record,
                  key='active_sensing_multi'):
    for ib in range(action_np.shape[0]):
        label = label_np[ib]
        n_record = num_record[ib]
        n_rec, iw, it, iv = 0, 0, 0, 0
        while n_rec < rec_length:
            ar = action_np[ib, iw, it, iv].copy()
            ma = mask_np[ib, iw, it, iv].copy()
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


def train_eval(train_loader, device, model, optimizer, p_dict, phase='train'):
    if phase == 'train':
        for k in model:
            model[k].train()
    else:
        for k in model:
            model[k].eval()
    pred_list, label_list = [], []
    loss_dict = {
        'predictor_loss': [],
        'critic_loss': [],
        'actor_loss': [],
        'loss': []
    }
    active_sensing_res = {
        'active_sensing_multi': [{'pred': [], 'label': [], 'total_num': 0} for _ in range(rec_length)]
    }
    i_batch = 0
    for batch in train_loader:
        i_batch += 1
        pid = batch[0]
        label, walking_data, tapping_data, tapping_accel_data, voice_data = [x.to(device) for x in batch[1:6]]
        n_rec = batch[6]
        loss = 0
        embedding, pd_prob = model['predictor'](walking_data, tapping_data, tapping_accel_data, voice_data, False)
        embedding = embedding.detach()
        action_reward = model['critic'](embedding.detach())
        reward_gt, mask_next, mask_current, p_dict = get_reward(p_dict, pd_prob, n_rec, label, phase)
        action_gt = get_action(p_dict, action_reward, mask_next)
        action = model['actor'](embedding.detach())
        if p_dict['train_actor']:
            actor_loss = cal_actor_loss(action, action_gt, mask_next)
            loss += actor_loss
            loss_dict['actor_loss'].append(actor_loss.to('cpu').data.numpy())
        action_reward_np = action_reward.cpu().data.numpy()
        action_np = action.cpu().data.numpy()
        action_reward_np -= p_dict['cost']
        # action_np -= p_dict['cost']
        mask_np = mask_next.cpu().data.numpy()
        pd_prob_np = pd_prob.cpu().data.numpy()
        label_np = label.cpu().data.numpy()
        action_result(active_sensing_res, pd_prob_np, label_np, action_np, mask_np, n_rec.cpu().data.numpy())
        if phase == 'train':
            if p_dict['train_actor']:
                optimizer['actor'].zero_grad()
            loss.backward()
            if p_dict['train_actor']:
                optimizer['actor'].step()
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
    if p_dict['train_actor']:
        actor_loss_val = np.mean(loss_dict['actor_loss'])
        print(f'Actor Loss: {actor_loss_val:5.5f}')
    print('--' * 20 + phase if phase != 'train' else '' + '--' * 20)
    performance_res = {}
    for key in ['active_sensing_multi']:
        mauc_list = []
        avg_record_list = []
        for i in range(rec_length):
            record = active_sensing_res[key][i]
            if len(record['label']) > 1:
                mauc = compute_auc(record['label'], record['pred'])
                auc_percent = mauc * 100.0
                avg_record = record["total_num"] / len(record["label"])
                mauc_list.append(auc_percent)
                avg_record_list.append(avg_record)
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
        print(
            f'Best Predictor Epoch: {p_dict["best_predictor_epoch"]}  AUC: {p_dict["best_predictor_auc"] * 100:5.2f}%')
        if p_dict['train_actor']:
            if mauc_list[4] > p_dict.get('best_actor_auc', 0):
                p_dict['best_actor_auc'] = mauc_list[4]
                p_dict['best_actor_epoch'] = p_dict['epoch']
                save_model(p_dict, model, optimizer, 'actor')
            print(f'Best Actor Epoch: {p_dict["best_actor_epoch"]}  AUC: {p_dict["best_actor_auc"] * 100:5.2f}%')
        print('--' * 40)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    num_filters = 64
    model = {'predictor': Predictor(num_filters).to(device),
             'actor': Actor(num_filters).to(device),
             'critic': Critic(num_filters).to(device)
             }
    optimizer = {
        'predictor': torch.optim.Adam(model['predictor'].parameters(), opt.lr),
        'actor': torch.optim.Adam(model['actor'].parameters(), opt.lr),
        'critic': torch.optim.Adam(model['critic'].parameters(), opt.lr)
    }
    p_dict = {}
    p_dict['train_predictor'] = False
    p_dict['train_actor'] = True
    p_dict['train_critic'] = False
    p_dict['device'] = device
    p_dict['loss_mat'] = None
    p_dict['w_reward_pop'] = 1 / len(train_loader)
    p_dict['model_folder'] = opt.checkpoint_dir
    
    if not os.path.exists(p_dict['model_folder']):
        os.makedirs(p_dict['model_folder'])
    p_dict['epoch'] = 0
    p_dict['gamma'] = opt.gamma
    p_dict['cost'] = [0, 0, 0, 0.5]
    start_epoch = 0
    for epoch_i in range(start_epoch, opt.epoch):
        p_dict['epoch'] = epoch_i
        train_eval(train_loader, device, model, optimizer, p_dict, phase='train')
        train_eval(valid_loader, device, model, optimizer, p_dict, phase='valid')

    
    test_model = {'predictor': Predictor(num_filters).to(device),
                  'actor': Actor(num_filters).to(device),
                  'critic': Critic(num_filters).to(device)
                  }
    load_model(p_dict, test_model, optimizer, key='predictor')
    load_model(p_dict, test_model, optimizer, key='critic')
    load_model(p_dict, test_model, optimizer, key='actor')
    train_eval(test_loader, device, test_model, optimizer, p_dict, phase='test')


if __name__ == '__main__':
    main()