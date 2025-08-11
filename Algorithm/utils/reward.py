import torch
import numpy as np


def get_reward(p_dict, pd_prob, n_rec, label, phase):
    pd_prob = pd_prob.to('cpu').data.numpy()
    label = label.to('cpu').data.numpy().reshape([-1, 1, 1, 1])
    n_rec = n_rec.to('cpu').data.numpy()
    short_term_reward, mask_next = get_short_term_reward(p_dict, pd_prob, n_rec, label)
    population_level_reward, p_dict = get_population_level_reward(p_dict, pd_prob, n_rec, label, phase)
    reward = short_term_reward - p_dict['gamma'] * population_level_reward
    pos_mask = label == 1
    neg_mask = label == 0
    pos_mask = pos_mask.reshape(-1)
    neg_mask = neg_mask.reshape(-1)
    reward[pos_mask, :, :, :, :] *= 0.9
    reward[neg_mask, :, :, :, :] *= 0.1
    mask_current = mask_next.max(-1)
    mask_current = mask_current.reshape(list(mask_current.shape) + [1])
    return torch.from_numpy(reward).to(p_dict['device']), torch.from_numpy(mask_next).to(
        p_dict['device']), torch.from_numpy(mask_current).to(p_dict['device']), p_dict


def get_short_term_reward(p_dict, pd_prob, n_rec, label):
    reward_val = np.zeros(list(pd_prob.shape)[:4] + [4], dtype=np.float32)
    mask = np.ones(list(pd_prob.shape)[:4] + [4], dtype=np.float32)
    reward_val[:, :-1, :, :, 0] = (pd_prob[:, 1:, :, :, 0] - pd_prob[:, :-1, :, :, 0]) * (label - 0.5) * 2
    reward_val[:, :, :-1, :, 1] = (pd_prob[:, :, 1:, :, 0] - pd_prob[:, :, :-1, :, 0]) * (label - 0.5) * 2
    reward_val[:, :, :, :-1, 2] = (pd_prob[:, :, :, 1:, 0] - pd_prob[:, :, :, :-1, 0]) * (label - 0.5) * 2
    for ib in range(len(n_rec)):
        nw, nt, nv = n_rec[ib]
        n_max = 10
        assert nw < pd_prob.shape[1] or nw >= n_max
        assert nt < pd_prob.shape[2] or nt >= n_max
        assert nv < pd_prob.shape[3] or nv >= n_max
        reward_val[ib, nw:, :, :, 0] = 0
        reward_val[ib, nw + 1:, :, :, :] = 0
        reward_val[ib, :, nt:, :, 1] = 0
        reward_val[ib, :, nt + 1:, :, :] = 0
        reward_val[ib, :, :, nv:, 2] = 0
        reward_val[ib, :, :, nv + 1:, :] = 0
        mask[ib, nw:, :, :, 0] = 0
        mask[ib, nw + 1:, :, :, :] = 0
        mask[ib, :, nt:, :, 1] = 0
        mask[ib, :, nt + 1:, :, :] = 0
        mask[ib, :, :, nv:, 2] = 0
        mask[ib, :, :, nv + 1:, :] = 0
    return reward_val, mask


def pairwise_logistic_loss(scores, labels):
    labels = labels.reshape([-1])
    batch_size, nw, nt, nv, dim = scores.shape
    loss_mat = np.zeros([1, nw, nt, nv, dim], dtype=np.float32)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_scores = scores[pos_mask, :, :, :, :]
    neg_scores = scores[neg_mask, :, :, :, :]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return loss_mat
    pairwise_diffs = pos_scores[:, None, :, :, :, :] - neg_scores[None, :, :, :, :, :]
    loss = np.log(1 + np.exp(-pairwise_diffs))
    loss = np.mean(loss, axis=(0, 1))
    loss_mat[0, :, :, :, :] = loss
    return loss_mat


def get_population_level_reward(p_dict, pd_prob, n_rec, label, phase):
    reward_val = np.zeros(list(pd_prob.shape)[:4] + [4], dtype=np.float32)
    loss_batch = pairwise_logistic_loss(pd_prob, label)
    loss_batch = np.mean(loss_batch, axis=0, keepdims=True)
    loss_mat = p_dict['loss_mat']
    if loss_mat is None:
        loss_mat = np.zeros([1] + list(pd_prob.shape[1:]), dtype=np.float32)
    if phase == 'train':
        loss_mat = loss_mat * (1 - p_dict['w_reward_pop']) + loss_batch * p_dict['w_reward_pop']
        p_dict['loss_mat'] = loss_mat
    else:
        loss_mat = loss_batch
    loss_mat = np.tile(loss_mat, (pd_prob.shape[0], 1, 1, 1, 1))
    reward_val[:, :-1, :, :, 0] = loss_mat[:, :-1, :, :, 0] - loss_mat[:, 1:, :, :, 0]
    reward_val[:, :, :-1, :, 1] = loss_mat[:, :, :-1, :, 0] - loss_mat[:, :, 1:, :, 0]
    reward_val[:, :, :, :-1, 2] = loss_mat[:, :, :, :-1, 0] - loss_mat[:, :, :, 1:, 0]
    for ib in range(len(n_rec)):
        nw, nt, nv = n_rec[ib]
        n_max = 10
        assert nw < pd_prob.shape[1] or nw >= n_max
        assert nt < pd_prob.shape[2] or nt >= n_max
        assert nv < pd_prob.shape[3] or nv >= n_max
        reward_val[ib, nw:, :, :, 0] = 0
        reward_val[ib, nw + 1:, :, :, :] = 0
        reward_val[ib, :, nt:, :, 1] = 0
        reward_val[ib, :, nt + 1:, :, :] = 0
        reward_val[ib, :, :, nv:, 2] = 0
        reward_val[ib, :, :, nv + 1:, :] = 0
    return reward_val, p_dict