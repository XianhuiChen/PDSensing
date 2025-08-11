import os
import torch
import sklearn.metrics

def load_model(p_dict, model, optimizer, key='predictor'):
    path = os.path.join(p_dict['model_folder'], f'PDSensing_best.{key}.ckpt')
    chkpoint = torch.load(path, map_location='cuda')
    model[key].load_state_dict(chkpoint['model'][key])
    optimizer[key].load_state_dict(chkpoint['optimizer'][key])
    p_dict['epoch'] = chkpoint['epoch']
    return p_dict, model, optimizer
def save_model(p_dict, model, optimizer, key):
    save_path = os.path.join(p_dict['model_folder'], f'PDSensing_best.{key}.ckpt')
    print(f'save to {save_path}')
    torch.save({
        'model': {
            key: model[key].state_dict()
            },
        'epoch': p_dict['epoch'],
        'optimizer': {
            key: optimizer[key].state_dict(),
            },
        },
        save_path)

def compute_auc(label_list, pred_list):
    fpr, tpr, thr = sklearn.metrics.roc_curve(label_list, pred_list)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc

