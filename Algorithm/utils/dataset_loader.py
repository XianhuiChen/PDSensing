import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class WTVDataset(Dataset):
    def __init__(self, opt, pid_rec_dict, phase='train'):
        self.opt = opt
        self.pid_list = []

        for pid in pid_rec_dict:
            num_walking = pid_rec_dict[pid]['num_record_outbound']
            num_tapping = pid_rec_dict[pid]['num_record_tapping']
            num_voice = pid_rec_dict[pid]['num_record']
            if num_walking + num_tapping + num_voice > 0:
                self.pid_list.append({
                    'pid': pid,
                    'num_walking': num_walking,
                    'num_tapping': num_tapping,
                    'num_voice': num_voice,
                })
        print(f'There are {len(self.pid_list)} {phase} patients with label')
        self.phase = phase
        self.max_walking_timestamps = 3000
        self.max_tapping_timestamps = 300
        self.max_tapping_accel_timestamps = 300
        self.max_voice_timestamps = 1000
        if self.phase == 'train':
            self.max_walking_record = 10
            self.max_tapping_record = 9
            self.max_voice_record = 5
        elif phase == 'valid':
            self.max_walking_record = 20
            self.max_tapping_record = 18
            self.max_voice_record = 10
        elif phase == 'test':
            self.max_walking_record = 20
            self.max_tapping_record = 18
            self.max_voice_record = 10


    def __len__(self):
        return len(self.pid_list)

    def load_data(self, healthCode):
        healthcode_dir = os.path.join(self.opt.data_dir, 'raw_data', healthCode)
        with open(os.path.join(healthcode_dir, 'pid.json'), 'r') as file:
            pid = json.load(file)

        walking_timestamps = [pid['walking']['time_stamp_outbound'], pid['walking']['time_stamp_rest'],
                              pid['walking']['time_stamp_return']] if 'walking' in pid else [0, 0, 0]
        tapping_timestamps = [pid['tapping']['time_stamp_tapping'],
                              pid['tapping']['time_stamp_accel']] if 'tapping' in pid else [0, 0]

        if 'label' not in pid:
            print('-' * 80)
            print(f'Missing label for {healthCode}')
            print('-' * 80)

        data = {'label': pid.get('label', 0)}

        if pid.get('walking', {}).get('num_record_outbound', 0) > 0:
            walking_data = np.load(os.path.join(healthcode_dir, 'walking.npy'), allow_pickle=True).item()
            data['walking'] = {
                "accel_outbound": walking_data.get("accel_outbound"),
                "rotation_outbound": walking_data.get("rotation_outbound"),
                "timestamp_outbound": walking_timestamps[0],
                "accel_rest": walking_data.get("accel_rest"),
                "rotation_rest": walking_data.get("rotation_rest"),
                "timestamp_rest": walking_timestamps[1],
                "accel_return": walking_data.get("accel_return"),
                "rotation_return": walking_data.get("rotation_return"),
                "timestamp_return": walking_timestamps[2]
            }

        if pid.get('voice', {}).get('num_record', 0) > 0:
            voice_preprocessed_file = os.path.join(healthcode_dir, 'voice_preprocessed.npy')
            if os.path.exists(voice_preprocessed_file):
                processed_audio = np.load(voice_preprocessed_file)
            else:
                voice_data = np.load(os.path.join(healthcode_dir, 'voice.npy'), allow_pickle=True).item()
                raw_audio = voice_data.get("raw_audio")
                processed_audio = []
                for audio in raw_audio:
                    try:
                        rec = np.zeros([self.max_voice_timestamps, 40])
                    except:
                        rec = np.zeros([self.max_voice_timestamps, 40])
                    if len(rec) < self.max_voice_timestamps:
                        rec = np.concatenate((rec, np.zeros([self.max_voice_timestamps, 40])), axis=0)
                    rec = rec[:self.max_voice_timestamps]
                    processed_audio.append(rec)
                processed_audio = np.array(processed_audio)
                np.save(voice_preprocessed_file, processed_audio)
                print(f'{healthCode} voice preprocessed data saved: {processed_audio.shape}')

            data['voice'] = {"processed_audio": processed_audio}

        if pid.get('tapping', {}).get('num_record_tapping', 0) > 0:
            tapping_data = np.load(os.path.join(healthcode_dir, 'tapping.npy'), allow_pickle=True).item()
            tapping_sample = tapping_data.get("tapping")
            tapping_accel = tapping_data.get("accel")
            data['tapping'] = {
                "tapping": tapping_sample,
                "timestamp_tapping": tapping_timestamps[0],
                "accel": tapping_accel,
                "timestamp_accel": tapping_timestamps[1]
            }

        return data

    def preprocess_walking(self, data):
        if 'walking' not in data:
            return np.zeros([self.max_walking_record, 3, self.max_walking_timestamps, 6]), 0
        walking_dict = data['walking']

        walking_data = []
        for i in range(len(walking_dict['accel_outbound'])):
            record_data = []
            for mode in ['outbound', 'rest', 'return']:
                accel = np.array(walking_dict[f'accel_{mode}'][i])
                rotation = np.array(walking_dict[f'rotation_{mode}'][i])
                try:
                    accel_rotation = np.concatenate((accel, rotation), axis=1)
                    padding = np.zeros([self.max_walking_timestamps, accel_rotation.shape[1]])
                    accel_rotation_padding = np.concatenate((accel_rotation, padding), axis=0)[
                                             :self.max_walking_timestamps]
                except:
                    accel_rotation_padding = np.zeros([self.max_walking_timestamps, 6])
                record_data.append(accel_rotation_padding)
            walking_data.append(record_data)
        n_walking = len(walking_data)
        if self.max_walking_record > 1:
            if len(walking_data) > self.max_walking_record:
                walking_data = self.random_selection([walking_data], self.max_walking_record)[0]
            else:
                while len(walking_data) < self.max_walking_record:
                    walking_data.append(np.zeros([3, self.max_walking_timestamps, 6]))

        return np.array(walking_data), min(n_walking, len(walking_data))

    def random_selection(self, data_list, n):
        ids = list(range(len(data_list[0])))
        if self.phase == 'train':
            np.random.shuffle(ids)
        ids = sorted(ids[:n])
        return [[data[i] for i in ids] for data in data_list]

    def preprocess_voice(self, data):
        if 'voice' not in data:
            return np.zeros([self.max_voice_record, self.max_voice_timestamps, 40]), 0
        voice_dict = data['voice']
        voice_data = []
        for i in range(len(voice_dict['processed_audio'])):
            record_data = voice_dict['processed_audio'][i]
            try:
                padding = np.zeros([self.max_voice_timestamps, record_data.shape[1]])
                voice_padding = np.concatenate((record_data, padding), axis=0)[:self.max_voice_timestamps]
            except:
                voice_padding = np.zeros([self.max_voice_timestamps, 40])
            voice_data.append(voice_padding)

        n_voice = len(voice_data)
        if self.max_voice_record > 1:
            if len(voice_data) > self.max_voice_record:
                voice_data = self.random_selection([voice_data], self.max_voice_record)[0]
            else:
                while len(voice_data) < self.max_voice_record:
                    voice_data.append(np.zeros([self.max_voice_timestamps, 40]))
        return np.array(voice_data), min(n_voice, len(voice_data))

    def preprocess_tapping(self, data):
        if 'tapping' not in data:
            return np.zeros([self.max_tapping_record, self.max_tapping_timestamps]), np.zeros(
                [self.max_tapping_record, self.max_tapping_accel_timestamps, 3]), 0
        try:
            tapping_dict = data['tapping']

            tapping_data, accel_data = [], []
            assert len(tapping_dict['tapping']) == len(tapping_dict['accel'])
            for i in range(len(tapping_dict['tapping'])):
                tapping = np.array(tapping_dict['tapping'][i])
                padding = np.zeros([self.max_tapping_timestamps])
                tapping_padding = np.concatenate([tapping, padding], axis=0)[:self.max_tapping_timestamps]

                accel = np.array(tapping_dict['accel'][i])
                padding = np.zeros([self.max_tapping_accel_timestamps, accel.shape[1]])
                accel_padding = np.concatenate([accel, padding], axis=0)[:self.max_tapping_accel_timestamps]
                tapping_data.append(tapping_padding)
                accel_data.append(accel_padding)

            n_tapping = len(tapping_data)
            if self.max_tapping_record > 1:
                if len(tapping_data) > self.max_tapping_record:
                    tapping_data, accel_data = self.random_selection([tapping_data, accel_data],
                                                                     self.max_tapping_record)
                else:
                    while len(tapping_data) < self.max_tapping_record:
                        tapping_data.append(np.zeros([self.max_tapping_timestamps]))
                        accel_data.append(np.zeros([self.max_tapping_accel_timestamps, 3]))
            return np.array(tapping_data), np.array(accel_data), min(n_tapping, len(tapping_data))
        except:
            return np.zeros([self.max_tapping_record, self.max_tapping_timestamps]), np.zeros(
                [self.max_tapping_record, self.max_tapping_accel_timestamps, 3]), 0

    def padding_zero(self, data):
        padding = np.zeros(data[:1].shape)
        data = np.concatenate((padding, data), axis=0)
        return data

    def __getitem__(self, idx):
        rec_dict = self.pid_list[idx]
        pid = rec_dict['pid']
        num_walking = rec_dict['num_walking']
        num_tapping = rec_dict['num_tapping']
        num_voice = rec_dict['num_voice']

        data = self.load_data(pid)
        n_max = 100
        walking_data, num_walking = self.preprocess_walking(data)
        tapping_data, tapping_accel_data, num_tapping = self.preprocess_tapping(data)
        voice_data, num_voice = self.preprocess_voice(data)

        walking_data = walking_data[:n_max]
        tapping_data = tapping_data[:n_max]
        tapping_accel_data = tapping_accel_data[:n_max]
        voice_data = voice_data[:n_max]

        walking_data = self.padding_zero(walking_data)
        tapping_data = self.padding_zero(tapping_data)
        tapping_accel_data = self.padding_zero(tapping_accel_data)
        voice_data = self.padding_zero(voice_data)

        return pid, \
            data['label'], \
            walking_data.astype(np.float32), tapping_data.astype(np.float32), tapping_accel_data.astype(np.float32), voice_data.astype(np.float32), \
            np.array([num_walking, num_tapping, num_voice])