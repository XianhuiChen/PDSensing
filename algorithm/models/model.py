import torch
import torch.nn as nn
import torch.nn.functional as F

class RecordModelResNet(nn.Module):
    def __init__(self, num_filters=64, kernel_size=3, dropout_prob=0.2, num_channels=6, rec_length=3000):
        super(RecordModelResNet, self).__init__()
        self.rec_length = rec_length
        self.mapping = nn.Conv1d(num_channels, num_filters, kernel_size=kernel_size, padding=1)
        self.conv_layers_x1 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.conv_layers_x2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
        )
        self.maxpooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.adaptivepooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.mapping(x)
        if self.rec_length > 2000:
            num_cov = 5
        elif self.rec_length >= 500:
            num_cov = 4
        else:
            num_cov = 2
        for i in range(num_cov):
            f = self.conv_layers_x1(x)
            x = self.maxpooling(x)
            f = f + x
        x = f
        for i in range(num_cov):
            f = self.conv_layers_x2(x)
            x = self.maxpooling(x)
            f = f + x
        x = self.adaptivepooling(f)
        return x


class Predictor(nn.Module):
    def __init__(self, num_filters):
        super(Predictor, self).__init__()
        self.num_filters = num_filters
        self.tapping_model = RecordModelResNet(num_channels=1, num_filters=num_filters, rec_length=300)
        self.tapping_accel_model = RecordModelResNet(num_channels=3, num_filters=num_filters, rec_length=300)
        self.voice_model = RecordModelResNet(num_channels=40, num_filters=num_filters, rec_length=250)
        self.walking_model = RecordModelResNet(num_channels=6, num_filters=num_filters, rec_length=300)

        self.fc_walking_out = nn.Linear(num_filters, num_filters)
        self.fc_walking_rest = nn.Linear(num_filters, num_filters)
        self.fc_walking_ret = nn.Linear(num_filters, num_filters)
        self.fc_tapping = nn.Linear(num_filters, num_filters)
        self.fc_tapping_accel = nn.Linear(num_filters, num_filters)
        self.fc_pd = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()
        self.maxpooling = nn.AdaptiveMaxPool1d(1)

    def get_walking_embedding(self, x):
        shape = x.size()
        x = x.reshape([shape[0] * shape[1] * shape[2], shape[3], shape[4]]).transpose(1, 2)
        x = self.walking_model(x)
        x = x.reshape([shape[0], shape[1], shape[2], self.num_filters])
        x_out = self.fc_walking_out(x[:, :, 0, :].contiguous())
        x_rest = self.fc_walking_rest(x[:, :, 1, :].contiguous())
        x_ret = self.fc_walking_ret(x[:, :, 2, :].contiguous())
        return x_out, x_rest, x_ret

    def get_voice_embedding(self, x):
        shape = x.size()
        x = x.reshape([shape[0] * shape[1], shape[2], shape[3]]).transpose(1, 2)
        x = self.voice_model(x)
        x = x.reshape([shape[0], shape[1], self.num_filters])
        x = self.fc_tapping(x)
        return x

    def get_tapping_embedding(self, x):
        shape = x.size()
        x = x.reshape([shape[0] * shape[1], shape[2], 1]).transpose(1, 2)
        x = self.tapping_model(x)
        x = x.reshape([shape[0], shape[1], self.num_filters])
        x = self.fc_tapping(x)
        return x

    def get_tapping_accel_embedding(self, x):
        shape = x.size()
        x = x.reshape([shape[0] * shape[1], shape[2], shape[3]]).transpose(1, 2)
        x = self.tapping_accel_model(x)
        x = x.reshape([shape[0], shape[1], self.num_filters])
        x = self.fc_tapping_accel(x)
        return x

    def merge_embedding(self, xs, vis_prediction=True):
        if vis_prediction:
            x = torch.cat(xs, dim=1).transpose(1, 2)
        else:
            walking_out_embedding, walking_rest_embedding, walking_ret_embedding, tapping_embedding, tapping_accel_embedding, voice_embedding = xs
            time_walking_embedding = []
            for i in range(walking_out_embedding.size(1)):
                x = torch.cat([walking_out_embedding[:, :i + 1].contiguous(),
                               walking_rest_embedding[:, :i + 1].contiguous(),
                               walking_ret_embedding[:, :i + 1].contiguous()], dim=1).transpose(1, 2)
                x = self.maxpooling(x)
                time_walking_embedding.append(x)
            time_walking_embedding = torch.cat(time_walking_embedding, dim=2).transpose(1, 2)
            shape = time_walking_embedding.size()
            num_walking = shape[1]
            time_walking_embedding = time_walking_embedding.view([shape[0], shape[1], 1, 1, shape[2]])

            time_tapping_embedding = []
            for i in range(tapping_embedding.size(1)):
                x = torch.cat([tapping_embedding[:, :i + 1], tapping_accel_embedding[:, :i + 1]], dim=1).transpose(1, 2)
                x = self.maxpooling(x)
                time_tapping_embedding.append(x)
            time_tapping_embedding = torch.cat(time_tapping_embedding, dim=2).transpose(1, 2)
            shape = time_tapping_embedding.size()
            num_tapping = shape[1]
            time_tapping_embedding = time_tapping_embedding.view([shape[0], 1, shape[1], 1, shape[2]])

            time_voice_embedding = []
            for i in range(voice_embedding.size(1)):
                x = voice_embedding[:, :i + 1].transpose(1, 2)
                x = self.maxpooling(x)
                time_voice_embedding.append(x)
            time_voice_embedding = torch.cat(time_voice_embedding, dim=2).transpose(1, 2)
            shape = time_voice_embedding.size()
            num_voice = shape[1]
            time_voice_embedding = time_voice_embedding.view([shape[0], 1, 1, shape[1], shape[2]])

            time_walking_embedding = time_walking_embedding.repeat(1, 1, num_tapping, num_voice, 1)
            time_tapping_embedding = time_tapping_embedding.repeat(1, num_walking, 1, num_voice, 1)
            time_voice_embedding = time_voice_embedding.repeat(1, num_walking, num_tapping, 1, 1)

            bwtv_shape = time_walking_embedding.size()
            time_walking_embedding = time_walking_embedding.view((-1, bwtv_shape[-1], 1))
            time_tapping_embedding = time_tapping_embedding.view((-1, bwtv_shape[-1], 1))
            time_voice_embedding = time_voice_embedding.view((-1, bwtv_shape[-1], 1))

            time_embedding = torch.cat([time_walking_embedding, time_tapping_embedding, time_voice_embedding], dim=2)
            time_embedding = self.maxpooling(time_embedding)
            x = time_embedding.view(bwtv_shape)
        return x

    def get_merged_embedding(self, walking_data, tapping_data, tapping_accel_data, voice_data, vis_prediction=True):
        walking_out_embedding, walking_rest_embedding, walking_ret_embedding = self.get_walking_embedding(walking_data)
        tapping_embedding = self.get_tapping_embedding(tapping_data)
        tapping_accel_embedding = self.get_tapping_accel_embedding(tapping_accel_data)
        voice_embedding = self.get_voice_embedding(voice_data)
        return self.merge_embedding(
            [walking_out_embedding, walking_rest_embedding, walking_ret_embedding, tapping_embedding,
             tapping_accel_embedding, voice_embedding],
            vis_prediction=vis_prediction
        )

    def estimate_risk(self, merge_embedding):
        prob = self.sigmoid(self.fc_pd(merge_embedding))
        return prob

    def forward(self, walking_data, tapping_data, tapping_accel_data, voice_data, vis_prediction=False):
        merge_embedding = self.get_merged_embedding(walking_data, tapping_data, tapping_accel_data, voice_data,
                                                    vis_prediction=vis_prediction)
        pred = self.estimate_risk(merge_embedding)
        return merge_embedding, pred


class Critic(nn.Module):
    def __init__(self, num_filters):
        super(Critic, self).__init__()
        self.num_filters = num_filters
        self.fc_reward = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 4)
        )

    def forward(self, merge_embedding):
        reward = self.fc_reward(merge_embedding)
        return reward


class Actor(nn.Module):
    def __init__(self, num_filters):
        super(Actor, self).__init__()
        self.num_filters = num_filters
        self.fc_action = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 4)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, merge_embedding):
        action = self.fc_action(merge_embedding)
        action = self.softmax(action)
        return action