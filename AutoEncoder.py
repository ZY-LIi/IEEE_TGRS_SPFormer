import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data import load_dataset
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=(1, 1), bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def generate_batch(X, batch_size=64, shuffle=True):
    '''

    :param X:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    row, col, band = X.shape
    X = X.reshape((row * col, -1)).astype(np.float)
    X = X - np.mean(X, axis=0)
    num_samples = row * col
    sequence = np.arange(num_samples)

    if shuffle:
        random_state = np.random.RandomState()
        random_state.shuffle(sequence)

    for i in range(0, num_samples, batch_size):
        # batch_i represents the i-th element in current batch
        batch_i = sequence[np.arange(i, min(num_samples, i + batch_size))]
        patches = np.expand_dims(X[batch_i, :], axis=-1)

        yield np.expand_dims(patches, axis=-1)


def train_ae(X, batch_size=64, epoches=120, lr=1e-2, out_channel=32, decay_step=1000):

    row, col, band = X.shape
    model = AutoEncoder(in_channels=band, out_channels=out_channel)
    model = model.to(device)

    train_loss_list = []

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    t = time.time()
    for epoch in range(epoches):
        data_loader = generate_batch(X, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        for step, x in enumerate(data_loader):

            x = torch.Tensor(x)
            x = x.to(device)
            y_hat = model(x)
            optimizer.zero_grad()
            loss = loss_func(y_hat, x)
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            if (step + 1) % decay_step == 0:
                lr_scheduler.step()
                print('epoch: %d, step_loss: %.5f, lr: %.5f'
                      % (epoch, loss.item(), lr_scheduler.get_last_lr()[0]))
        train_loss_list.append(epoch_loss / (step + 1))
        if epoch >= epoches - 10:
            torch.save(model.state_dict(), './save/autoencoder/epoch-' + str(epoch) +
                       'train_loss-' + str(epoch_loss / (step + 1)) + '.pt')

    print("Training Time: {}s".format(time.time() - t))
    plt.plot(train_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.show()


def resolve_hp(hp: dict):
    return hp.get('batch_size'), hp.get('lr'), hp.get('epoch'), \
           hp.get('out_channel'), hp.get('decay_step')


def delete_pt_file():
    for file in os.listdir('./save/autoencoder'):
        if file.endswith('.pt'):
            os.remove(os.path.join('./save/autoencoder', file))


def IP_experiment(hp: dict):
    X = load_dataset(dataset_name='IP', key=1)
    batch_size, lr, epoch, out_channel, decay_step = resolve_hp(hp)
    train_ae(X, batch_size=batch_size, lr=lr, epoches=epoch, out_channel=out_channel, decay_step=decay_step)


def PU_experiment(hp: dict):
    X = load_dataset(dataset_name='PU', key=1)
    batch_size, lr, epoch, out_channel, decay_step = resolve_hp(hp)
    train_ae(X, batch_size=batch_size, lr=lr, epoches=epoch, out_channel=out_channel, decay_step=decay_step)


def Salinas_experiment(hp: dict):
    X = load_dataset(dataset_name='Salinas', key=1)
    batch_size, lr, epoch, out_channel, decay_step = resolve_hp(hp)
    train_ae(X, batch_size=batch_size, lr=lr, epoches=epoch, out_channel=out_channel, decay_step=decay_step)


def HU_experiment(hp: dict):
    X = load_dataset(dataset_name='Houston', key=1)
    batch_size, lr, epoch, out_channel, decay_step = resolve_hp(hp)
    train_ae(X, batch_size=batch_size, lr=lr, epoches=epoch, out_channel=out_channel, decay_step=decay_step)


if __name__ == '__main__':
    delete_pt_file()
    hyperparameter_pu = {
        'batch_size': 64,
        'lr': 1e-2,
        'epoch': 50,
        'out_channel': 32,
        'decay_step': 800,
    }
    PU_experiment(hp=hyperparameter_pu)

    hyperparameter_ip = {
        'batch_size': 128,
        'lr': 1e-2,
        'epoch': 20,
        'out_channel': 40,
        'decay_step': 20,
    }
    IP_experiment(hp=hyperparameter_ip)

    hyperparameter_salinas = {
        'batch_size': 128,
        'lr': 1e-2,
        'epoch': 50,
        'out_channel': 40,
        'decay_step': 200,
    }
    Salinas_experiment(hp=hyperparameter_salinas)

    hyperparameter_hu = {
        'batch_size': 128,
        'lr': 1e-2,
        'epoch': 50,
        'out_channel': 32,
        'decay_step': 1000,
    }
    HU_experiment(hp=hyperparameter_hu)
