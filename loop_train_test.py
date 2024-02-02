from data import *
import time
import torch
torch.cuda.current_device()
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
from loss_function import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(n_class, autoencoder_path, dim=40):
    '''
    This function needs to be implemented according to the certain situation.
    :return:
    '''

    from SelfPoolingTransformer import SelfPoolingTransformer
    model = SelfPoolingTransformer(dim=dim, autoencoder_path=autoencoder_path, n_class=n_class)
    return model


def Model_EMA(n_class, state_dicts: list, dim=40, patch_size=13, strict=True):

    model = get_model(n_class=n_class, dim=dim)
    model_aux = get_model(n_class=n_class, dim=dim)
    temp_model = get_model(n_class=n_class, dim=dim)
    for param, param_aux in zip(model.parameters(), model_aux.parameters()):
        param_aux.data.copy_(param)
        param_aux.requires_grad = False
    n = len(state_dicts)
    for state_dict in state_dicts:
        temp_model.load_state_dict(state_dict, strict=strict)
        for param1, param2 in zip(model.parameters(), temp_model.parameters()):
            param1.data += 1/n * param2.data
    for param, param_aux in zip(model.parameters(), model_aux.parameters()):
        param.data -= param_aux.data
        param.requires_grad = False
    return model


def train(model, X_PCAMirror, Y, train_set, label=None, dataset_name='PU', train_num=5, patch_size=9, batch_size=64,
          epoches=120, lr=1e-2, shuffle=True):

    train_loss_list = []

    gamma = 0.5 if train_num <= 5 else 0.0
    loss_func = FocalLoss(n_class=len(dataset_class_dict.get(dataset_name)), alpha=1, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # ---------- optimizer and loss implementation ---------- #

    model.train()
    t = time.time()
    epoches = epoches + 10
    for epoch in range(epoches):
        train_loader = generate_batch(train_set, X_PCAMirror, Y, dataset_name=dataset_name, patch_size=patch_size,
                                      batch_size=batch_size, shuffle=shuffle, augment=True)

        epoch_loss = 0
        for step, (x, y) in enumerate(train_loader):

            x, y = torch.Tensor(x), torch.Tensor(y).type(torch.long)
            x, y = x.to(device), y.squeeze(1).to(device)
            y_hat = model(x)

            # ---------- loss back-propagation and parameter optimization ---------- #

            optimizer.zero_grad()
            loss = loss_func(y_hat, y)
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        train_loss_list.append(epoch_loss / (step + 1))
        lr_scheduler.step()
        if epoch >= epoches - 3:
            torch.save(model.state_dict(), './save/models/epoch-' + str(epoch) + 'train_loss-' + str(epoch_loss / (step + 1)) + '.pt')


    plt.plot(train_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.show()
    print('Training model consumes %.2f seconds' % (time.time() - t))


def test(model, X_PCAMirror, Y, test_set, dataset_name='PU', patch_size=9):

    y_pred = []
    y_true = []
    model = model.to(device)
    model.eval()
    t = time.time()
    with torch.no_grad():
        test_loader = generate_batch(test_set, X_PCAMirror, Y,
                                     dataset_name=dataset_name,
                                     patch_size=patch_size, batch_size=64,
                                     shuffle=False, mode='test')
        for x, y in test_loader:
            x = torch.Tensor(x)
            x = x.to(device)
            y_hat = model(x)
            y_pred.extend(y_hat.cpu().argmax(axis=1).tolist())
            y_true.extend(y.reshape(-1).tolist())

    print('Testing model consumes %.2f seconds' % (time.time() - t))

    class_dict = dataset_class_dict.get(dataset_name)
    y_true = [class_dict[i] for i in y_true]
    y_pred = [class_dict[i] for i in y_pred]
    cm = confusion_matrix(y_true, y_pred, labels=class_dict)
    # print(cm)
    oa = accuracy_score(y_true, y_pred)
    mask = np.eye(cm.shape[0])
    acc_for_each_class = np.sum(cm * mask, axis=1) / np.sum(cm, axis=1)
    aa = acc_for_each_class.mean()
    kappa = cohen_kappa_score(y_true, y_pred)
    return cm, oa, aa, kappa, acc_for_each_class


def loop_train_test(dataset_name, run_times, num_PC, train_num, patch_size, batch_size, lr, epoch, autoencoder_path, ema=False):

    print('>' * 10, 'Loading Data', '<' * 10)
    X_PCAMirror, Y, [row, col, band] = HSI_LazyProcessing(dataset_name=dataset_name)

    print('>' * 10, 'Start Experiment', '<' * 10)
    n_class = len(dataset_class_dict.get(dataset_name))

    # X_PCAMirror-Padding
    patch_radius = patch_size // 2
    X_PCAMirror = mirror_concatenate(X_PCAMirror)
    b = default_mirror_width - patch_radius
    X_PCAMirror = X_PCAMirror[b: -b, b: -b]

    metrics = []
    seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]    # same with DCFSL
    for seed in seeds:
        seed_dir = os.path.join('./save/result/', dataset_name, str(train_num), str(seed))
        try:
            for file in os.listdir(seed_dir):
                os.remove(os.path.join(seed_dir, file))
        except:
            pass
        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)

        train_set, train_step, test_set, test_step = \
            split_train_test_set(Y, dataset_name=dataset_name,
                                 train_num=train_num, batch_size=batch_size, seed=seed)

        metric = []

        print('Seed: %d' % seed)
        for run in range(run_times):
            for file in os.listdir('./save/models'):
                os.remove(os.path.join('./save/models', file))
            print('Round: %d' % run)
            model = get_model(n_class, autoencoder_path=autoencoder_path, dim=num_PC)
            print('Model Params: ' + str(sum(torch.numel(parameter) for parameter in model.parameters())))
            model = model.to(device)
            train(model, X_PCAMirror, Y, train_set, dataset_name=dataset_name, train_num=train_num, patch_size=patch_size,
                  batch_size=batch_size, epoches=epoch, lr=lr)

            print('>' * 10, 'Start Testing', '<' * 10)
            model_path = './save/models'
            model_files = os.listdir(model_path)
            state_dicts = []
            for m in model_files:
                state_dicts.append(torch.load(os.path.join(model_path, m)))

            if ema:
                model = Model_EMA(n_class=n_class, state_dicts=state_dicts, dim=num_PC, patch_size=patch_size)
            else:
                model.load_state_dict(state_dicts[-1])

            cm, oa, aa, kappa, acc_for_each_class = test(model, X_PCAMirror, Y, test_set,
                                                         dataset_name=dataset_name, patch_size=patch_size)
            # print(cm)
            print('OA: %.4f' % oa)
            print('AA: %.4f' % aa)
            print('Kappa: %.4f' % kappa)
            metric.append([oa, aa, kappa])

            fn = os.path.join('./save/result/', dataset_name, str(train_num), str(seed), 'confusion_matrix_' + str(run) + '.npy')
            np.save(fn, cm)
            fn = os.path.join('./save/result/', dataset_name, str(train_num), str(seed), 'acc_for_each_class_' + str(run) + '.npy')
            np.save(fn, acc_for_each_class)
        metrics.append(metric)
    fn = os.path.join('./save/result/', dataset_name, str(train_num), 'metrics.npy')
    np.save(fn, np.array(metrics))
    metrics = np.array(metrics)
    print(np.mean(metrics, axis=0))


def lazy_test(model, X_PCAMirror, Y, dataset_name='PU', patch_size=9):
    '''
    only output each pixel's feature and then form a feature map
    :param model:
    :param X_PCAMirror:
    :param dataset_name:
    :param patch_size:
    :return:
    '''

    row, col, band = dataset_size_dict.get(dataset_name)
    test_set = np.arange(row * col)
    output_feature_map = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        test_loader = generate_batch(test_set, X_PCAMirror, Y,
                                     dataset_name=dataset_name,
                                     patch_size=patch_size, batch_size=64,
                                     shuffle=False, mode='test')
        for x, _ in test_loader:
            x = torch.Tensor(x)
            x = x.to(device)
            f = model(x)
            f = f.to("cpu").numpy()
            output_feature_map.append(f)

    output_feature_map = np.concatenate(output_feature_map, axis=0)
    print(output_feature_map.shape)
    output_feature_map = output_feature_map.reshape((row, col, -1))
    return output_feature_map

