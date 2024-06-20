import matplotlib.pyplot as plt
from matplotlib import patches
from data import *


palette = ["red", "green", "blue",
           "cyan", "magenta", "yellow", "black",
           "orange", "deepskyblue", "chocolate", "brown", "lime"]
symbol = ["o", "^", "s", "p", "H", "D", "*", "P", "x", "d", ">", "1"]


def linear_stretch(data, q):
    """

    :param data: raw data
    :param q: stretch power
    :return: stretched data
    """
    if q < 0 or q >= 50:
        raise ValueError("Parameter q is out of range!")
    h = np.percentile(data, 100-q)
    l = np.percentile(data, q)
    data = np.clip(data, a_min=l, a_max=h)
    data = (data - l) * 1.0 / (h - l)
    return data


def draw_gt_picture(dataset_name: str):
    """

    :param dataset_name:
    :return:
    """
    gt = load_dataset(dataset_name, key=2)
    color_map = color_map_dict.get(dataset_name)
    gt_map = np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.uint8) * 255

    cls = np.unique(gt)
    for c in cls:
        if c == 0: continue
        h_idx, w_idx = np.where(gt == c)
        gt_map[h_idx, w_idx, :] = color_map[c-1]

    edge = 1
    gt_map_edge = np.zeros((gt.shape[0]+edge*2, gt.shape[1]+edge*2, 3), dtype=np.uint8)
    gt_map_edge[edge:gt.shape[0]+edge, edge:gt.shape[1]+edge, :] = gt_map
    plt.imsave('./save/gt_map/' + dataset_name + '_gt_map.svg', gt_map_edge)
    plt.imshow(gt_map)
    plt.axis('off')
    plt.show()


def draw_class_bar(dataset_name: str, bar_row, bar_col):
    '''

    :param dataset_name:
    :param bar_row:
    :param bar_col:
    :return:
    '''

    from matplotlib import rcParams
    config = {
        "mathtext.fontset": 'stix',
        "font.family": 'serif',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)

    save_path = './save/class_bar'
    color_map = color_map_dict.get(dataset_name) * 1.0 / 255
    dataset_class = dataset_class_dict.get(dataset_name)
    # assert bar_col * bar_row == color_map.shape[0]

    bar_width = 0.1
    bar_height = 0.05
    bar_y_interval = 0.08

    for c in range(bar_col):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='auto')
        for r in range(bar_row):
            bottom_idx = bar_row - r - 1
            idx = c * bar_row + bottom_idx
            color = color_map[idx, :]
            rect = patches.Rectangle((0.05, bar_y_interval * r + 0.05), bar_width, bar_height, color=color)
            ax.add_patch(rect)
            class_name = dataset_class[idx]
            plt.text(bar_width + 0.06, r * bar_y_interval + 0.05, class_name, fontsize=16)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(os.path.join(save_path, dataset_name, 'class_bar-col=' + str(c) + '.png'), bbox_inches='tight', pad_inches=0.0, dpi=600)


def draw_false_color_picture(dataset_name: str):
    """

    :param dataset_name:
    :return:
    """
    try:
        hsi = load_dataset(dataset_name, key=1)
    except:
        hsi = load_dataset(dataset_name, key=0)
    false_color_bands = false_color_dict.get(dataset_name)
    false_color_picture = hsi[:, :, false_color_bands]
    false_color_picture = false_color_picture.astype(np.float)

    for b in range(false_color_picture.shape[-1]):
        false_color_picture[:, :, b] = linear_stretch(false_color_picture[:, :, b], 2)

    plt.imsave('./save/false_color_picture/' + dataset_name + '_false_color_picture.svg', false_color_picture)
    plt.imshow(false_color_picture)
    plt.axis('off')
    plt.show()


def draw_true_color_picture(dataset_name: str):
    """

    :param dataset_name:
    :return:
    """
    try:
        hsi = load_dataset(dataset_name, key=1)
    except:
        hsi = load_dataset(dataset_name, key=0)
    true_color_bands = true_color_dict.get(dataset_name)
    true_color_picture = hsi[:, :, true_color_bands]
    true_color_picture = true_color_picture.astype(np.float)

    for b in range(true_color_picture.shape[-1]):
        true_color_picture[:, :, b] = linear_stretch(true_color_picture[:, :, b], 2)

    plt.imsave('./save/true_color_picture/' + dataset_name + '_true_color_picture.svg', true_color_picture)
    plt.imshow(true_color_picture)
    plt.axis('off')
    plt.show()


def parameter_analysis(result_file_path: str, v: str, save_path: str):

    f = open(result_file_path, 'r', encoding='utf-8')
    text = f.readlines()
    f.close()
    data = []
    for line in text:
        if line.strip() == '':
            continue
        data.append(line.strip().split('\t'))

    ticks = data[0][1:]
    datasets = []
    oa_max, oa_min = 0, 100
    for i, d in enumerate(data[1:]):
        mean_std = [oa.split('±') for oa in d[1:]]
        mean_std = np.array(mean_std, dtype='float')
        if (mean_std[:, 0] + mean_std[:, 1]).max() > oa_max:
            oa_max = (mean_std[:, 0] + mean_std[:, 1]).max()
        if (mean_std[:, 0] - mean_std[:, 1]).min() < oa_min:
            oa_min = (mean_std[:, 0] - mean_std[:, 1]).min()
        plt.errorbar(x=np.arange(len(ticks)),
                     y=mean_std[:, 0],
                     yerr=mean_std[:, 1],
                     fmt=symbol[i] + '--',
                     color=palette[i],
                     elinewidth=1,
                     capsize=2,
                     ms=6,
                     )
        datasets.append(d[0])
    plt.legend(datasets, loc='lower right')
    plt.xticks(np.arange(len(ticks)), labels=ticks)
    plt.yticks(range(int(oa_min // 5 * 5), int((oa_max // 5 + 2) * 5), 5))
    plt.xlabel(v)
    plt.ylabel("OA(%)")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, v + '.svg'), dpi=600, format='svg')
    plt.show()


def comparison_method_analysis(result_file_path: str, save_path: str):

    f = open(result_file_path, 'r', encoding='utf-8')
    text = f.readlines()
    f.close()
    data = []
    for line in text:
        if line.strip() == '':
            continue
        data.append(line.strip().split('\t'))

    ticks = data[0][1:]
    models = []
    oa_max, oa_min = 0, 100
    for i, d in enumerate(data[1:]):
        mean_std = [oa.split('±') for oa in d[1:]]
        mean_std = np.array(mean_std, dtype='float')
        if (mean_std[:, 0] + mean_std[:, 1]).max() > oa_max:
            oa_max = (mean_std[:, 0] + mean_std[:, 1]).max()
        if (mean_std[:, 0] - mean_std[:, 1]).min() < oa_min:
            oa_min = (mean_std[:, 0] - mean_std[:, 1]).min()
        plt.errorbar(x=np.arange(len(ticks)),
                     y=mean_std[:, 0],
                     yerr=mean_std[:, 1],
                     fmt=symbol[-(i+1)] + '--',
                     color=palette[-(i+1)],
                     elinewidth=1,
                     capsize=2,
                     ms=6,
                     )
        models.append(d[0])
    plt.legend(models, loc='lower right')
    plt.xticks(np.arange(len(ticks)), labels=ticks)
    plt.yticks(range(int(oa_min // 5 * 5), int((oa_max // 5 + 2) * 5), 5))
    plt.xlabel("Training samples per class")
    plt.ylabel("OA(%)")
    plt.grid(True)
    dataset_name = result_file_path.split('/')[-1]
    plt.savefig(os.path.join(save_path, dataset_name.split('.')[0] + '.svg'), dpi=600, format='svg')
    plt.show()


def feature_separability(dataset_name, feature_map, label, save_path, perplexity=30):
    from sklearn import manifold
    n_class = label.max()
    palette_ = color_map_dict.get(dataset_name)
    palette_ = palette_ * 1.0 / 255
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=perplexity)
    row, col, band = dataset_size_dict.get(dataset_name)
    feature_map = feature_map.reshape((row * col, -1))
    label = label.reshape((row * col))
    idx = np.where(label != 0)[0]
    feature_map = feature_map[idx, :]
    label = label[idx]
    X_tsne = tsne.fit_transform(feature_map)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(n_class):
        index = np.where(label == i + 1)
        # print(np.max(index) > X_norm.shape)
        xx1 = X_norm[index, 0]
        yy1 = X_norm[index, 1]
        plt.scatter(xx1, yy1, color=palette_[i].reshape(1, -1))
    plt.savefig(os.path.join(save_path, dataset_name + '_perplexity=' + str(perplexity) + '.svg'), dpi=600,
                format='svg')
    plt.savefig(os.path.join(save_path, dataset_name + '_perplexity=' + str(perplexity) + '.png'), dpi=600,
                bbox_inches='tight')
    plt.show()


def output_classification_map(model, X_PCAMirror, Y, dataset_name='PU', patch_size=9):
    import torch
    from PIL import Image
    row, col, band = dataset_size_dict.get(dataset_name)
    palette_ = color_map_dict.get(dataset_name)
    test_set = np.arange(row * col)
    classification_result = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            y_hat = model(x)
            classification_result.extend(y_hat.cpu().argmax(axis=1).tolist())

    classification_result = np.array(classification_result).reshape((row, col))
    classification_map = palette_[classification_result].reshape((row, col, -1))
    save_path = './save/classification_map'
    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)
    plt.figure()
    plt.imshow(classification_map, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_path, 'SPFormer_' + dataset_name + '_background.png'), bbox_inches='tight', dpi=600)
    classification_map[Y.reshape((row, col)) == 0, :] = np.array([255, 255, 255], dtype='uint8')
    plt.figure()
    plt.imshow(classification_map, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_path, 'SPFormer_' + dataset_name + '_nobackground.png'), bbox_inches='tight', dpi=600)


def calculate_feature_dispersion(dataset_name, feature_map, label):

    perplexities = {'IP': 35, 'PU': 90, 'Salinas': 90}
    perplexity = perplexities.get(dataset_name)

    from sklearn import manifold
    from sklearn.decomposition import PCA
    n_class = label.max()
    row, col, band = dataset_size_dict.get(dataset_name)
    feature_map = feature_map.reshape((row * col, -1))
    label = label.reshape((row * col))
    idx = np.where(label != 0)[0]
    feature_map = feature_map[idx, :]
    label = label[idx]

    # tsne = manifold.TSNE(n_components=n_class, init='pca', perplexity=perplexity)
    # X_tsne = tsne.fit_transform(feature_map)
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)
    # feature_map = X_norm
    # del X_tsne, x_min, x_max

    pca = PCA(n_components=n_class, whiten=True)
    feature_map = pca.fit_transform(feature_map)

    # feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

    class_mean_vector = np.zeros((n_class, feature_map.shape[1]))
    Sb = np.zeros((n_class, n_class))
    for c in range(n_class):
        x_c = feature_map[label == c+1]
        class_mean_vector[c, :] = np.mean(x_c, axis=0)

    for i in range(n_class):
        for j in range(n_class):
            if j == i:
                continue
            class_diff = class_mean_vector[i, :] - class_mean_vector[j, :]
            class_diff = np.mat(class_diff)
            sb = class_diff.transpose() * class_diff
            Sb[i, j] = np.mean(np.abs(sb))

    diversion = Sb.sum() / (n_class * (n_class - 1))
    print('%.2f' % diversion)


def DrawCluster(label, cluster, classcount, id):
    # tsne = manifold.TSNE(n_components=classcount,init='pca')
    # cluster = tsne.fit_transform(cluster)
    x = np.zeros([classcount,classcount])
    y = np.zeros([classcount,classcount])
    for i in range(classcount):
        xx1 = cluster[np.where(label==i+1)]
        x[i,:]=np.mean(xx1,axis=0)
    for i in range(classcount):
        for j in range(classcount):
            if j != i:
                Sb = np.dot(x[i,:] - x[j,:], np.transpose(x[i,:] - x[j,:]))
                y[i,j] = np.mean(np.abs(Sb))

    other = np.zeros([classcount, 1])
    for i in range(classcount):
        a = 0
        for j in range(classcount):
            if i != j:
                a = a + np.abs(y[i, j])
        other[i] = a / (classcount-1)
    # print(other)
    print(id+'----Evaluation of divisibility:', np.mean(other))


if __name__ == '__main__':
    # gt = draw_gt_picture('PU')
    false_color = draw_false_color_picture('Houston')
    # true_color = draw_true_color_picture('PU')

    # file_path = './save/parameter_analysis/lr.txt'
    # v = 'Learning rate'
    # parameter_analysis(result_file_path=file_path, v=v, save_path='./save/parameter_analysis')
    #
    # file_path = './save/comparison_method_analysis/Houston.txt'
    # comparison_method_analysis(result_file_path=file_path, save_path='./save/comparison_method_analysis')

    # feature_map_path = './save/feature_separability'
    # model_name = 'SPFormer'
    # dataset_name = 'Salinas'
    # row, col, band = dataset_size_dict.get(dataset_name)
    # save_path = os.path.join(feature_map_path, model_name, model_name + '_' + dataset_name)
    # feature_map_path = os.path.join(feature_map_path, model_name, model_name + '_' + dataset_name, 'feature_map.npy')
    # feature_map = np.load(feature_map_path)
    # label = load_dataset(dataset_name=dataset_name, key=2)
    # perplexity = [80, 90, 100, 110, 120, 130]
    # for p in perplexity:
    #     feature_separability(dataset_name=dataset_name, feature_map=feature_map, label=label, save_path=save_path,
    #                          perplexity=p)

    # ----- classification map ----- #
    # import torch
    # # from loop_train_test import Model_EMA
    # from loop_train_test import get_model
    # dataset_name = 'Salinas'
    # model_path = './save/models'
    # dim = 40
    # patch_size = 13
    # row, col, band = dataset_size_dict.get(dataset_name)
    # n_class = len(dataset_class_dict.get(dataset_name))
    # model_files = os.listdir(model_path)
    # state_dicts = []
    # for m in model_files:
    #     if m.split('.')[-1] == 'pt':
    #         print(m + ' is pt file, continue!')
    #         state_dicts.append(torch.load(os.path.join(model_path, m)))
    #
    # X_PCAMirror, Y, _ = HSI_LazyProcessing(dataset_name=dataset_name, no_processing=True)
    # # X_PCAMirror-Padding
    # patch_radius = patch_size // 2
    # X_PCAMirror = mirror_concatenate(X_PCAMirror)
    # b = default_mirror_width - patch_radius
    # X_PCAMirror = X_PCAMirror[b: -b, b: -b]
    # # model = Model_EMA(n_class=n_class, state_dicts=state_dicts, raw_dim=band, dim=dim, patch_size=patch_size)
    # model = get_model(n_class=n_class, dim=dim, patch_size=patch_size)
    # output_classification_map(model, X_PCAMirror, Y, dataset_name=dataset_name, patch_size=patch_size)

    # ----- draw class bar ----- #
    # draw_class_bar(dataset_name='PU', bar_row=9, bar_col=1)

    # ----- analyse feature separability ----- #
    # dataset = ['IP', 'PU', 'Salinas']
    # path = './save/feature_separability'
    # ablations = ['SPFormer_ablation_MHSP_CTM',
    #              'SPFormer_ablation_MHSP',
    #              'SPFormer_ablation_CTM',
    #              'SPFormer']
    # for dataset_name in dataset:
    #     print()
    #     print(dataset_name)
    #     label = load_dataset(dataset_name=dataset_name, key=2)
    #     for ablation in ablations:
    #         ablation_dir = ablation + '_' + dataset_name
    #         feature_map = np.load(os.path.join(path, ablation, ablation_dir, 'feature_map.npy'))
    #         calculate_feature_dispersion(dataset_name=dataset_name, feature_map=feature_map, label=label)