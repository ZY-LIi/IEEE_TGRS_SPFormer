import scipy.io as sio
import numpy as np
import cv2

dataset_path = {

    'PU': {'corrected': './datasets/PU/PaviaU.mat',
           'gt': './datasets/PU/PaviaU_gt.mat'},

    'IP': {'corrected': './datasets/IP/Indian_pines_corrected.mat',
           'gt': './datasets/IP/Indian_pines_gt.mat'},

    'Salinas': {'raw': './datasets/Salinas/Salinas.mat',
                'corrected': './datasets/Salinas/Salinas_corrected.mat',
                'gt': './datasets/Salinas/Salinas_gt.mat'},

    'KSC': {'corrected': './datasets/KSC/KSC.mat',
            'gt': './datasets/KSC/KSC_gt.mat'},

    'Botswana': {'corrected': './datasets/Botswana/Botswana.mat',
                 'gt': './datasets/Botswana/Botswana_gt.mat'},

    'ZY_HHK': {'corrected': './datasets/ZY_HHK/ZY_hhk.mat',
            'gt': './datasets/ZY_HHK/ZY_hhk_gt.mat'},

    'Houston': {'corrected': './datasets/Houston/Houston.mat',
                'gt': './datasets/Houston/Houston_gt.mat'}

}

dataset_name_dict = {
    'PU': 'PaviaU',
    'PC': 'Pavia',
    'IP': 'Indian_pines',
    'Salinas': 'Salinas',
    'KSC': 'KSC',
    'Botswana': 'Botswana',
    'HHK': 'HuangHeKou',
    'Houston': 'Houston_University'
}

dataset_size_dict = {
    'PU': [610, 340, 103],
    'IP': [145, 145, 200],
    'Salinas': [512, 217, 204],
    'KSC': [512, 614, 176],
    'Botswana': [1476, 256, 145],
    'ZY_HHK': [1147, 1600, 119],
    'Houston': [349, 1905, 144],
}

dataset_class_dict = {

    'PU': ["Asphalt", "Meadows", "Gravel", "Trees",
           "Painted metal sheets", "Bare Soil", "Bitumen",
           "Self-Blocking Bricks", "Shadows"],

    'IP': ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
           "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
           "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"],

    'Salinas': ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
                "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
                "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"],

    'KSC': ["Scrub", "Willow swamp", "Cabbage palm hammock", "Cabbage palm/oak hammock",
            "Slash pine", "Oak/broadleaf hammock", "Hardwood swamp", "Graminoid marsh",
            "Spartina marsh", "Cattail marsh", "Salt marsh", "Mud flats", "Wate"],

    'Botswana': ["Water", "Hippo grass", "Floodplain grasses 1", "Floodplain grasses 2", "Reeds", "Riparian",
                 "Firescar", "Island interior", "Acacia woodlands", "Acacia shrublands", "Acacia grasslands",
                 "Short mopane", "Mixed mopane", "Exposed soils"],

    'ZY_HHK': ["Reed", "Spartina alterniflora", "Salt filter pond", "Salt evaporation pond", "Dry pond", "Tamarisk",
               "Salt pan", "Seepweed", "River", "Sea", "Mudbank", "Tidal creek", "Fallow land",
               "Ecological restoration pond", "Robinia", "Fishpond", "Pit pond", "Building", "Bare land", "Paddyfield",
               "Cotton", "Soybean", "Corn"],

    'Houston': ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential",
                "Commercial", "Road", "Highway", "Railway", "Parking Lot1", "Parking Lot2", "Tennis court",
                "Running track"],

}

color_map_dict = {

    'PU': np.array([[0, 0, 255], [76, 230, 0], [255, 190, 232], [255, 0, 0], [156, 156, 156],
                    [255, 255, 115], [0, 255, 197], [132, 0, 168], [0, 0, 0]], dtype=np.uint8),

    'IP': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                    [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                    [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                    [255, 170, 0]], dtype=np.uint8),

    'Salinas': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                         [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                         [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                         [255, 170, 0]], dtype=np.uint8),

    'KSC': np.array([[0, 168, 132], [76, 0, 115], [255, 0, 0], [190, 255, 232], [0, 0, 0],
                     [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                     [0, 0, 255], [156, 156, 156], [115, 223, 255]], dtype=np.uint8),

    'Botswana': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                          [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                          [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0]], dtype=np.uint8),

    'Houston': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0], [115, 0, 0],
                         [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223], [0, 0, 255],
                         [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0]], dtype=np.uint8),

}

false_color_dict = {
    'PU': [102, 56, 31],
    'IP': [50, 27, 17],
    'Salinas': [57, 27, 17],
    'KSC': [],
    'Botswana': [],
}

true_color_dict = {
    'PU': [56, 31, 6],
    'IP': [27, 17, 7],
    'Salinas': [27, 17, 7],
    'KSC': [],
    'Botswana': [],
}


def load_dataset(dataset_name: str, key: int):
    """
    load data
    :param dataset_name: dataset's dictionary
    :param key: indicator
    :return: numpy.ndarray
    """
    kv = {0: 'raw',
          1: 'corrected',
          2: 'gt'}
    path = dataset_path[dataset_name]
    data = sio.loadmat(path[kv[key]])
    for item in data.items():
        if type(item[1]) is np.ndarray:
            return item[1]


default_mirror_width = 35
def mirror_concatenate(x, mirror_width=default_mirror_width):
    return cv2.copyMakeBorder(x, mirror_width, mirror_width, mirror_width, mirror_width, cv2.BORDER_REFLECT)


def HSI_LazyProcessing(dataset_name='PU'):
    """

    :param dataset_name:
    :param n_pc:
    :param patch_size:
    :return:
    """
    # patch_radius = patch_size // 2
    Y = load_dataset(dataset_name, key=2)
    X_extension = load_dataset(dataset_name, key=1)
    row, col, band = X_extension.shape
    X_extension = X_extension.reshape((row * col, -1)).astype(np.float)
    X_extension = X_extension - np.mean(X_extension, axis=0)
    X_extension = X_extension.reshape((row, col, -1))
    Y = Y.reshape(row * col, -1)
    return X_extension, Y, [row, col, band]


def data_augmentation(patch):
    # patch.shape = [H, W, C]

    def vertical_rotation(patch):

        return np.flip(patch, axis=0)

    def horizontal_rotation(patch):

        return np.flip(patch, axis=1)

    def transpose(patch):

        return patch.transpose((1, 0, 2))

    patch = vertical_rotation(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    patch = horizontal_rotation(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    patch = transpose(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    return patch


def generate_batch(train_test_set, X_PCAMirror, Y, label=None,
                   dataset_name='PU', patch_size=9, batch_size=64,
                   shuffle=True, mode='train', process=None, augment=False):
    """

    :param train_test_set: the index of train(test) set from the universal set
    :param X_PCAMirror:
    :param Y:
    :param batch_size:
    :param shuffle:
    :return:
    """

    num_samples = train_test_set.size
    row, col, band = dataset_size_dict.get(dataset_name)
    patch_radius = patch_size // 2

    set_idx = np.arange(train_test_set.size)
    if shuffle:
        random_state = np.random.RandomState()
        random_state.shuffle(set_idx)
    train_test_set = train_test_set[set_idx]

    for i in range(0, num_samples, batch_size):
        # batch_i represents the i-th element in current batch
        batch_i = train_test_set[np.arange(i, min(num_samples, i + batch_size))]
        batch_i_row = np.floor(batch_i * 1.0 / col).astype(np.int32)
        batch_i_col = (batch_i - batch_i_row * col).astype(np.int32)
        upper_edge, bottom_edge = (batch_i_row - patch_radius), (batch_i_row + patch_radius + 1)
        left_edge, right_edge = (batch_i_col - patch_radius), (batch_i_col + patch_radius + 1)

        patches = []
        for j in range(batch_i.size):
            patch = X_PCAMirror[
                    upper_edge[j] + patch_radius: bottom_edge[j] + patch_radius,
                    left_edge[j] + patch_radius: right_edge[j] + patch_radius,
                    :]
            patch = data_augmentation(patch) if mode == 'train' and augment else patch
            patches.append(patch)
        patches = np.array(patches)
        patches = np.transpose(patches, (0, 3, 1, 2))
        labels = Y[batch_i, :] - 1

        yield patches, labels


def split_train_test_set(Y, dataset_name='PU', train_num=5, batch_size=64, seed=0, prepare_sample=None, zone=None, resample_seed=False):
    """

    :param dataset_name:
    :param train_num: number of train samples for each class
    :param n_pc: number of principal component
    :param patch_size: size of the patch
    :param batch_size:
    :param shuffle:
    :return:
    """
    if isinstance(train_num, int):
        train_num = [train_num] * len(dataset_class_dict.get(dataset_name))
    elif isinstance(train_num, list):
        pass

    n_class = Y.max()
    train_set, test_set = [], []

    random_state = np.random.RandomState(seed=seed)
    for i in range(1, n_class + 1):
        index = np.where(Y == i)[0]
        #  variable n_data is used only when data is split by percentage rule.
        n_data = index.shape[0]
        random_state.shuffle(index)

        print('Preparing training samples -- ' + str(i) + ' - ' + str(n_class))
        if train_num[i - 1] * 2 > n_data:
            train_set.extend(index[:int(np.ceil(n_data / 2))])
            test_set.extend(index[int(np.ceil(n_data / 2)):])
        else:
            train_set.extend(index[:train_num[i - 1]])
            test_set.extend(index[train_num[i - 1]:])

    if len(train_set) < batch_size:
        train_batch_size = len(train_set)
    else:
        train_batch_size = batch_size

    train_set, test_set = np.array(train_set), np.array(test_set)
    train_step = int(np.ceil(train_set.size / train_batch_size))
    test_step = int(np.ceil(test_set.size / batch_size))

    # train_set and test_set represent the index of train samples and test samples
    return train_set, train_step, test_set, test_step

