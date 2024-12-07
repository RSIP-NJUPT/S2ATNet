import numpy as np
import scipy.io as sio

from sklearn.decomposition import PCA


def apply_pca(x, num_components):

    y = np.reshape(x, (-1, x.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    y = pca.fit_transform(y)
    y = np.reshape(y, (x.shape[0], x.shape[1], num_components))

    return y
def load_dataset(args):
    assert args.dataset_name.lower() in ["augsburg", "muufl", "houston", "trento", "yancheng"]
    # augsburg
    if args.dataset_name.lower() == "augsburg":
        hsi_data = sio.loadmat('data/augsburg/data_HS_LR.mat')['data_HS_LR']
        lidar_data = sio.loadmat('data/augsburg/data_DSM.mat')['data_DSM']
        labels_train = sio.loadmat('data/augsburg/train_test_gt.mat')['train_data']
        labels_test = sio.loadmat('data/augsburg/train_test_gt.mat')['test_data']
        labels = labels_test + labels_train
    elif args.dataset_name.lower() == "muufl":
        data = sio.loadmat('data/muufl/muufl.mat')
        hsi_data = data["hsi"]
        lidar_data = data['lidar_1'][..., 0]  # todo 2->1
        labels = data['gt']
        labels[labels==-1] = 0
    elif args.dataset_name.lower() == "houston":
        hsi_data = sio.loadmat('data/houston/Houston.mat')['HSI']
        lidar_data = sio.loadmat('data/houston/LiDAR.mat')['LiDAR']
        labels_train =  sio.loadmat('data/houston/train_test_gt.mat')['train_data']
        labels_test = sio.loadmat('data/houston/train_test_gt.mat')['test_data']
        labels = labels_test + labels_train
    elif args.dataset_name.lower() == "trento":
        hsi_data = sio.loadmat('data/trento/HSI_Trento.mat')['hsi_trento']
        lidar_data = sio.loadmat('data/trento/Lidar1_Trento.mat')['lidar1_trento']
        labels = sio.loadmat('data/trento/GT_Trento.mat')['gt_trento']
    elif args.dataset_name.lower() == "yancheng":
        lidar_data = sio.loadmat('data/cut/lidar_DSM_45.mat')['lidar']
        hsi_data = sio.loadmat('data/cut/hsi_45.mat')['hsi']
        labels = sio.loadmat('data/cut/label_45(revised).mat')['label']
    pass
    return hsi_data, lidar_data, labels


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def sampling_with_bg(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m+1):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m+1):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def split_traintest(args, groundTruth):  # todo
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))
    dataset_name = args.dataset_name.lower()
    # assert dataset_name in ["augsburg", "muufl"]
    if dataset_name == "augsburg":
        # augsburg-7
        amount = [
            675, 1516, 192, 1342, 28, 82, 76
        ]
        # amount = [250 for _ in range(7)]
    elif dataset_name == "muufl":
        # muufl-11
        amount = [150 for _ in range(11)]
    elif dataset_name == "houston":
        # Houston-15
        # amount = [
        #         198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187
        #     ]
        amount = [220 for _ in range(15)]
    elif dataset_name == "trento":
        # trento-6
        amount = [
            129, 125, 105, 154, 184, 122,
        ]
    elif dataset_name == "yancheng":
        # cut-9
        amount = [220 for _ in range(9)]

    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]

        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
