import torch
import numpy as np
import torch.utils.data as Data

def obtain_index(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def crop_patches(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def crop_hsi_patches(data_size, data_indices, whole_data, hsi_patch_padding, padded_data, pca_components):
    small_cubic_data = np.zeros((data_size, 2 * hsi_patch_padding + 1, 2 * hsi_patch_padding + 1, pca_components))
    data_assign = obtain_index(data_indices, whole_data.shape[0], whole_data.shape[1], hsi_patch_padding)
    for i in range(len(data_assign)):
        small_cubic_data[i] = crop_patches(padded_data, data_assign[i][0], data_assign[i][1], hsi_patch_padding)
    return small_cubic_data

def crop_lidar_patches(data_size, data_indices, whole_data, hsi_patch_padding, padded_data):
    small_cubic_data = np.zeros((data_size, 2 * hsi_patch_padding + 1, 2 * hsi_patch_padding + 1))
    data_assign = obtain_index(data_indices, whole_data.shape[0], whole_data.shape[1], hsi_patch_padding)
    for i in range(len(data_assign)):
        small_cubic_data[i] = crop_patches(padded_data, data_assign[i][0], data_assign[i][1], hsi_patch_padding)
    return small_cubic_data

def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, ALL_SIZE, all_indices,
                  whole_hsi, whole_lidar, hsi_patch_padding, lidar_patch_padding, padded_hsi, padded_lidar,  gt, args):
    args.logger.info("begin")
    gt_all = gt[all_indices] - 1
    gt_total = gt[total_indices]-1
    y_train = gt[train_indices]-1
    y_test = gt[test_indices]-1

    hsi_total_data = crop_hsi_patches(TOTAL_SIZE, total_indices, whole_hsi, hsi_patch_padding, padded_hsi, args.pca_components)
    # args.logger.info(f"{hsi_total_data.shape}")
    lidar_total_data = crop_lidar_patches(TOTAL_SIZE, total_indices, whole_lidar, lidar_patch_padding, padded_lidar)
    # args.logger.info(lidar_total_data.shape)

    hsi_all_data = crop_hsi_patches(ALL_SIZE, all_indices, whole_hsi, hsi_patch_padding, padded_hsi, args.pca_components)
    # args.logger.info(hsi_all_data.shape)
    lidar_all_data = crop_lidar_patches(ALL_SIZE, all_indices, whole_lidar, lidar_patch_padding, padded_lidar)
    # args.logger.info(lidar_all_data.shape)

    hsi_train_data = crop_hsi_patches(TRAIN_SIZE, train_indices, whole_hsi, hsi_patch_padding, padded_hsi, args.pca_components)
    # args.logger.info(hsi_train_data.shape)
    lidar_train_data = crop_lidar_patches(TRAIN_SIZE, train_indices, whole_lidar, lidar_patch_padding, padded_lidar)
    # args.logger.info(lidar_train_data.shape)

    hsi_test_data = crop_hsi_patches(TEST_SIZE, test_indices, whole_hsi, hsi_patch_padding, padded_hsi, args.pca_components)
    # args.logger.info(hsi_test_data.shape)
    lidar_test_data = crop_lidar_patches(TEST_SIZE, test_indices, whole_lidar, lidar_patch_padding, padded_lidar)
    # args.logger.info(lidar_test_data.shape)

    hsi_train = hsi_train_data.transpose(0, 3, 1, 2)
    hsi_test = hsi_test_data.transpose(0, 3, 1, 2)
    args.logger.info(f'after transpose: hsi_train: { hsi_train.shape}')
    args.logger.info(f'after transpose: hsi_test: {hsi_test.shape}')
    args.logger.info(f'after transpose: lidar_train_data: {lidar_train_data.shape}')
    args.logger.info(f'after transpose: lidar_test_data: {lidar_test_data.shape}')
    hsi_tensor_train = torch.from_numpy(hsi_train).type(torch.FloatTensor)
    lidar_tensor_train = torch.from_numpy(lidar_train_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_dataset_train = Data.TensorDataset(hsi_tensor_train, lidar_tensor_train, y1_tensor_train)

    hsi_tensor_test = torch.from_numpy(hsi_test).type(torch.FloatTensor)
    lidar_tensor_test = torch.from_numpy(lidar_test_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.LongTensor)
    # args.logger.info('yishape',y1_tensor_test.shape)
    torch_dataset_test = Data.TensorDataset(hsi_tensor_test, lidar_tensor_test, y1_tensor_test)

    hsi_total = hsi_total_data.transpose(0, 3, 1, 2)
    hsi_all = hsi_all_data.transpose(0, 3, 1, 2)

    hsi_total_tensor_data = torch.from_numpy(hsi_total).type(torch.FloatTensor)
    lidar_total_tensor_data = torch.from_numpy(lidar_total_data).type(torch.FloatTensor).unsqueeze(1)
    total_tensor_data_label = torch.from_numpy(gt_total).type(torch.LongTensor)
    torch_dataset_total = Data.TensorDataset(hsi_total_tensor_data, lidar_total_tensor_data, total_tensor_data_label)

    hsi_all_tensor_data = torch.from_numpy(hsi_all).type(torch.FloatTensor)
    lidar_all_tensor_data = torch.from_numpy(lidar_all_data).type(torch.FloatTensor).unsqueeze(1)
    # all_tensor_data_label = torch.from_numpy(gt_all).type(torch.LongTensor)
    torch_dataset_all = Data.TensorDataset(hsi_all_tensor_data, lidar_all_tensor_data)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    total_iter = Data.DataLoader(
        dataset=torch_dataset_total,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    args.logger.info("end")
    return train_iter, test_iter, total_iter, all_iter #, y_test
