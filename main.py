import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import time
from obtain_patches import generate_iter
import utils
from models import *
from torch.optim.lr_scheduler import MultiStepLR
from options import OptInit
from prepare_dataset import load_dataset, sampling, sampling_with_bg, split_traintest, apply_pca
from metrics import report_AA_CA, report_metrics
import os

def create_data_loader(args):

    hsi, lidar, y = load_dataset(args)

    hsi_patch_size = args.hsi_patch_size
    lidar_patch_size = args.lidar_patch_size
    hsi_patch_padding = int(hsi_patch_size // 2)
    lidar_patch_padding = int(lidar_patch_size // 2)
    logger = args.logger

    all_samples = np.prod(y.shape)

    logger.info("hsi data shape: {}".format(hsi.shape))
    logger.info("Lidar data shape:{}".format(lidar.shape))
    logger.info("Label shape:{}".format(y.shape))
    # pca
    logger.info("\n... ... PCA tranformation ... ...")
    hsi = apply_pca(hsi, args.pca_components)
    logger.info("Data shape after PCA: {}".format(hsi.shape))

    hsi_all_data = hsi.reshape(np.prod(hsi.shape[:2]), np.prod(hsi.shape[2:]))
    lidar_all_data = lidar.reshape(
        np.prod(lidar.shape[:2]),
    )
    gt = y.reshape(
        np.prod(y.shape[:2]),
    )
    gt = gt.astype(np.int_)

    assert args.num_classes == max(gt)

    logger.info("num_classes = {}".format(args.num_classes))
    # 归一化
    hsi_all_data = preprocessing.scale(hsi_all_data)
    hsi_data = hsi_all_data.reshape(hsi.shape[0], hsi.shape[1], hsi.shape[2])
    whole_data_hsi = hsi_data
    # padding
    padded_data_hsi = np.lib.pad(
        whole_data_hsi,
        ((hsi_patch_padding, hsi_patch_padding), (hsi_patch_padding, hsi_patch_padding), (0, 0)),
        "constant",
        constant_values=0,
    )

    lidar_all_data = preprocessing.scale(lidar_all_data)
    lidar_data = lidar_all_data.reshape(lidar.shape[0], lidar.shape[1])
    whole_data_lidar = lidar_data
    padded_data_lidar = np.lib.pad(
        whole_data_lidar,
        ((lidar_patch_padding, lidar_patch_padding), (lidar_patch_padding, lidar_patch_padding)),
        "constant",
        constant_values=0,
    )
    logger.info("\n... ... create train & test data ... ...")
    # 用gt划分
    train_indices, test_indices = split_traintest(args, gt)
    # train_indices, test_indices = sampling(0.99, gt)
    total_samples = len(train_indices) + len(test_indices)
    _, all_indices = sampling_with_bg(1, gt)
    _, total_indices = sampling(1, gt)
    train_samples = len(train_indices)
    logger.info("Train size:{}".format(train_samples))
    test_samples = total_samples - train_samples
    logger.info("Test size: {}".format(test_samples))

    logger.info("\n-----Selecting Small Cube from the Original Cube Data-----")
    train_iter, test_iter, total_iter, all_iter = generate_iter(
        train_samples,
        train_indices,
        test_samples,
        test_indices,
        total_samples,
        total_indices,
        all_samples,
        all_indices,
        whole_data_hsi,
        whole_data_lidar,
        hsi_patch_padding,
        lidar_patch_padding,
        padded_data_hsi,
        padded_data_lidar,
        gt,
        args,
    )

    return train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices


# 测试模型，提取预测结果
def test(model, test_loader, args):
    count = 0
    # 模型测试
    model.eval()
    y_pred_test = 0
    y_test = 0
    for hsi_data, lidar_data, labels in test_loader:
        hsi_data, lidar_data = hsi_data.to(args.device), lidar_data.to(args.device)
        if args.model_type == 'ms2canet': # only use output1
            output, _, _ = model(hsi_data, lidar_data)
        elif args.model_type == 'coupledcnn':
            output, _, _ = model(hsi_data, lidar_data)
        else:
            output = model(hsi_data, lidar_data)
        output = np.argmax(output.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = output
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, output))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def main():
    args = OptInit().get_args()

    start = time.time()
    for i in range(args.times):
        train_loader, test_iter, total_iter, all_iter, y, total_indices, all_indices = create_data_loader(args)

        if args.model_type.lower() == "s2atnet":
            model = S2ATNet(args.num_classes,num_tokens=args.num_tokens).to(args.device)
        elif args.model_type.lower() == "hypermlp": # hsi_patch_size=11, lidar_patch_size=15
            model = MLPMixer(num_classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "hct":
            model = HCTnet(num_classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "crosshl":
            model = CrossHL_Transformer(Classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "mhst":
            model = MHST(num_classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "ms2canet":  # todo 3个loss
            para_tune = False
            if args.dataset_name.lower() == "houston":
                para_tune = True
            model = MS2CANet(NC=args.hsi_channels, Classes=args.num_classes, FM=64, para_tune=para_tune).to(args.device)
        elif args.model_type.lower() == "exvit":
            model = MViT(
                num_classes=args.num_classes,
                num_patches=[args.hsi_channels, args.lidar_channels],
                patch_size=args.hsi_patch_size,
            ).to(args.device)
        elif args.model_type.lower() == "s2enet":  # patch_size=7 nice!!!
            model = S2ENet(n_classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "fusatnet":  # patch_size=11
            model = FusAtNet(num_classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "endnet":  # patch_size=11
            model = EndNet(n_classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "nncnet":  # patch_size=11
            model = NNCNet(num_classes=args.num_classes).to(args.device)
        elif args.model_type.lower() == "coupledcnn":  # patch_size=11
            model = CoupledCNNs(classes=args.num_classes, 
                                hsi_inchannel=args.pca_components).to(args.device)

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)  # todo

        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # todo
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        milestones = [50, 150, 220]
        # milestones = [50, 100, 150]

        # lr_scheduler = ExponentialLR(optimizer, gamma=0.9)  # todo
        lr_scheduler = MultiStepLR(optimizer, milestones, gamma=0.2, last_epoch=-1, verbose=False)
        # lr_scheduler = PolynomialLR(optimizer, 20, power=0.9)
        total_loss = 0
        best_loss = 100.0

        tic1 = time.perf_counter()
        best_loss = [float('inf'),float('inf'),float('inf')]
        best_model_paths = ''
        for epoch in range(args.epochs):
            model.train()
            for j, (hsi_data, lidar_data, label) in enumerate(train_loader):
                hsi_data, lidar_data, label = (
                    hsi_data.to(args.device),
                    lidar_data.to(args.device),
                    label.to(args.device),
                )
                # print(f"hsi.shape{hsi_data.shape}, lidar.shape={lidar_data.shape}")
                if args.model_type == 'ms2canet':
                    output1, output2, output3 = model(hsi_data, lidar_data)
                    loss1 = criterion(output1, label)
                    loss2 = criterion(output2, label)
                    loss3 = criterion(output3, label)
                    loss = loss1 + loss2 + loss3
                elif args.model_type == 'coupledcnn':
                    output1, output2, output3 = model(hsi_data, lidar_data)
                    loss1 = criterion(output1, label)
                    loss2 = criterion(output2, label)
                    loss3 = criterion(output3, label)
                    loss = 0.01*loss1 + 0.01*loss2 + 1*loss3
                else:
                    output = model(hsi_data, lidar_data)
                    loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            args.logger.info(
                "[%d-Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]"
                % (i + 1, epoch + 1, total_loss / (epoch + 1), loss.item())
            )
            lr_scheduler.step()
            # 只保存模型参数
            # if loss.item() < best_loss:
            #     best_loss = loss.item()
            #     torch.save(model.state_dict(), f"{args.work_dirs}/{i+1}_best_model.pth")
            if loss.item() < best_loss[0]:
                best_loss[2] = best_loss[1]
                best_loss[1] = best_loss[0]
                best_loss[0] = loss.item()
                # best_model_paths = f"{args.work_dirs}/{i+1}_best_model.pth"
                # model.load_state_dict(torch.load(best_model_paths))
                torch.save(model.state_dict(), f"{args.work_dirs}/{i + 1}_best_model.pth")
            elif loss.item() < best_loss[1]:
                best_loss[2] = best_loss[1]
                best_loss[1] = loss.item()

            elif loss.item() < best_loss[2]:
                best_loss[2] = loss.item()
        args.logger.info("Finished Training")
        toc1 = time.perf_counter()


        # test
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(model, test_iter, args)
        toc2 = time.perf_counter()
        # 评价指标
        classification, oa, confusion, each_acc, aa, kappa = report_metrics(y_test, y_pred_test)
        classification = str(classification)
        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2
        args.logger.info("{} Training_Time (s)".format(Training_Time))
        args.logger.info("{} Test_time (s)".format(Test_time))

        args.logger.info("{} Overall accuracy (%)".format(oa))
        args.logger.info("{} Average accuracy (%)".format(aa))
        args.logger.info("{} Kappa accuracy (%)".format(kappa))
        args.logger.info("\n{} Each accuracy (%)".format(each_acc))
        args.logger.info("\n{}".format(classification))
        args.logger.info("\n{}".format(confusion))
        args.logger.info("------Get classification results successful-------")
    end = time.time()
    args.logger.info("Total running time: {:.2f} s".format(end - start))

    color_map_dir = os.path.join(args.color_map_dir, f'{args.model_type}_{args.dataset_name}_{i}')
    if not os.path.exists(color_map_dir):
        os.makedirs(color_map_dir)
    utils.generate_png(
        total_iter, model, y, args.device, total_indices, os.path.join(color_map_dir, f'color_map'))
    utils.generate_all_png(
        all_iter, model, y, args.device, all_indices, os.path.join(color_map_dir, f'all_color_map'))


if __name__ == "__main__":
    main()
