import logging
import argparse
import sys
import torch
import random
from typing import List
import numpy as np
import datetime
import os


class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description="HSI & Lidar Classification")

        # model paras
        parser.add_argument(
            "--model_type",
            default="s2atnet",
            choices=[
                "s2atnet",  # nice!
                "hypermlp",  # good
                "hct",  # good
                "crosshl",  # good
                "mhst",  # good
                "ms2canet",  # good
                "exvit",  # good
                "s2enet",  # nice!
                "nncnet",  # good
                "coupledcnn", # todo
                "fusatnet",  # bug!
                "endnet",  # bug!
            ],
            help="select a model to train",
        )

        # dataset
        parser.add_argument("--dataset_name", default=r"augsburg", help="augsburg, muufl, houston, trento, yancheng")  # todo
        parser.add_argument("--batch_size", type=int, default=144)
        parser.add_argument("--pca_components", type=int, default=30)
        parser.add_argument("--hsi_channels", default=30, type=int, help="channels of hsi")
        parser.add_argument("--lidar_channels", default=1, type=int, help="channels of lidar")
        parser.add_argument("--num_tokens", type=int, default=121, help="num_tokens % 2 == 1") #
        parser.add_argument("--hsi_patch_size", default=11, type=int, help="channels of hsi")
        parser.add_argument("--lidar_patch_size", default=11, type=int, help="channels of lidar")  # todo

        # optimizer
        parser.add_argument("--epochs", default=200, type=int)
        parser.add_argument("--times", default=1, type=int)
        parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")

        # misc
        parser.add_argument("--work_dirs", default="./work_dirs", help="experiment results saved at work_dirs")
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--color_map_dir", default="./color_map_dir", help="save color maps")

        self.args = parser.parse_args()
        self.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model setting
        if self.args.model_type == "mhst":
            self.args.hsi_patch_size = 8
            self.args.lidar_patch_size = 8
        elif self.args.model_type == "hypermlp":
            self.args.hsi_patch_size = 11
            self.args.lidar_patch_size = 15
        elif self.args.model_type == "exvit":
            self.args.hsi_patch_size = 13
            self.args.lidar_patch_size = 13
        elif self.args.model_type == "s2enet":
            self.args.hsi_patch_size = 7
            self.args.lidar_patch_size = 7
        elif self.args.model_type == "endnet":  # todo
            self.args.hsi_patch_size = 1
            self.args.lidar_patch_size = 1

        # dataset setting
        if self.args.dataset_name == "augsburg":
            self.args.num_classes = 7
        elif self.args.dataset_name == "muufl":  # todo
            self.args.num_classes = 11
            # self.args.image_dir = r"../dataset/munich_s1_output_folder/munich_s1"
            # self.args.gt_dir = r"../dataset/munich_s1_output_folder/munich_anno"
            # self.args.segments_dir = r"../dataset/munich_s1_output_folder/munich_segments"
        elif self.args.dataset_name == "houston":
            self.args.num_classes = 15
        elif self.args.dataset_name == "trento":
            self.args.num_classes = 6
        elif self.args.dataset_name == "yancheng":
            self.args.num_classes = 9
        self._set_seed(self.args.seed)
        self._configure_logger()
        self._print_args()

    def get_args(self):
        return self.args

    def _print_args(self):
        self.args.logger.info("*******************       start  args      *******************")
        for arg, content in self.args.__dict__.items():
            self.args.logger.info("{}:{}".format(arg, content))
        self.args.logger.info("*******************        end   args     *******************")

    def _configure_logger(self):
        logger = logging.getLogger(name="logger")
        logger.setLevel("DEBUG")

        date_time = datetime.datetime.now()
        time = date_time.strftime("%Y%m%d_%H%M%S")
        # print(time)
        self.args.time = time
        # experiment dir name
        experiment_name = self.args.model_type + "_" + self.args.dataset_name
        self.args.log_file_name = experiment_name + ".log"
        self.args.work_dirs = os.path.join(self.args.work_dirs, experiment_name, time)
        os.makedirs(self.args.work_dirs)
        log_file = os.path.join(self.args.work_dirs, self.args.log_file_name)  # TODO
        file_handler = logging.FileHandler(log_file)
        stdout_handler = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter(
        #     fmt='%(asctime)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(fmt="%(asctime)s| %(message)s")
        file_handler.setLevel("INFO")
        file_handler.setFormatter(formatter)
        stdout_handler.setLevel("DEBUG")
        stdout_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

        logger.info("=" * 88)
        logger.info("experiment log file saved at {}".format(self.args.work_dirs))
        self.args.logger = logger

    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
