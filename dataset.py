import os
import numpy as np
from PIL import Image
from calibration import KittiCalibration, AudiCalibration
import cv2


class Kitti:
    def __init__(self):
        self.root_dir = "/home/user/work/master_thesis/datasets/kitti/kitti/object/training"
        self.lidar_path = os.path.join(self.root_dir, "velodyne")
        self.image_path = os.path.join(self.root_dir, "image_2")
        self.calib_path = os.path.join(self.root_dir, "calib")
        self.lidar_fov_path = "/home/user/work/master_thesis/datasets/lidar_fov_images/kitti/training"
        self.files_list = os.listdir(self.lidar_path)

    def get_lidar(self, idx):
        n_vec = 4
        dtype = np.float32
        lidar_file = os.path.join(self.lidar_path, self.files_list[idx])
        assert os.path.exists(lidar_file)
        lidar_pc_raw = np.fromfile(lidar_file, dtype)
        return lidar_pc_raw.reshape((-1, n_vec))

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, self.files_list[idx].replace('.bin', '.png'))
        assert os.path.exists(img_file)
        return Image.open(img_file).convert("L")

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, self.files_list[idx].replace('.bin', '.txt'))
        assert os.path.exists(calib_file)
        return KittiCalibration(calib_file)


class Lyft:
    def __init__(self):
        self.root_dir = "/home/user/work/master_thesis/datasets/lyft_kitti/object/training"
        self.lidar_path = os.path.join(self.root_dir, "velodyne")
        self.image_path = os.path.join(self.root_dir, "image_2")
        self.calib_path = os.path.join(self.root_dir, "calib")
        self.lidar_fov_path = "/home/user/work/master_thesis/datasets/lidar_fov_images/lyft"
        self.files_list = os.listdir(self.lidar_path)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, self.files_list[idx])
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, self.files_list[idx].replace('.bin', '.png'))
        assert os.path.exists(img_file)
        return Image.open(img_file).convert("L")

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, self.files_list[idx].replace('.bin', '.txt'))
        assert os.path.exists(calib_file)
        return KittiCalibration(calib_file)


class Audi:
    def __init__(self):
        self.root_dir = "/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes"
        self.lidar_path = os.path.join(self.root_dir, "lidar", "cam_front_center")
        self.image_path = os.path.join(self.root_dir, "camera", "cam_front_center")
        self.calib_path = os.path.join(self.root_dir, "cams_lidars.json")
        self.lidar_fov_path = os.path.join(self.root_dir, "/home/user/work/master_thesis/datasets/lidar_fov_images/audi")
        self.files_list = os.listdir(self.lidar_path)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, self.files_list[idx])
        assert os.path.exists(lidar_file)
        lidar_pc_raw = np.load(lidar_file)
        return lidar_pc_raw

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, self.files_list[idx].replace('.npz', '.png').replace('lidar', 'camera'))
        print("IMG: ", img_file)
        assert os.path.exists(img_file)
        return cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    def get_calib(self, idx):
        calib_file = self.calib_path
        assert os.path.exists(calib_file)
        return AudiCalibration(calib_file)
