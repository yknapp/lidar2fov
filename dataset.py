import os
import numpy as np
from PIL import Image
from calibration import Calibration, KittiCalibration, AudiCalibration
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
        #calib_file = os.path.join(self.calib_path, self.files_list[idx].replace('.bin', '.txt'))
        #assert os.path.exists(calib_file)
        #return KittiCalibration(calib_file)

        # calibration with mean values of all kitti calibration files
        calib = Calibration()
        calib.P = np.array([[719.787081, 0., 608.463003, 44.9538775],
                      [0., 719.787081, 174.545111, 0.1066855],
                      [0., 0., 1., 3.0106472e-03]
                      ])
        calib.V2C = np.array([
            [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
            [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
            [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01]
        ])
        calib.R0 = np.array([
            [0.99992475, 0.00975976, -0.00734152],
            [-0.0097913, 0.99994262, -0.00430371],
            [0.00729911, 0.0043753, 0.99996319]
        ])

        calib.c_u = calib.P[0, 2]
        calib.c_v = calib.P[1, 2]
        calib.f_u = calib.P[0, 0]
        calib.f_v = calib.P[1, 1]
        calib.b_x = calib.P[0, 3] / (-calib.f_u)  # relative
        calib.b_y = calib.P[1, 3] / (-calib.f_v)

        return calib


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
        lidar_pc = np.zeros([lidar_pc_raw['points'].shape[0], 4])
        lidar_pc[:, :3] = lidar_pc_raw['points']
        lidar_pc[:, 3] = lidar_pc_raw['reflectance']
        return lidar_pc

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, self.files_list[idx].replace('.npz', '.png').replace('lidar', 'camera'))
        print("IMG: ", img_file)
        assert os.path.exists(img_file)
        return Image.open(img_file).convert("L")

    def get_calib(self, idx):
        calib_file = self.calib_path
        assert os.path.exists(calib_file)
        return AudiCalibration(calib_file)
