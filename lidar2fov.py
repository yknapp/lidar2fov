"""
    Projection of LiDAR point cloud to Field-of-View(FOV) of the camera.
    Supported datasets: KITTI, Lyft Level 5, Audi A2D2
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import labels

from PIL import Image
from dataset import Kitti, Lyft, Audi


# CLASS NAMES WITH IDs
CLASS_NAME_TO_ID_KITTI = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Van': 0,
    'Person_sitting': 1
}
CLASS_NAME_TO_ID_LYFT = {
    'car': 				    0,
    'pedestrian': 		    1,
    'bicycle': 			    2
}
CLASS_NAME_TO_ID_AUDI = {
    'Car': 				    0,
    'Pedestrian': 		    1,
    'Bicycle': 			    2,
    'Cyclist':              2
}


def get_mask(rect_pts, points_2d, imgsize):
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < imgsize[0]) & \
           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < imgsize[1])
    mask = mask & (rect_pts[:, 2] >= 2)  # minimal distance 2 meters

    # pts_on_image_with_depth = np.append(points_2d[mask, 0:2], rect_pts[mask, 2], axis=1)

    pts_on_image_with_depth = np.zeros([mask.sum(), 3])
    pts_on_image_with_depth[:, 0:2] = points_2d[mask, 0:2]
    pts_on_image_with_depth[:, 2] = rect_pts[mask, 2]

    # return points_2d[mask, 0:2], rect_pts[mask, ]

    return pts_on_image_with_depth, rect_pts[mask, ]


def lidarimg2grid(pts_image, img_shape, max_distance):
    size_0 = img_shape[0]
    size_1 = img_shape[1]
    grid = np.zeros(img_shape[:2])
    print(grid.shape)
    for p in pts_image:
        i = int(p[0]) - 1
        j = int(p[1]) - 1

        value = p[2] / max_distance  # representation of depth, i.e. p[2], 1/p[2], log(p[2])
        if value > 1.0:
            value = 0.0

        grid[i, j] = value

        # add 8 pixels next to "real" pixel coordinate to make the data point bigger
        # X: original pixel, 0: additional pixel
        # 0 0 0
        # 0 X 0
        # 0 0 0
        if i + 1 < size_0 and j + 1 < size_1:
            grid[i+1, j+1] = value
        if i < size_0 and j + 1 < size_1:
            grid[i, j+1] = value
        if i + 1 < size_0 and j < size_1:
            grid[i+1, j] = value
        if i - 1 >= 0 and j - 1 >= 0:
            grid[i-1, j-1] = value
        if i - 1 >= 0 and j < size_1:
            grid[i-1, j] = value
        if i < size_0 and j - 1 >= 0:
            grid[i, j-1] = value
        if i - 1 >= 0 and j + 1 < size_1:
            grid[i-1, j+1] = value
        if i + 1 < size_0 and j - 1 >= 0:
            grid[i+1, j-1] = value

        # add another 9 pixels
        # 1: additional pixels
        #   1 1 1
        # 1 0 0 0 1
        # 1 0 X 0 1
        # 1 0 0 0 1
        #   1 1 1
        if i + 2 < size_0 and j < size_1:
            grid[i+2, j] = value
        if i < size_0 and j + 2 < size_1:
            grid[i, j+2] = value
        if i - 2 >= 0 and j + 2 < size_1:
            grid[i-2, j] = value
        if i < size_0 and j - 2 >= 0:
            grid[i, j-2] = value
        if i + 2 < size_0 and j + 1 < size_1:
            grid[i+2, j+1] = value
        if i + 2 < size_0 and j - 1 >= 0:
            grid[i+2, j-1] = value
        if i - 2 >= 0 and j + 1 < size_1:
            grid[i-2, j+1] = value
        if i - 2 >= 0 and j - 1 >= 0:
            grid[i-2, j-1] = value
        if i + 1 < size_0 and j + 2 < size_1:
            grid[i+1, j+2] = value
        if i + 1 < size_0 and j - 2 >= 0:
            grid[i+1, j-2] = value
        if i - 1 >= 0 and j + 2 < size_1:
            grid[i-1, j+2] = value
        if i - 1 >= 0 and j - 2 >= 0:
            grid[i-1, j-2] = value
    return grid.T


def setup_plt():
    plt.figure()
    axes = plt.axes()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)


def save_as_grid_plt(grid_img, output_path):
    plt.imsave(output_path, grid_img)
    plt.close()
    print("Grid size: ", Image.open(output_path).size)
    

def main(chosen_dataset, export_type='show'):
    dataset = None
    kitti = Kitti()
    # no crop by default
    crop = (None, None, None, None)  # (y_min, y_max, x_min, x_max)
    if chosen_dataset == "kitti":
        dataset = Kitti()
        class_names_to_id = CLASS_NAME_TO_ID_KITTI
    elif chosen_dataset == "lyft":
        dataset = Lyft()
        class_names_to_id = CLASS_NAME_TO_ID_LYFT
    elif chosen_dataset == "audi":
        dataset = Audi()
        class_names_to_id = CLASS_NAME_TO_ID_AUDI
        crop = (None, None, 195, 1002)  # crop Audi images due to limited FOV of Audi's LiDAR
    elif chosen_dataset == "kitti_cropped":
        dataset = Kitti()
        class_names_to_id = CLASS_NAME_TO_ID_KITTI
        crop = (None, None, 195, 1002)  # crop KITTI images due to limited FOV of Audi's LiDAR
    else:
        print("Error: Unknown dataset '%s'" % chosen_dataset)
        exit()

    # set maximum distance to KITTI
    max_distance = 80.0

    for idx in range(len(dataset.files_list)):
        #idx=dataset.files_list.index("20180810142822_lidar_frontcenter_000046936.npz")
        lidar = dataset.get_lidar(idx)
        img_kitti = kitti.get_image(idx=1)
        calib = dataset.get_calib(idx)
        calib_kitti = kitti.get_mean_calib()

        # take KITTI calibration for AUDI dataset, because each lidars (and corresponding bounding boxes) have no own
        # unique calibrations
        if chosen_dataset == 'audi':
            rect_pts = calib_kitti.project_velo_to_rect(lidar[:, 0:3])
        # take own calibration, since each lidar has its own unique calibration, so bounding boxes for each lidar are
        # shifted differently (KITTI and Lyft)
        else:
            rect_pts = calib.project_velo_to_rect(lidar[:, 0:3])

        points_2d = calib_kitti.project_rect_to_image(rect_pts)

        # collect points in fov
        pts_image, pts_xyz_mask = get_mask(rect_pts, points_2d, imgsize=img_kitti.size)

        # project points onto image
        grid_img = lidarimg2grid(pts_image, img_kitti.size, max_distance)

        # crop image horizontally. I.e. for audi's fov, because only objects inside audi's fov is labeled and kitti's
        # fov is bigger
        grid_img = grid_img[crop[0]:crop[1], crop[2]:crop[3]]

        # show bounding boxes
        #object_labels = dataset.get_label(idx)
        #labels.draw_2d_bboxes_label(grid_img, object_labels)  # doesn't support cropping
        #labels.draw_bboxes(grid_img, object_labels, calib_kitti, max_distance, grid_img.shape, crop)
        #labels.draw_bboxes_audi(grid_img, object_labels, calib_kitti, max_distance, grid_img.shape, crop)

        # export LiDAR FOV image
        output_name = dataset.files_list[idx].split('.')[0]
        output_path = os.path.join(dataset.lidar_fov_path, output_name)

        # export image by type
        if export_type == 'show':
            # only show output instead of saving to numpy file
            cv2.imshow("Camera FOV Projected LiDAR: %s" % output_name, grid_img)
            cv2.waitKey()
            exit()
        elif export_type == 'numpy':
            # save as numpy array
            output_path_npy = output_path + '.npy'
            np.save(output_path_npy, grid_img)
            print("Image saved to ", output_path_npy)
        elif export_type == 'plot':
            # save as plot
            setup_plt()
            output_path_plt = output_path + '.png'
            save_as_grid_plt(grid_img, output_path_plt)
            print("Image saved to ", output_path_plt)
        elif export_type == 'png':
            # save as (resized) png
            output_path_png = output_path + '.png'
            # transform to PIL image and resize as in UNIT
            grid_img_pil = Image.fromarray(grid_img)
            if chosen_dataset not in ('audi', 'kitti_cropped'):
                grid_img_pil_resized = grid_img_pil.resize((844, 256), Image.BILINEAR)  # lyft2kitti: same method as torch resizing
            else:
                grid_img_pil_resized = grid_img_pil.resize((548, 256), Image.BILINEAR)  # audi2kitti: same method as torch resizing
            grid_img_np = np.asarray(grid_img_pil_resized)
            imageio.imwrite(output_path_png, grid_img_np)
            print("Image saved to ", output_path_png)

            # export YOLO formatted label
            object_labels = dataset.get_label(idx)
            if chosen_dataset != 'audi':
                bbox_2d_list = labels.get_2d_bboxes(object_labels, calib_kitti, max_distance, grid_img.shape, crop)
            else:
                bbox_2d_list = labels.get_2d_bboxes_audi(object_labels, calib_kitti, max_distance, grid_img.shape, crop)
            output_path_yolo_label = output_path + '.txt'
            labels.export_yolo_label(bbox_2d_list, class_names_to_id, output_path_yolo_label, grid_img.shape)
        else:
            print("Error: Unknown export type '%s'" % export_type)
            exit()


if __name__ == "__main__":
    _chosen_dataset = "kitti_cropped"  # 'kitti', 'lyft', 'audi', 'kitti_cropped'
    export_type="png"  # 'show', 'numpy', 'plot', 'png'
    main(_chosen_dataset, export_type=export_type)  # just show FOV images
