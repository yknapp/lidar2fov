"""
    Projection of LiDAR point cloud to Field-of-View(FOV) of the camera.
    Supported datasets: KITTI, Lyft Level 5, Audi A2D2
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from dataset import Kitti, Lyft, Audi


def get_mask(rect_pts, points_2d, imgsize):
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < imgsize[0]) & \
           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < imgsize[1])
    mask = mask & (rect_pts[:, 2] >= 2)

    # pts_on_image_with_depth = np.append(points_2d[mask, 0:2], rect_pts[mask, 2], axis=1)

    pts_on_image_with_depth = np.zeros([mask.sum(), 3])
    pts_on_image_with_depth[:, 0:2] = points_2d[mask, 0:2]
    pts_on_image_with_depth[:, 2] = rect_pts[mask, 2]

    # return points_2d[mask, 0:2], rect_pts[mask, ]

    return pts_on_image_with_depth, rect_pts[mask,]


def lidarimg2grid(pts_image, img_shape):
    size_0 = img_shape[0]
    size_1 = img_shape[1]
    grid = np.zeros(img_shape[:2])
    for p in pts_image:
        i = int(p[0]) - 1
        j = int(p[1]) - 1

        value = p[2]   # representation of depth, i.e. p[2], 1/p[2], log(p[2])

        grid[i, j] = value

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
        if i + 2 < size_0 and j + 1 >= 0:
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


def save_as_scatter_plt(pts_image, output_path):
    plt.scatter(pts_image[:, 0], -pts_image[:, 1], c=pts_image[:, 2], cmap='gray', s=1)  # negative, so image isn't upside down
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("scatter plot size: ", Image.open(output_path).size)


def save_as_grid_plt(pts_image, img_size, output_path):
    # projection lidar to image
    grid = lidarimg2grid(pts_image, img_size)

    plt.imsave(output_path, grid)
    plt.close()
    print("Grid size: ", Image.open(output_path).size)
    

def main(chosen_dataset):
    dataset = None
    kitti = Kitti()
    if chosen_dataset == "kitti":
        dataset = Kitti()
    elif chosen_dataset == "lyft":
        dataset = Lyft()
    elif chosen_dataset == "audi":
        dataset = Audi()
    else:
        print("Error: Unknown dataset '%s'" % chosen_dataset)
        exit()

    for idx in range(len(dataset.files_list)):
        lidar = dataset.get_lidar(idx)
        img = kitti.get_image(idx)
        calib = kitti.get_calib(idx)
        rect_pts = calib.project_velo_to_rect(lidar[:, 0:3])
        points_2d = calib.project_rect_to_image(rect_pts)
        pts_image, pts_xyz_mask = get_mask(rect_pts, points_2d, imgsize=img.size)

        # save as plots
        setup_plt()
        output_name = dataset.files_list[idx].split('.')[0] + '.png'
        output_path = os.path.join(dataset.lidar_fov_path, output_name)
        #save_as_scatter_plt(pts_image, output_path)
        save_as_grid_plt(pts_image, img.size, output_path)


if __name__ == "__main__":
    _chosen_dataset = "lyft"  # 'kitti', 'lyft', 'audi'
    main(_chosen_dataset)
