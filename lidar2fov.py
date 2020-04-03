"""
    Projection of LiDAR point cloud to Field-of-View(FOV) of the camera.
    Supported datasets: KITTI, Lyft Level 5, Audi A2D2
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from dataset import Kitti, Lyft, Audi
from object import AudiObject3d


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


def draw_2d_bboxes_label(grid_img, object_labels):
    #grid_img = np.array(grid_img)
    grid_img = grid_img.astype(np.uint8)
    line_thickness = 1
    for object_label in object_labels:
        grid_img = cv2.line(grid_img, (int(object_label.xmin), int(object_label.ymin)), (int(object_label.xmax), int(object_label.ymin)), (255, 255, 0), line_thickness)
        grid_img = cv2.line(grid_img, (int(object_label.xmin), int(object_label.ymin)), (int(object_label.xmin), int(object_label.ymax)), (255, 255, 0), line_thickness)
        grid_img = cv2.line(grid_img, (int(object_label.xmax), int(object_label.ymin)), (int(object_label.xmax), int(object_label.ymax)), (255, 255, 0), line_thickness)
        grid_img = cv2.line(grid_img, (int(object_label.xmin), int(object_label.ymax)), (int(object_label.xmax), int(object_label.ymax)), (255, 255, 0), line_thickness)
    cv2.imshow("Bounding boxes", grid_img)
    cv2.waitKey()
    exit()


def draw_bboxes(grid_img, object_labels, calib, max_distance):
    #grid_img = np.array(grid_img)
    # to view grid image, map [0, 1] to [0, 255]
    grid_img *= 255
    grid_img = grid_img.astype(np.uint8)
    line_thickness = 1

    # get object coordinates
    for object_label in object_labels:
        bbox_2d, bbox_2d_rect = calib.compute_box_3d(object_label, max_distance)

        if bbox_2d is not None:
            min_x = np.amin(bbox_2d[:, 0])
            max_x = np.amax(bbox_2d[:, 0])
            min_y = np.amin(bbox_2d[:, 1])
            max_y = np.amax(bbox_2d[:, 1])

            # draw 2D bounding box into grid image
            grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(max_x), int(min_y)), (255, 255, 0), line_thickness)
            grid_img = cv2.line(grid_img, (int(min_x), int(max_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)
            grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(min_x), int(max_y)), (255, 255, 0), line_thickness)
            grid_img = cv2.line(grid_img, (int(max_x), int(min_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)

    # show image
    cv2.imshow("Bounding boxes", grid_img)
    cv2.waitKey()
    exit()


def draw_bboxes_audi(grid_img, object_labels, calib, max_distance):
    # to view grid image, map [0, 1] to [0, 255]
    grid_img *= 255
    grid_img = grid_img.astype(np.uint8)
    line_thickness = 1

    # get object coordinates
    for object_label in object_labels:
        axis = object_label.axis
        angle = object_label.ry
        bbox_rotation = AudiObject3d.axis_angle_to_rotation_mat(axis, angle)
        bbox_3d = AudiObject3d.get_3d_bbox_points(object_label, bbox_rotation)
        bbox_3d_rect = calib.project_velo_to_rect(bbox_3d)

        #if np.any(bbox_3d_rect[2, :] < 0.1) or
        # This condition is from KITTI's "calib.compute_box_3d()" method.
        # Even if this condition is removed there, the bounding boxes are still not visible.
        # BESIDES: these invisible bounding boxes are also not displayed with method "draw_2d_bboxes_label", which
        # means that these bounding boxes are not wanted for camera 2D object detection
        if np.all(bbox_3d_rect[2, :] > max_distance):

            continue

        bbox_2d = calib.project_rect_to_image(bbox_3d_rect)

        if bbox_2d is not None:
            min_x = np.amin(bbox_2d[:, 0])
            max_x = np.amax(bbox_2d[:, 0])
            min_y = np.amin(bbox_2d[:, 1])
            max_y = np.amax(bbox_2d[:, 1])

            # draw 2D bounding box into grid image
            grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(max_x), int(min_y)), (255, 255, 0), line_thickness)
            grid_img = cv2.line(grid_img, (int(min_x), int(max_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)
            grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(min_x), int(max_y)), (255, 255, 0), line_thickness)
            grid_img = cv2.line(grid_img, (int(max_x), int(min_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)

    # show image
    cv2.imshow("Bounding boxes", grid_img)
    cv2.waitKey()
    exit()


def save_as_grid_plt(grid_img, output_path):
    plt.imsave(output_path, grid_img)
    plt.close()
    print("Grid size: ", Image.open(output_path).size)
    

def main(chosen_dataset, crop=(None, None, None, None), show=False):
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

    # set maximum distance to KITTI
    max_distance = 80.0

    for idx in range(len(dataset.files_list)):
        idx=3
        lidar = dataset.get_lidar(idx)
        img = kitti.get_image(idx=idx)
        calib = dataset.get_calib(idx)
        calib_kitti = kitti.get_calib(idx)

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
        pts_image, pts_xyz_mask = get_mask(rect_pts, points_2d, imgsize=img.size)

        # project points onto image
        grid_img = lidarimg2grid(pts_image, img.size, max_distance)

        # show bounding boxes
        #object_labels = dataset.get_label(idx)
        #draw_2d_bboxes_label(grid_img, object_labels)
        #draw_bboxes(grid_img, object_labels, calib_kitti, max_distance)
        #draw_bboxes_audi(grid_img, object_labels, calib_kitti, max_distance)

        # crop image horizontally. I.e. for audi's fov, because only objects inside audi's fov is labeled and kitti's
        # fov is bigger
        grid_img = grid_img[crop[0]:crop[1], crop[2]:crop[3]]

        # only show output instead of saving to numpy file
        if show:
            cv2.imshow("Camera FOV Projected LiDAR", grid_img)
            cv2.waitKey()
            exit()

        # save as numpy array
        output_name = dataset.files_list[idx].split('.')[0] + '.npy'
        output_path = os.path.join(dataset.lidar_fov_path, output_name)
        np.save(output_path, grid_img)
        # save as plot
        #setup_plt()
        #output_name = dataset.files_list[idx].split('.')[0] + '.png'
        #output_path = os.path.join(dataset.lidar_fov_path, output_name)
        #save_as_grid_plt(grid_img, output_path)


if __name__ == "__main__":
    _chosen_dataset = "kitti"  # 'kitti', 'lyft', 'audi'
    show = True  # if true, then FOV projected lidar images are just shown and not stored to numpy files
    #main(_chosen_dataset, show)
    main(_chosen_dataset, (None, None, 195, 1002), show)  # crops FOV image, if crop != None; structure: (y_min, y_max, x_min, x_max)
