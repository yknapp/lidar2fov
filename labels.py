import numpy as np
import cv2

from object import AudiObject3d


def draw_2d_bboxes_label(grid_img, object_labels):
    #grid_img = np.array(grid_img)
    grid_img *= 255
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

        if np.any(bbox_3d_rect[:, 2] < 0.1) or np.all(bbox_3d_rect[:, 2] > max_distance):
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