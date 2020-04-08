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


def draw_bboxes(grid_img, object_labels, calib, max_distance, img_shape, crop):
    #grid_img = np.array(grid_img)
    # to view grid image, map [0, 1] to [0, 255]
    grid_img *= 255
    grid_img = grid_img.astype(np.uint8)
    line_thickness = 1

    # get image resolution
    img_height = img_shape[0]
    img_width = img_shape[1]

    # get object coordinates
    for object_label in object_labels:
        bbox_2d, bbox_2d_rect = calib.compute_box_3d(object_label, max_distance)

        if bbox_2d is not None:
            # shift bbox coordinates according to crop (crop=(y_min, y_max, x_min, x_max))
            x_shift = crop[2] if crop[2] != None else 0
            y_shift = crop[0] if crop[0] != None else 0

            # get extrema of x and y
            min_x = np.amin(bbox_2d[:, 0]) - x_shift
            max_x = np.amax(bbox_2d[:, 0]) - x_shift
            min_y = np.amin(bbox_2d[:, 1]) - y_shift
            max_y = np.amax(bbox_2d[:, 1]) - y_shift

            if (min_x >= img_width) or (max_x <= 0) or (min_y >= img_height) or (max_y <= 0):
                # drop boxes which are not inside the camera FOV
                print("Skip bbox, because it's outside of FOV image")

            else:
                # clip bounding box extrema to image size
                min_x = max(0, min(img_width - 1, min_x))
                min_y = max(0, min(img_height - 1, min_y))
                max_x = min(img_width - 1, max(0, max_x))
                max_y = min(img_height - 1, max(0, max_y))

                # draw 2D bounding box into grid image
                grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(max_x), int(min_y)), (255, 255, 0), line_thickness)
                grid_img = cv2.line(grid_img, (int(min_x), int(max_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)
                grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(min_x), int(max_y)), (255, 255, 0), line_thickness)
                grid_img = cv2.line(grid_img, (int(max_x), int(min_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)

    # show image
    cv2.imshow("Bounding boxes", grid_img)
    cv2.waitKey()
    exit()


def draw_bboxes_audi(grid_img, object_labels, calib, max_distance, img_shape, crop):
    # to view grid image, map [0, 1] to [0, 255]
    grid_img *= 255
    grid_img = grid_img.astype(np.uint8)
    line_thickness = 1

    # get image resolution
    img_height = img_shape[0]
    img_width = img_shape[1]

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
            # shift bbox coordinates according to crop (crop=(y_min, y_max, x_min, x_max))
            x_shift = crop[2] if crop[2] != None else 0
            y_shift = crop[0] if crop[0] != None else 0

            # get extrema of x and y
            min_x = np.amin(bbox_2d[:, 0]) - x_shift
            max_x = np.amax(bbox_2d[:, 0]) - x_shift
            min_y = np.amin(bbox_2d[:, 1]) - y_shift
            max_y = np.amax(bbox_2d[:, 1]) - y_shift

            if (min_x >= img_width) or (max_x <= 0) or (min_y >= img_height) or (max_y <= 0):
                # drop boxes which are not inside the camera FOV
                print("Skip bbox, because it's outside of FOV image")

            else:
                # clip bounding box extrema to image size
                min_x = max(0, min(img_width-1, min_x))
                min_y = max(0, min(img_height-1, min_y))
                max_x = min(img_width-1, max(0, max_x))
                max_y = min(img_height-1, max(0, max_y))

                # draw 2D bounding box into grid image
                grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(max_x), int(min_y)), (255, 255, 0), line_thickness)
                grid_img = cv2.line(grid_img, (int(min_x), int(max_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)
                grid_img = cv2.line(grid_img, (int(min_x), int(min_y)), (int(min_x), int(max_y)), (255, 255, 0), line_thickness)
                grid_img = cv2.line(grid_img, (int(max_x), int(min_y)), (int(max_x), int(max_y)), (255, 255, 0), line_thickness)

    # show image
    cv2.imshow("Bounding boxes", grid_img)
    cv2.waitKey()
    exit()


def get_2d_bboxes(object_labels, calib, max_distance, img_shape, crop):
    # get image resolution
    img_height = img_shape[0]
    img_width = img_shape[1]

    bbox_2d_list = []
    # get object coordinates
    for object_label in object_labels:
        bbox_2d, bbox_2d_rect = calib.compute_box_3d(object_label, max_distance)
        if bbox_2d is not None:
            # shift bbox coordinates according to crop (crop=(y_min, y_max, x_min, x_max))
            x_shift = crop[2] if crop[2] != None else 0
            y_shift = crop[0] if crop[0] != None else 0

            # get extrema of x and y
            min_x = np.amin(bbox_2d[:, 0]) - x_shift
            max_x = np.amax(bbox_2d[:, 0]) - x_shift
            min_y = np.amin(bbox_2d[:, 1]) - y_shift
            max_y = np.amax(bbox_2d[:, 1]) - y_shift

            if (min_x >= img_width) or (max_x <= 0) or (min_y >= img_height) or (max_y <= 0):
                # drop boxes which are not inside the camera FOV
                print("Skip bbox, because it's outside of FOV image")

            else:
                # clip bounding box extrema to image size
                min_x = max(0, min(img_width - 1, min_x))
                min_y = max(0, min(img_height - 1, min_y))
                max_x = min(img_width - 1, max(0, max_x))
                max_y = min(img_height - 1, max(0, max_y))
                bbox_2d_list.append([object_label.class_name, min_x, min_y, max_x, max_y])
    return bbox_2d_list


def get_2d_bboxes_audi(object_labels, calib, max_distance, img_shape, crop):
    bbox_2d_list = []

    # get image resolution
    img_height = img_shape[0]
    img_width = img_shape[1]

    # get object coordinates
    for object_label in object_labels:
        axis = object_label.axis
        angle = object_label.ry
        bbox_rotation = AudiObject3d.axis_angle_to_rotation_mat(axis, angle)
        bbox_3d = AudiObject3d.get_3d_bbox_points(object_label, bbox_rotation)
        bbox_3d_rect = calib.project_velo_to_rect(bbox_3d)

        # only draw 3d bounding box for objs in front of the camera and smaller than maximum distance
        if np.any(bbox_3d_rect[:, 2] < 0.1) or np.all(bbox_3d_rect[:, 2] > max_distance):
            continue

        bbox_2d = calib.project_rect_to_image(bbox_3d_rect)
        if bbox_2d is not None:
            # shift bbox coordinates according to crop (crop=(y_min, y_max, x_min, x_max))
            x_shift = crop[2] if crop[2] != None else 0
            y_shift = crop[0] if crop[0] != None else 0

            # get extrema of x and y
            min_x = np.amin(bbox_2d[:, 0]) - x_shift
            max_x = np.amax(bbox_2d[:, 0]) - x_shift
            min_y = np.amin(bbox_2d[:, 1]) - y_shift
            max_y = np.amax(bbox_2d[:, 1]) - y_shift

            if (min_x >= img_width) or (max_x <= 0) or (min_y >= img_height) or (max_y <= 0):
                # drop boxes which are not inside the camera FOV
                print("Skip bbox, because it's outside of FOV image")

            else:
                # clip bounding box extrema to image size
                min_x = max(0, min(img_width - 1, min_x))
                min_y = max(0, min(img_height - 1, min_y))
                max_x = min(img_width - 1, max(0, max_x))
                max_y = min(img_height - 1, max(0, max_y))
                bbox_2d_list.append([object_label.class_name, min_x, min_y, max_x, max_y])
    return bbox_2d_list


def convert_bbox_to_yolo_format(bbox, img_shape):
    """
        Convert information of BoundingBox object into YOLO format: x_min, x_max, y_min, y_max coordinates to
        x, y coordinate, width, height relative to image shape.
        :param bbox: BoundingBox object
        :param img_shape: shape of the image containing an object with a bounding box
        :return:
            x: x coordinate of bounding box in image relative to image width
            y: y coordinate of bounding box in image relative to image height
            w: width of bounding box relative to image width
            h: height of bounding box relative to image height
    """
    img_width = 1. / img_shape[1]
    img_height = 1. / img_shape[0]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * img_width
    w = w * img_width
    y = y * img_height
    h = h * img_height
    return x, y, w, h


def write_to_txt_file(output_path, content_str):
    file = open(output_path, 'w')
    file.write(content_str)
    file.close()


def export_yolo_label(bbox_2d_list, class_names_to_id, output_path_yolo_label, img_shape):
    content_str = ''
    for bbox_2d in bbox_2d_list:
        class_name = bbox_2d[0]
        # only process objects of classes included in class_names_to_id dict
        if class_name in class_names_to_id.keys():
            class_id = class_names_to_id[class_name]
        else:
            print("UNWANTED CLASS: ", class_name)
            continue

        # calculate yolo format values
        x, y, w, h = convert_bbox_to_yolo_format(bbox_2d[1:], img_shape)
        if x < 0 or y < 0 or w < 0 or h< 0:
            print("LABEL < 0: x: %s, y: %s, w: %s, h: %s" % (x, y, w, h))
            exit()
        content_str += ' '.join([str(class_id), str(x), str(y), str(w), str(h)]) + '\n'

        # write to text file
        write_to_txt_file(output_path_yolo_label, content_str)
