import numpy as np
from calibration import AudiCalibration

class KittiObject3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        # extract label, truncation, occlusion
        self.class_name = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.dis_to_cam = np.linalg.norm(self.t)
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = data[15] if data.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.class_name, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


class AudiObject3d(object):
    def __init__(self, bbox):
        self.class_name = bbox['class']
        self.truncation = float(bbox['truncation'])
        self.occlusion = float(bbox['occlusion'])
        self.alpha = float(bbox['alpha'])

        # 2d bounding box
        self.xmin = float(bbox['2d_bbox'][1])  # left
        self.xmax = float(bbox['2d_bbox'][3])  # right
        self.ymin = float(bbox['2d_bbox'][0])  # top
        self.ymax = float(bbox['2d_bbox'][2])  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # 3d bounding box
        self.h = float(bbox['size'][2])  # bbox height
        self.w = float(bbox['size'][0])  # bbox width
        self.l = float(bbox['size'][1])  # bbox length (in meters)
        center_x = float(bbox['center'][0])
        center_y = float(bbox['center'][1])
        center_z = float(bbox['center'][2])
        self.t = (center_x, center_y, center_z)  # location (x,y,z) in camera coord.
        self.ry = float(bbox['rot_angle'])
        self.score = -1.0
        self.level = self.get_obj_level()

        # additional things for testing
        self.axis = np.array(bbox['axis'])
        self.size = np.array(bbox['size'])

    @staticmethod
    def axis_angle_to_rotation_mat(axis, angle):
        return np.cos(angle) * np.eye(3) + \
               np.sin(angle) * AudiCalibration.skew_sym_matrix(axis) + \
               (1 - np.cos(angle)) * np.outer(axis, axis)

    @staticmethod
    def get_3d_bbox_points(object_label, rotation):
        half_size = object_label.size / 2.

        if half_size[0] > 0:
            # calculate unrotated corner point offsets relative to center
            brl = np.asarray([-half_size[0], +half_size[1], -half_size[2]])
            bfl = np.asarray([+half_size[0], +half_size[1], -half_size[2]])
            bfr = np.asarray([+half_size[0], -half_size[1], -half_size[2]])
            brr = np.asarray([-half_size[0], -half_size[1], -half_size[2]])
            trl = np.asarray([-half_size[0], +half_size[1], +half_size[2]])
            tfl = np.asarray([+half_size[0], +half_size[1], +half_size[2]])
            tfr = np.asarray([+half_size[0], -half_size[1], +half_size[2]])
            trr = np.asarray([-half_size[0], -half_size[1], +half_size[2]])

            # rotate points
            points = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
            points = np.dot(points, rotation.T)

            # add center position
            points = points + np.array(object_label.t)

        return points

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.class_name, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))