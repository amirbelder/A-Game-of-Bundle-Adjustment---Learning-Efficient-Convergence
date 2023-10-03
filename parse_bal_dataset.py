
"""
First:
Projection - each is represented by a single line
cam_idx, point_idx, x_pixel, y_pixel

Second:
Camera - each is represented by 9 lines
3 lines - R - Rotation in Rodriguez-Euler - azimuth, pitch, roll // find if azimuth is negative or positive and fix to Quat format
3 lines - t - translation - either x,y,z or -Rt (to reach x,y,z)
1 line - f - focal
1 line - k1
1 line - k2

Third:
Points in real world
Each represented by 3 lines - x,y,z

TODO:
1. first line is the number of points and cameras - save it
2. save which camera sees each point in a dictionary
3. find a way to convert their R-t to my Quat-t

Things I may need:
We use a pinhole camera model; the parameters we estimate for each camera area rotation R, a translation t, a focal length f and two radial distortion parameters k1 and k2. The formula for projecting a 3D point X into a camera R,t,f,k1,k2 is:

P  =  R * X + t       (conversion from world to camera coordinates)
p  = -P / P.z         (perspective division)
p' =  f * r(p) * p    (conversion to pixel coordinates)

where P.z is the third (z) coordinate of P. In the last equation, r(p) is a function that computes a scaling factor to undo the radial distortion:

r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
This is what I'll have to use if I don't use their projections

<num_cameras> <num_points> <num_observations>
<camera_index_1> <point_index_1> <x_1> <y_1>
...
<camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
<camera_1>
...
<camera_num_cameras>
<point_1>
...
<point_num_points>

https://grail.cs.washington.edu/projects/bal/
"""

import re
from scipy.spatial.transform import Rotation as R
import numpy as np

def convert_from_Rodriguez_to_Quat(rotation_matrix:dict = None):
    if rotation_matrix is None:
        return
    cam_quat_matrix = {}
    for k,v in enumerate(rotation_matrix):
        cam_quat_matrix[k] = []
        for idx, comp in enumerate(v):
            if idx == 0:
                r = R.from_mrp(comp)
                quat = r.as_quat()
                cam_quat_matrix[k].append(quat)
            else:
                cam_quat_matrix[k].append(comp)
    return cam_quat_matrix

def parse_txt(txt_path=None):
    if txt_path is None:
        return
    with open (txt_path, 'r') as f:
        num_cameras = -1
        num_points = -1
        num_projections = -1
        cameras_to_point_dict = None

        rotation_matrix = []
        points_matrix = []
        local_var_in_progres_idx = 0
        first_appearance = -1

        first_cam = False
        first_point = False

        for i, line in enumerate(f.readlines()):
            parts = re.split(string=line[:-1], pattern=' ')
            if i == 0:
                num_cameras = int(parts[0])
                num_points = int(parts[1])
                num_projections = int(parts[2])
                cameras_to_point_dict = {i: [] for i in range(num_cameras)}
                continue

            #if len(parts) > 1:
            elif i < num_projections + 1:
                # we are at the projections part
                cam_num = int(parts[0])
                point_num = int(parts[1])
                #projection = (float(parts[2]), float(parts[3]))
                cameras_to_point_dict[cam_num].append(point_num)
            #else:
            #    first_appearance = i

            #elif first_appearance > 0 and i < first_appearance + 9 * num_cameras:
            elif i < num_projections + 9 * num_cameras + 1:
                if first_cam is False:
                    first_cam = True
                    local_var_in_progres_idx = 0
                    print(i)
                # we are at the cameras part, local_var_in_progres_idx < 9
                #if len(parts) > 1:
                #    continue
                if local_var_in_progres_idx == 0:
                    local_R = []
                    local_t = []
                    local_f = []
                    local_k1 = []
                    local_k2 = []
                value = float(line[:-1].upper())
                if local_var_in_progres_idx < 3:
                    local_R.append(value)
                elif local_var_in_progres_idx < 6:
                    local_t.append(value)
                elif local_var_in_progres_idx == 6:
                    local_f.append(value)
                elif local_var_in_progres_idx == 7:
                    local_k1.append(value)
                else:
                    local_k2.append(value)
                    local_var_in_progres_idx = 0
                    rotation_matrix.append([local_R, local_t, local_f, local_k1, local_k2])
                    continue
                local_var_in_progres_idx+=1
            #elif first_appearance > 0:
            else:
                # we are at the points part, local_var_in_progres_idx < 3
                if first_point is False:
                    first_point = True
                    local_var_in_progres_idx = 0
                    print(i)

                if local_var_in_progres_idx == 0:
                    local_point = []
                if local_var_in_progres_idx == 2:
                    local_var_in_progres_idx = 0
                    value = float(line[:-1].upper())
                    local_point.append(value)
                    points_matrix.append(local_point)
                    continue

                value = float(line[:-1].upper())
                local_point.append(value)
                local_var_in_progres_idx+=1

        #print('cameras_to_point_dict', 'rotation_matrix', 'len(points_matrix)')
        #print(len(cameras_to_point_dict), len(rotation_matrix), len(points_matrix))
        return cameras_to_point_dict, rotation_matrix, points_matrix


def parse_bal_file(bal_txt_file = None):
    if bal_txt_file is None:
        exit('Bad (None) bal txt reference')
    cameras_to_point_dict, rotation_matrix, points_matrix = parse_txt(txt_path=bal_txt_file)
    cam_quat_matrix = convert_from_Rodriguez_to_Quat(rotation_matrix=rotation_matrix)
    return cameras_to_point_dict, cam_quat_matrix, points_matrix

def main():
    bal_txt_path = './BAL/data'
    parse_bal_file(bal_txt_file=bal_txt_path)


if __name__ == "__main__":
    main()
