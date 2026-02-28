""" Source: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html """

import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import yaml

import calib

def main():
    # Load calibration settings from YAML
    with open('calibration_settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)

    checkerboard_rows = settings['checkerboard_rows']
    checkerboard_columns = settings['checkerboard_columns']
    box_size_scale = settings['checkerboard_box_size_scale']

    # Calibrate and store intrinsics once
    mtx1, dist1 = calib.calibrate_camera_for_intrinsic_parameters(
        camera_id=0,
        images_prefix='frames/camera0/*',
        checkerboard_rows=checkerboard_rows,
        checkerboard_columns=checkerboard_columns,
        box_size_scale=box_size_scale
    )
    mtx2, dist2 = calib.calibrate_camera_for_intrinsic_parameters(
        camera_id=1,
        images_prefix='frames/camera1/*',
        checkerboard_rows=checkerboard_rows,
        checkerboard_columns=checkerboard_columns,
        box_size_scale=box_size_scale
    )
    # Stereo calibration and store extrinsics
    R, T = calib.stereo_calibrate(
        mtx1, dist1, mtx2, dist2,
        'frames/synched/camera0/*.png', 'frames/synched/camera1/*.png',
        checkerboard_rows, checkerboard_columns, box_size_scale
    )
    calib.save_extrinsic_calibration_parameters(np.eye(3), np.zeros((3,1)), R, T)  # Save stereo extrinsics



    mtx1, dist1 = calib.load_intrinsics('camera0')
    mtx2, dist2 = calib.load_intrinsics('camera1')
    R, T = calib.load_extrinsics('camera_parameters/camera1_rot_trans.dat')

    print("Triangulating and visualizing 3D chessboard corners from stereo images ...")
    p3ds = calib.triangulate(
        mtx1, mtx2, R, T,
        'frames/synched/camera0/*.png',
        'frames/synched/camera1/*.png',
        rows=checkerboard_rows,
        columns=checkerboard_columns,
        show_2d=True,
        show_3d=True
    )
    print("3D triangulation and visualization complete.")
    print("Triangulated 3D points (first 5):\n", p3ds[:5])
    print("Calibration complete.")
    # All unreachable and legacy code removed. Only main logic remains.
if __name__ == "__main__":
    main()