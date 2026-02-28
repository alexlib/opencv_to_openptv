import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import matplotlib.pyplot as plt


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(camera_id=0, images_prefix='', checkerboard_rows=4, checkerboard_columns=7, box_size_scale=None, criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001), show=True):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 

    rows = checkerboard_rows
    columns = checkerboard_columns
    world_scaling = box_size_scale #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)

            # Show checkerboard detection success

            if show:
                frame_vis = frame.copy()
                cv.drawChessboardCorners(frame_vis, (rows, columns), corners, ret)
                plt.figure()
                plt.imshow(cv.cvtColor(frame_vis, cv.COLOR_BGR2RGB))
                plt.title(f'Checkerboard detected: {images_names[i]}')
                plt.axis('off')
                plt.show()


            objpoints.append(objp)
            imgpoints.append(corners)


    # cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)


    # Save calibration results to file using camera name from images_prefix
    camera_name = f'camera{camera_id}'
    save_camera_intrinsics(cmtx, dist, camera_name)
    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


#open both cameras and take calibration frames


#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1, rows, columns, box_size_scale):
    
    # Read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    if not c0_images_names or not c1_images_names:
        raise RuntimeError(f"No images found for patterns: {frames_prefix_c0}, {frames_prefix_c1}")
    
    if len(c0_images_names) != len(c1_images_names):
        raise RuntimeError(f"Number of left/right images does not match: {len(c0_images_names)} vs {len(c1_images_names)}")
    
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    world_scaling = box_size_scale
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]
    imgpoints_left = []
    imgpoints_right = []
    objpoints = []
    import matplotlib.pyplot as plt
    valid_pairs = 0
    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), criteria)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), criteria)
        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            # Visualize both frames with detected corners
            f0_vis = frame0.copy()
            f1_vis = frame1.copy()
            cv.drawChessboardCorners(f0_vis, (rows, columns), corners1, c_ret1)
            cv.drawChessboardCorners(f1_vis, (rows, columns), corners2, c_ret2)
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.imshow(cv.cvtColor(f0_vis, cv.COLOR_BGR2RGB))
            plt.title('Stereo Left')
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(cv.cvtColor(f1_vis, cv.COLOR_BGR2RGB))
            plt.title('Stereo Right')
            plt.axis('off')
            plt.show()
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            valid_pairs += 1
    if valid_pairs == 0:
        raise RuntimeError("No valid stereo pairs with detected checkerboards found.")
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
        mtx1, dist1, (width, height), criteria=criteria, flags=stereocalibration_flags)
    print('rmse: ', ret)
    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    #open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(cv.cvtColor(frame0, cv.COLOR_BGR2RGB))
        plt.title('frame0')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(cv.cvtColor(frame1, cv.COLOR_BGR2RGB))
        plt.title('frame1')
        plt.axis('off')
        plt.show()

        # k = cv.waitKey(1)
        # if k == 27: break

    cv.destroyAllWindows()

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(cv.cvtColor(frame0, cv.COLOR_BGR2RGB))
    plt.title('frame0')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cv.cvtColor(frame1, cv.COLOR_BGR2RGB))
    plt.title('frame1')
    plt.axis('off')
    plt.show()
    # cv.waitKey(0)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1


def triangulate(mtx1, mtx2, R, T, frames_prefix_c0="frames/synched/D2/*.png", frames_prefix_c1='frames/synched/J2/*.png', rows=4, columns=7, show_2d=True, show_3d=True):
    """
    Triangulate chessboard corners from two selected frames and plot the 3D reconstruction.
    Args:
        mtx1, mtx2: Intrinsic matrices for camera 1 and 2
        R, T: Rotation and translation from stereo calibration
        synched_folder: Glob pattern for stereo image pairs
        rows, columns: Checkerboard inner corners
        show_2d: Show 2D corner detection plots
        show_3d: Show 3D triangulation plot
    Returns:
        p3ds: Nx3 array of triangulated 3D points
    """
       # Read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    if not c0_images_names or not c1_images_names:
        raise RuntimeError(f"No images found for patterns: {frames_prefix_c0}, {frames_prefix_c1}")
    
    if len(c0_images_names) != len(c1_images_names):
        raise RuntimeError(f"Number of left/right images does not match: {len(c0_images_names)} vs {len(c1_images_names)}")
    
    # Load and undistort images
    dist1 = None
    dist2 = None
    try:
        _, dist1 = load_intrinsics('camera0')
        _, dist2 = load_intrinsics('camera1')
    except Exception:
        pass
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]
    # Undistort if distortion coefficients are available
    if dist1 is not None and dist2 is not None:
        c0_images = [cv.undistort(img, mtx1, dist1) for img in c0_images]
        c1_images = [cv.undistort(img, mtx2, dist2) for img in c1_images]


    frame1 = c0_images[0]
    frame2 = c1_images[0]
    if frame1 is None or frame2 is None:
        raise RuntimeError(f"Could not load images: {c0_images_names[0]}, {c1_images_names[0]}")
        
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), criteria)
    ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), criteria)
    
    
    if not ret1 or not ret2:
        raise RuntimeError("Could not detect chessboard corners in one or both images.")
    corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
    uvs1 = corners1.reshape(-1, 2)
    uvs2 = corners2.reshape(-1, 2)



    if show_2d:
        import matplotlib.pyplot as plt
        plt.imshow(frame1[:,:,[2,1,0]])
        plt.scatter(uvs1[:,0], uvs1[:,1])
        plt.title('Detected corners in frame 1')
        plt.show()
        plt.imshow(frame2[:,:,[2,1,0]])
        plt.scatter(uvs2[:,0], uvs2[:,1])
        plt.title('Detected corners in frame 2')
        plt.show()
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
    def DLT(P1, P2, point1, point2):
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
        return Vh[3,0:3]/Vh[3,3]
    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    if show_3d:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p3ds[:,0], p3ds[:,1], p3ds[:,2], c='b', marker='o')
        ax.set_title('Triangulated 3D Chessboard Corners')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    return p3ds

# Triangulation: only read from stored camera_parameters
def load_intrinsics(camera_name):
    fname = f'camera_parameters/{camera_name}_intrinsics.dat'
    with open(fname, 'r') as f:
        lines = f.readlines()
    cmtx = []
    dist = []
    mode = None
    for line in lines:
        if 'intrinsic' in line:
            mode = 'intrinsic'
            continue
        if 'distortion' in line:
            mode = 'distortion'
            continue
        if mode == 'intrinsic' and line.strip():
            cmtx.append([float(x) for x in line.strip().split()])
        if mode == 'distortion' and line.strip():
            dist.extend([float(x) for x in line.strip().split()])
    return np.array(cmtx), np.array([dist])

def load_extrinsics(fname = 'camera_parameters/camera1_rot_trans.dat'):

    with open(fname, 'r') as f:
        lines = f.readlines()
    R = []
    T = []
    mode = None
    for line in lines:
        if 'R:' in line:
            mode = 'R'
            continue
        if 'T:' in line:
            mode = 'T'
            continue
        if mode == 'R' and line.strip():
            R.append([float(x) for x in line.strip().split()])
        if mode == 'T' and line.strip():
            T.append([float(x) for x in line.strip().split()])
    return np.array(R), np.array(T).reshape(3,1)