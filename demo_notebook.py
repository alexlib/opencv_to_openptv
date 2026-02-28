import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    """ Source: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html """

    import cv2 as cv
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import calib

    return calib, cv, glob, np, os, plt


@app.cell
def _(calib, cv, glob, np, os, plt):
    def calibrate_camera(images_folder):
        images_names = glob.glob(images_folder)
        print(f"Found {len(images_names)} images in {images_folder}:")
        for name in images_names:
            print("  ", name)
        images = []
        for imname in images_names:
            im = cv.imread(imname, 1)
            if im is None:
                print(f"Warning: Could not read image {imname}")
            images.append(im)
        if not images:
            raise RuntimeError(f"No images loaded from {images_folder}. Check your path and files.")

        # plt.figure(figsize = (10,10))
        # ax = [plt.subplot(2,2,i+1) for i in range(4)]
        #
        # for a, frame in zip(ax, images):
        #     a.imshow(frame[:,:,[2,1,0]])
        #     a.set_xticklabels([])
        #     a.set_yticklabels([])
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.show()

        #criteria used by checkerboard pattern detector.
        #Change this if the code can't find the checkerboard
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # === SET YOUR CHECKERBOARD SIZE HERE ===
        rows = 4      # Number of checkerboard rows (inner corners)
        columns = 7   # Number of checkerboard columns (inner corners)
        # Example: For a 9x6 checkerboard, set rows=6, columns=9
        # =======================================
        world_scaling = 1. #change this to the real world square size. Or not.

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



        failed_dir = "checkerboard_failed"
        os.makedirs(failed_dir, exist_ok=True)
        for idx, frame in enumerate(images):
            if frame is None:
                continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Try normal detection first
            ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

            # If not found, try adaptive thresholding
            used_adaptive = False
            if not ret:
                gray_adapt = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv.THRESH_BINARY, 11, 2)
                ret, corners = cv.findChessboardCorners(gray_adapt, (rows, columns), None)
                if ret:
                    used_adaptive = True
                    print(f"Checkerboard detected in image {images_names[idx]} using adaptive thresholding.")
            if ret == True:
                #Convolution size used to improve corner detection. Don't make this too large.
                conv_size = (11, 11)

                #opencv can attempt to improve the checkerboard coordinates
                corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
                cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
                # Save successful detection overlay for review
                out_path = os.path.join(failed_dir, f"success_{idx}{'_adapt' if used_adaptive else ''}.png")
                cv.imwrite(out_path, frame)
                print(f"Checkerboard detected in image {images_names[idx]}. Saved overlay to {out_path}")
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print(f"Checkerboard NOT detected in image {images_names[idx]}")
                # Save failed image for debugging
                fail_path = os.path.join(failed_dir, f"fail_{idx}.png")
                cv.imwrite(fail_path, frame)
                print(f"Saved failed detection image to {fail_path}")

        if not objpoints or not imgpoints:
            raise RuntimeError("No checkerboard corners found in any image. Check your images and checkerboard pattern.")



        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('distortion coeffs:', dist)
        print('Rs:\n', rvecs)
        print('Ts:\n', tvecs)

        return mtx, dist

    def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
        #read the synched frames
        images_names = glob.glob(frames_folder)
        images_names = sorted(images_names)
        c1_images_names = images_names[:len(images_names)//2]
        c2_images_names = images_names[len(images_names)//2:]

        c1_images = []
        c2_images = []
        for im1, im2 in zip(c1_images_names, c2_images_names):
            _im = cv.imread(im1, 1)
            c1_images.append(_im)

            _im = cv.imread(im2, 1)
            c2_images.append(_im)

        #change this if stereo calibration not good.
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        rows = 4 #number of checkerboard rows.
        columns = 7 #number of checkerboard columns.
        world_scaling = 1. #change this to the real world square size. Or not.

        #coordinates of squares in the checkerboard world space
        objp = np.zeros((rows*columns,3), np.float32)
        objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
        objp = world_scaling* objp

        #frame dimensions. Frames should be the same size.
        width = c1_images[0].shape[1]
        height = c1_images[0].shape[0]

        #Pixel coordinates of checkerboards
        imgpoints_left = [] # 2d points in image plane.
        imgpoints_right = []

        #coordinates of the checkerboard in checkerboard world space.
        objpoints = [] # 3d point in real world space

        for frame1, frame2 in zip(c1_images, c2_images):
            gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
            c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

            if c_ret1 == True and c_ret2 == True:
                corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
                plt.imshow(frame1)

                cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
                plt.imshow(frame2)

                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)

        stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                     mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)

        print(ret)
        return R, T




    def main():
        mtx1, dist1 = calibrate_camera(images_folder = 'frames/D2/*')
        mtx2, dist2 = calibrate_camera(images_folder = 'frames/J2/*')
        R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'frames/synched/*')
        print("Triangulating and visualizing 3D chessboard corners from stereo images ...")
        calib.triangulate(mtx1, mtx2, R, T, 'frames/synched/*.png', 4, 7, True, True)
        print("3D triangulation and visualization complete.")

    def _main_():
        main()

    _main_()
    return


if __name__ == "__main__":
    app.run()
