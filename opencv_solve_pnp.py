import cv2
import numpy as np

# Extended to 6 known 3D coordinates of object points
object_points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]
], dtype=float)

# Corresponding 2D coordinates on the camera image
image_points = np.array([
    [320, 240], [420, 240], [320, 340], [320, 240], [420, 340], [420, 240]
], dtype=float)

# Camera matrix (ensure you replace with your camera's calibration data)
camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
])

# Assuming no lens distortion
dist_coeffs = np.zeros(4)

# Solve for pose using SOLVEPNP_ITERATIVE
success, rotation_vector, translation_vector = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
)

# Check if solvePnP was successful
if success:
    print("Rotation Vector:\n", rotation_vector)
    print("Translation Vector:\n", translation_vector)
else:
    print("solvePnP failed to find a solution.")