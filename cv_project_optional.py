import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

# Lists to track all the widths and heights of the chessboard detected in each frame
width_list = []
height_list = []

# Number of stripes in which the ROI is divided (optional point 2)
num_stripes = 5


plt.ion()

def measure_chessboard(img, z):
    pattern_size = (6, 8)
    found, corners = cv2.findChessboardCorners(img, pattern_size)

    if found:
        # Refining corner position
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
        cv2.cornerSubPix(img, corners, (5, 5), (0, 0), term)

        # Visualize detected corners
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, None, fx=1, fy=1)
        cv2.drawChessboardCorners(vis, pattern_size, corners * 1, found)
        cv2.imshow("Frame L with corners", vis)

    if not found:
        print('chessboard not found')
        return None

    # Reshape corners to a 8x6 matrix

    corners = np.reshape(corners, [8, 6, 2])
    # print(corners)

    top_left_point = corners[0, 0]
    # print(f"top left point coordinates in pixel are {top_left_point}")
    top_right_point = corners[0, 5]
    # print(f"top right point coordinates in pixel are {top_right_point}")
    down_left_point = corners[7, 0]
    down_right_point = corners[7, 5]

    chessboard_width_1 = np.linalg.norm(np.array(top_right_point) - np.array(top_left_point))
    chessboard_width_2 = np.linalg.norm(np.array(down_right_point) - np.array(down_left_point))
    chessboard_width = (chessboard_width_2 + chessboard_width_1) / 2
    #print(f"chessboard width in pixel is {chessboard_width}")

    chessboard_height_1 = np.linalg.norm(np.array(down_right_point) - np.array(top_right_point))
    chessboard_height_2 = np.linalg.norm(np.array(down_left_point) - np.array(top_left_point))
    chessboard_height = (chessboard_height_2 + chessboard_height_1) / 2
    # print(f"chessboard height in pixel is {chessboard_height}")

    width_mm = ((z * 1000) * chessboard_width) / 567.2
    height_mm = ((z * 1000) * chessboard_height) / 567.2

    width_list.append(width_mm)
    height_list.append((height_mm))

    print(width_mm, height_mm)


frame_counter = 0
# Keep track of the distance values to plot it at the end of the code
z_meters_list = []

# Initialize a buffer for the moving average filter
buffer_size = 5
z_meters_buffer = deque(maxlen=buffer_size)




# Acquire the videos
cap_L = cv2.VideoCapture('robot-navigation-video/robotL.avi')
cap_R = cv2.VideoCapture('robot-navigation-video/robotR.avi')

# Declare the offset (from optional point 1)
offset = 64
dmain_prev = offset/2


window_size = 12
min_disp = 0
nDispFactor = 8







while cap_L.isOpened() and cap_R.isOpened():
    # Declare num_disp and min_disp in order to have dmain lying in the center of the interval (optional point 1)
    num_disp = int(dmain_prev + (offset/2))
    min_disp = int(dmain_prev - (offset/2))

    # Create the object StereoSGBM with the new parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        # P1 and P2 are parameters which control the smoothness of the disparity map
        P1=8 * 3 * window_size ** 2,
        P2=16 * 3 * window_size ** 2,
        # To speed up the computation
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


    frame_counter += 1

    # Selected ROI dimensions to measure the disparity
    roi_width = 60
    roi_height = 60

    stripe_width = roi_width // num_stripes

    # Initialize an array to store the disparities for each stripe

    #disparities = np.zeros(num_stripes)
    disparities = []
    # Initialize an array to store the distances for each stripe
    distances = []
    #distances = np.zeros(num_stripes)
    # Initialize an array to store the mean disparities for each stripe
    mean_disparities = []
    #mean_disparities = np.zeros(num_stripes)



    ret_L, frame_L = cap_L.read()
    ret_R, frame_R = cap_R.read()
    if not ret_L and not ret_R:
        break
    frame_width = frame_R.shape[1]
    frame_height = frame_R.shape[0]


    roi_x = (frame_width - roi_width) // 2
    roi_y = (frame_height - roi_height) // 2


    if ret_L and frame_L is not None:
        frame_L_gray = cv2.cvtColor(frame_L, cv2.COLOR_BGR2GRAY)
        frame_L_gray_roi = frame_L_gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    if ret_R and frame_R is not None:
        frame_R_gray = cv2.cvtColor(frame_R, cv2.COLOR_BGR2GRAY)
        frame_R_gray_roi = frame_R_gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]


    if not ret_L or frame_L is None:
        # Release the video if ret is false
        cap_L.release()
        print("Released Video Resource")
        break
    # Disparity from stereoSGBM is divided by 16 because the algorithm multiplies the disparity by this value in order to be more accurate
    tot_disparity = stereo.compute(frame_L_gray, frame_R_gray) / 16



    # Manipulation to compute the disparity for each stripe
    roi_x_2 = roi_x
    # List to be sent to ChessBoard Measure function
    z_meters = []
    # List to be displayed on video
    z_meters_to_display = []

    # Measure the disparity for each stripe using a for cycle
    for i in range(num_stripes):

        # Take only the disparity on a singular stripe
        disparity = tot_disparity[roi_y:roi_y + roi_height, roi_x_2:roi_x_2 + roi_width//num_stripes]

        # As there is some noise in the measure of disparity (black zones on the chessboard), the lower values of disparity are removed (15%)
        p15 = np.percentile(disparity, 15)
        # Create a boolean mask
        mask = (disparity >= p15)
        disparity = disparity[mask]

        #disparities = np.append(disparities, disparity)
        disparities.append(disparity)

        # Change the value to consider the next stripe in the next iteration
        roi_x_2 = roi_x_2 + round(roi_width/num_stripes)

        # Measure the median of the selected values
        mean = np.median(disparity)
        # Update dmain_prev to adapt the window size (optional point 1)
        dmain_prev = mean

        #mean_disparities = np.append(mean_disparities, mean)
        mean_disparities.append(mean)

        # Measure the distance in mm
        distance = (92.226 * 567.2) / mean

        #distances = np.append(distances, distance)
        distances.append(distance)

        # Measure the distance in m
        z = distance / 1000

        ######################z_meters = np.append(z_meters, z)
        z_meters.append(z)

        # Round the distance to show it on video
        z_rounded_to_display = np.round(distance/1000,2)
        z_meters_to_display = np.append(z_meters_to_display, z_rounded_to_display)

    # Normalize the disparity map from 0 to 255
    normalized_disparity = cv2.normalize(tot_disparity, tot_disparity, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)



    # Alert when one of the distances is less than 0.8m
    for i in z_meters:
        if i < 0.8:
            cv2.putText(frame_L, "ALERT", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # Mean for all the distances from every frame
    z_mean = np.mean(z_meters)
    # Add z_mean to the buffer to compute the mobile mean
    z_meters_buffer.append(z_mean)

    # Measure the mean of z_meters in the buffer
    z_meters_avg = np.round(sum(z_meters_buffer) / len(z_meters_buffer), 2)
    # Add z_meters_list to the buffer to plot z mean values
    z_meters_list.append(z_meters_avg)

    # Consider only one third of the frames to speed-up the computation
    if frame_counter % 3 == 0:

        position_x = np.arange(num_stripes)
        plt.bar(position_x, z_meters, align='center', alpha=0.5)
        plt.xticks(position_x, ['1', '2', '3', '4', '5'])
        plt.xlabel("Stripes")
        plt.ylabel("Distance [m]")
        plt.ylim(0, 3)
        plt.draw()
        plt.pause(0.1)
        plt.clf()

        measure_chessboard(frame_L_gray, z_mean)

    # Draw the ROI rectangle on the video
    cv2.rectangle(frame_L, (roi_x, roi_y), (roi_x + roi_width, roi_height + roi_y), (0, 255, 0), 2)
    cv2.putText(frame_L, f'{z_meters_to_display}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the L frame
    cv2.imshow('Left frame', frame_L)
    # Show the disparity map
    cv2.imshow('Disparity map', normalized_disparity)

    cv2.waitKey(1)


plt.ioff()
# Plot the mean distance
plt.figure(figsize=(10, 6))
plt.plot(z_meters_list)
plt.title('z_meters nel tempo')
plt.xlabel('Frame')
plt.ylabel('z_meters')
plt.show()


# Print an estimation of the width and the height of the chessboard
print(np.median(height_list))
print(np.median(width_list))


