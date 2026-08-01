import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
import time


start_time = time.time()

def measure_chessboard(img, z):
    real_width = 125
    real_height = 178
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

    # Reshape corners

    corners = np.reshape(corners, [8, 6, 2])
    # print(corners)

    top_left_point = corners[0, 0]
    #print(f"top left point coordinates in pixel are {top_left_point}")
    top_right_point = corners[0, 5]
    #print(f"top right point coordinates in pixel are {top_right_point}")
    down_left_point = corners[7, 0]
    #print(f"down left point coordinates in pixel are {down_left_point}")
    down_right_point = corners[7, 5]
    #print(f"down right point coordinates in pixel are {down_right_point}")

    chessboard_width_1 = np.linalg.norm(np.array(top_right_point) - np.array(top_left_point))
    chessboard_width_2 = np.linalg.norm(np.array(down_right_point) - np.array(down_left_point))
    chessboard_width = (chessboard_width_2+chessboard_width_1)/2

    #print(f"chessboard width in pixel is {chessboard_width}")


    chessboard_height_1 = np.linalg.norm(np.array(down_right_point) - np.array(top_right_point))
    chessboard_height_2 = np.linalg.norm(np.array(down_left_point) - np.array(top_left_point))
    chessboard_height = (chessboard_height_2+chessboard_height_1)/2
    # print(f"chessboard height in pixel is {chessboard_height}")

    width_mm = ((z * 1000) * chessboard_width) / 567.2
    height_mm = ((z * 1000) * chessboard_height) / 567.2

    error_width =abs ( ( real_width- width_mm)/real_width)*100
    error_height =abs ( ( real_height- height_mm)/real_height)*100

    error_width_list.append(error_width)
    error_height_list.append(error_height)



    width_list.append(width_mm)
    height_list.append((height_mm))

    print(width_mm, height_mm)

width_list = []
height_list = []

error_width_list = []
error_height_list = []

frame_counter = 0
z_meters_list = []

# Inizializza una coda per il filtro di media mobile
buffer_size = 5
z_meters_buffer = deque(maxlen=buffer_size)

cap_L = cv2.VideoCapture('robot-navigation-video/robotL.avi')
cap_R = cv2.VideoCapture('robot-navigation-video/robotR.avi')


window_size = 12
min_disp = 0
nDispFactor = 8

num_disp = 16 * nDispFactor - min_disp


stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=16*3 * window_size ** 2, #anche 16 buono da provare al posto di 32

    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY

)

while cap_L.isOpened() and cap_R.isOpened():

    frame_counter += 1

    roi_width = 60
    roi_height = 60

    ret_L, frame_L = cap_L.read()
    ret_R, frame_R = cap_R.read()
    if not ret_L and not ret_R:
        break
    frame_width = frame_R.shape[1]
    # print(frame_width)
    frame_height = frame_R.shape[0]
    # print(frame_height)

    roi_x = (frame_width - roi_width) // 2
    roi_y = (frame_height - roi_height) // 2

    # cv2.imshow(' Frame Destro', frame_R)
    # cv2.imshow(' Frame Sinistro', frame_L)

    if ret_L and frame_L is not None:
        frame_L_gray = cv2.cvtColor(frame_L, cv2.COLOR_BGR2GRAY)
        frame_L_gray_roi = frame_L_gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    if ret_R and frame_R is not None:
        frame_R_gray = cv2.cvtColor(frame_R, cv2.COLOR_BGR2GRAY)
        frame_R_gray_roi = frame_R_gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # cv2.imshow(' Frame Destro', frame_R_gray_roi)
        # cv2.imshow(' Frame Sinistro', frame_L_gray_roi)

    if not ret_L or frame_L is None:
        # Release the Video if ret is false
        cap_L.release()
        print("Released Video Resource")
        break

    disparity = stereo.compute(frame_L_gray, frame_R_gray)/16

    #disparity = disparity.astype(np.uint8)
    #disparity = cv2.medianBlur(disparity, ksize=11)

    disparity_roi = disparity[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    p15 = np.percentile(disparity_roi, 15)

    mask = (disparity_roi >= p15)
    disparity_roi = disparity_roi[mask]



    dmain = np.median(disparity_roi)
    #dmain = np.mean(disparity_roi)


    # Normalize the disparity map
    normalized_disparity = cv2.normalize(disparity,disparity, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    z = (92.226 * 567.2) / dmain
    z_meters = z / 1000



    z_meters_buffer.append(z_meters)


    z_meters_avg = np.round(sum(z_meters_buffer) / len(z_meters_buffer), 2)
    # plot
    z_meters_list.append(z_meters_avg)



    if z_meters_avg < 0.8:
        cv2.putText(frame_L, "ALERT", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if frame_counter % 3 == 0:
        measure_chessboard(frame_L_gray, z_meters)

    cv2.rectangle(frame_R, (roi_x, roi_y), (roi_x + roi_width, roi_height + roi_y), (0, 255, 0), 2)
    cv2.rectangle(frame_L, (roi_x, roi_y), (roi_x + roi_width, roi_height + roi_y), (0, 255, 0), 2)
    cv2.putText(frame_L, f'{z_meters_avg}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(' Frame Sinistro', frame_L)
    # cv2.imshow(' Frame Destro', frame_R)
    cv2.imshow('Disparity map', normalized_disparity)

    cv2.waitKey(1)
    # cv2.destroyAllWindows()


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time is  {elapsed_time} seconds.")


plt.figure(figsize=(10, 6))
plt.plot(z_meters_list)
plt.title('Distance')
plt.xlabel('Frame')
plt.ylabel('z_meters')


plt.figure(figsize=(10, 6))
plt.plot(error_height_list)
plt.title('Measure error')
plt.xlabel('Frame')
plt.ylabel('Percentage error')
plt.show()



print(np.median(height_list))
print(np.median(width_list))

