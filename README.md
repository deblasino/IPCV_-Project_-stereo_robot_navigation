# IPCV_-Project_-stereo_robot_navigation
Image Processing and Compyter Vision's final project. 

Two synchronized video streams from a stereo rig mounted on a moving robot go in. Out comes a live estimate of how far away the obstacle in front is, an on-screen alarm when it gets closer than 0.8 m, and a measurement of a real object's physical size in millimetres from images alone, no depth sensor.

Disparity → distance. For each synchronized frame pair, a dense disparity map is computed with semi-global block matching. Inside a 60×60 px ROI at the centre of the frame (the patch of world the robot would drive into if it kept going straight) a single representative disparity d_main is extracted, and depth follows from the standard stereo relation:

z [mm] = baseline [mm] × focal [px] / d_main [px]

d_main is the median of the ROI disparity, not the mean. That choice matters: outliers from the low-texture black squares of the chessboard would otherwise drag the estimate around every frame.

Measuring a real object. The chessboard corners are detected and refined to sub-pixel accuracy, then its pixel width and height are converted to millimetres using the estimated z. This is the part that gives a hard number to check the whole pipeline against — if the depth estimate is off, the measured size is off with it.

The optional extensions (cv_project_optional.py).

Adaptive disparity range: Instead of a fixed search range, min_disp and num_disp are recentred each frame on the previous frame's d_main, keeping a 64 px window around it. Narrower search = less noise and less compute, and it tracks the disparity growth as the robot closes in on the obstacle.

ROI split into 5 vertical stripes: Five independent distance estimates instead of one. The proximity check runs per stripe, and the spread across stripes reveals the inclination of the camera relative to the obstacle, a flat wall seen head-on gives five equal distances, an angled one gives a gradient. Rendered live as a small graphical readout.

Noise handling: The lowest 15% of disparity values per stripe are discarded before taking the median, and the final distance passes through a moving-average filter to stop the readout jittering frame to frame.
