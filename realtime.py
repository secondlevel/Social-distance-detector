import cv2
from time import time
import numpy as np
from utils import cal_reproject_error, sliding_window_calibrate, scatter_hist,calculate_the_worst

def capture(frame_count=2, slide_threshold=10):        #frame_counter=> how many frames in total
    counter=0
    corner_x = 7   # pattern is 7*7
    corner_y = 7
    objp = np.zeros((corner_x*corner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)#[0 0 0],[1 0 0],[2 0 0]........[6 6 0]
    _width=0
    _height=0

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(0)

    start_time = time()
    print("a frame will be captured in three seconds")
    while True:         #using infinite loop with timer to do the realtime capture and calibrate
        cur_time = time()
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 若按下 q 鍵則離開迴圈
            break
        if cur_time-start_time > 3:
            if ret: #capture success
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)  #find if the image have chessboard inside
                if ret == True: #chessboard is found in this frame
                    counter += 1
                    print("capture success and chessboard is founded, {}/{}".format(counter,frame_count))
                    objpoints.append(objp)
                    imgpoints.append(corners)  
                    #above part for finding chessboard
                    _width=frame.shape[1]
                    _height=frame.shape[0]
                    img_size = (_width, _height)

                    if counter>slide_threshold:  #choosing when to do the sliding window
                        # ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints, err = sliding_window_calibrate(objpoints, imgpoints, img_size, counter, frame_count)
                        err, imgpoints, objpoints = calculate_the_worst(objpoints, imgpoints, img_size, counter, frame_count, eliminate=False)

                    else:
                        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
                        # err = cal_reproject_error(imgpoints,objpoints,rvecs,tvecs,mtx,dist)
                        err, imgpoints, objpoints = calculate_the_worst(objpoints, imgpoints, img_size, counter, frame_count, eliminate=False)
                        pass

                    print("error:{}".format(err))
                else:
                    print("No chessboard is found in this frame")

                print("相機資料數:",len(imgpoints))
                print("空間資料數:",len(objpoints))
                print('\n')
                if counter == frame_count:  #meet the number of frames defined in the begining
                    cap.release()           #release the camera
                    cv2.destroyAllWindows()
                    break
            start_time=cur_time
            print("a frame will be captured in three seconds")
    scatter_hist(imgpoints, _width, _height)


if __name__ == '__main__':
  capture()
  #cv2.destroyAllWindows()