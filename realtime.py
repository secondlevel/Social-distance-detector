import cv2
from time import time
import numpy as np
import os
from utils import parse_imgpoints, calculate_parameters, scatter_hist, _pixel, \
    pick_corner_find_uncovered_pixel, show_block, check_pixel, draw_block
import pickle

def capture(frame_count=20):        #frame_counter=> how many frames in total
    
    #type-in file name
    file_name = 't'  
    os.makedirs('./{}'.format(file_name))
    
    print("file name: ", file_name)
    counter=0
    corner_x = 7   # pattern is 7*7
    corner_y = 7
    objp = np.zeros((corner_x*corner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)#[0 0 0],[1 0 0],[2 0 0]........[6 6 0]
    setting = False
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    block = []
    del_history = []
    pic_del = 0

    cap = cv2.VideoCapture(0)
    start_time = time()
    #print("a frame will be captured in three seconds")
    while True:         #using infinite loop with timer to do the realtime capture and calibrate
        cur_time = time()
        ret, frame = cap.read()
        if setting == True and cur_time-start_time < 2.9:
            text = "{}".format(int(round( 3-(cur_time-start_time),0)))
            cv2.putText(frame, text, (285, 270), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2, cv2.LINE_AA)
            # print("block_coverage:",block_coverage)
            draw_block(frame, block, block_coverage)
        if setting == False:
            cv2.putText(frame, "setting", (260, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 若按下 q 鍵則離開迴圈
            break
        if cur_time-start_time > 3:
            if ret: #capture success
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)  #find if the image have chessboard inside
                if ret == True: #chessboard is found in this frame
                    if setting == False:
                        print("! setting !")
                        _width = frame.shape[1]
                        _height = frame.shape[0]
                        img_size = (_width, _height)
                        side_num1, side_num2, block_length1, block_length2, block = show_block(_width, _height)
                        print("(width, height) = ({}, {})".format(_width, _height))
                        block_num = len(block)
                        print("side length of block = {} x {}".format(block_length1, block_length2))
                        print("number of block = {} x {} = {}".format(side_num1, side_num2, block_num))
                        block_coverage = [0]*block_num   
                        pixel = _pixel(0, _width, 0, _height)
                        init_pixel_number = len(pixel) 
                        print("number of pixel = {}".format(init_pixel_number))  # pixel_width*pixel_height=len(pixel)
                        initial_pixel = []
                        for i in range(len(block)):
                            count_pixel = check_pixel(pixel, block[i][0], block[i][1], block[i][2], block[i][3])
                            print("block {} : {}    initial pixel number : {}".format(i, block[i], count_pixel))
                            initial_pixel.append(count_pixel)
                        setting = True
                        start_time=cur_time
                        continue
                    counter += 1
                    print("capture success and chessboard is founded, {}/{}".format(counter,frame_count))
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    #cv2.imwrite('./{}/output{}.jpg'.format(file_name,counter), gray)
                    #above part for finding chessboard, append points, save picture

                    imgpoints, objpoints, packed_tmp, ret_tmp, mtx_tmp, dist_tmp, rvecs_tmp, tvecs_tmp, all_error_tmp, mean_error_tmp \
                        = calculate_parameters(objpoints, imgpoints, img_size, counter-pic_del, frame_count, eliminate=False)

                    p_imgpoints = parse_imgpoints(imgpoints)    #resize from (n, 49, 1, 2) <class 'list'> to (49n, 2) <class 'list'>
                    uncovered_pixel, discard = pick_corner_find_uncovered_pixel(p_imgpoints, counter, pic_del, pixel) 
                    del_history.append(discard)

                    print("error for each frame:{}".format(all_error_tmp))
                    error_avg = np.average(all_error_tmp)
                    error_std = np.std(all_error_tmp)
                    print("average:", error_avg)     
                    print("standard:", error_std)
                    pixel = uncovered_pixel

                    if counter == 10:
                        print("check")
                        for i in range(counter-1, -1, -1):
                            if all_error_tmp[i] >= error_avg + 2*error_std:
                                imgpoints.pop(i)
                                objpoints.pop(i)
                                all_error_tmp.pop(i)
                                for j in range(len(del_history[i])):
                                    pixel.append(del_history[i][j])
                                del_history.pop(i)
                                print("delete")
                                pic_del += 1
                        print("error for each frame (deleted):{}".format(all_error_tmp))
                    if counter >10:
                        if all_error_tmp[-1] >= error_avg + 2*error_std:
                            imgpoints.pop(-1)
                            objpoints.pop(-1)
                            all_error_tmp.pop(-1)
                            for j in range(len(del_history[-1])):
                                pixel.append(del_history[-1][j])
                            del_history.pop(-1)
                            print("delete")
                            pic_del += 1
                            print("error for each frame (deleted):{}".format(all_error_tmp))
                    print("picture deleted: ", pic_del)

                    for i in range(len(block)):
                        count_pixel = check_pixel(pixel, block[i][0], block[i][1], block[i][2], block[i][3])
                        block_coverage[i] = round((initial_pixel[i]-count_pixel)/initial_pixel[i], 3)
                        #print("block {} : {}     coverage : {}/{} = {}".format(i, block[i], initial_pixel[i]-count_pixel, initial_pixel[i], block_coverage[i]))
                    pixel_num = len(pixel)
                    coverage_tmp = (init_pixel_number - pixel_num)/init_pixel_number

                    qualify = 0
                    for i in range(len(block_coverage)):
                        if block_coverage[i] > 0.3:
                            qualify += 1
                    print("block>0.3:", qualify, "/", len(block)) 
                    if counter >= 10:
                        if qualify == len(block_coverage):
                            print("\n\n end \n\n")
                            cap.release()       #release the camera
                            cv2.destroyAllWindows()
                            break
                    if counter == frame_count:  #meet the number of frames defined in the begining
                        print("\n\n end \n\n")
                        cap.release()           #release the camera
                        cv2.destroyAllWindows()
                        break
                else:
                    print("No chessboard is found in this frame")
            start_time=cur_time
            #print("\na frame will be captured in three seconds\n")

    scatter_hist(imgpoints, _width, _height)
    fixed_param = {'img_points':imgpoints, 'ret':ret_tmp, 'mtx':mtx_tmp, 'dist':dist_tmp, 'rvecs':rvecs_tmp, 'tvecs':tvecs_tmp, \
        'error':all_error_tmp, 'mean_error':mean_error_tmp, 'block_coverage':block_coverage, 'coverage':coverage_tmp}
    #print("fixed_param:\n",fixed_param)

    file = open('./{}/param.pickle'.format(file_name),'wb')
    pickle.dump(fixed_param,file)   #save parameters
    file.close()


if __name__ == '__main__':
  capture()
  #cv2.destroyAllWindows()