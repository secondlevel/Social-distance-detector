import cv2
import numpy as np
import math
from math import ceil
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from shapely import geometry

def parse_imgpoints(imgpoints):     #sub_function used by dimension statistic
    
    # print("before:",np.array(imgpoints).shape)
    pts = np.array(imgpoints).squeeze(axis=None)
    # print("after:",pts.shape)

    if len(imgpoints) == 1:
        return pts.tolist()
    else:
        a, b, _ = pts.shape
        return np.resize(pts,(a*b,2)).tolist() #resize to the format we want

"""
#use parameters to calculate reprojection error
def cal_reproject_error(imgpoints, objpoints, rvecs, tvecs, mtx, dist):
    sum_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        sum_error += error
    #print( "reprojection error: {}".format(sum_error/len(objpoints)))
    reprojection_error = sum_error/len(objpoints)
    return reprojection_error
"""

#use frames to calculate every parameters
def calculate_parameters(objpoints, imgpoints, img_size, cur_count, total, eliminate=False):
    #reprojection_error = None
    packed_tmp = ret_tmp, mtx_tmp, dist_tmp, rvecs_tmp, tvecs_tmp = cv2.calibrateCamera(objpoints[:], imgpoints[:], 
                                                                                        img_size, None, None)
    all_error_tmp=[]
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_tmp[i], tvecs_tmp[i], mtx_tmp, dist_tmp)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        all_error_tmp.append(error) # all_error: 由每張frame各自的error所組成的array

    mean_error_tmp = sum(all_error_tmp)/len(all_error_tmp)

    return imgpoints[:], objpoints[:], packed_tmp, ret_tmp, mtx_tmp, dist_tmp, rvecs_tmp, tvecs_tmp, all_error_tmp, mean_error_tmp

def _pixel(_wid0th, _width, _hei0ght, _height):

    pixel=[]
    spacing = 2
    pixel_width = math.floor(((_width-_wid0th-2)/spacing)+1)
    pixel_height = math.floor(((_height-_hei0ght-2)/spacing)+1)
    
    # (1, 1) ----> first point, (width-1, height-1) ----> last point
    for m in range(pixel_width):
        for n in range(pixel_height):
            pixel.append([1+m*spacing, 1+n*spacing])
            
    return pixel

def check_pixel(pixel, x_left, x_right, y_top, y_bot):
    counter = 0
    for i in range(len(pixel)):
        if (pixel[i][0] > x_left) and (pixel[i][0] < x_right) and (pixel[i][1] > y_top) and (pixel[i][1] <y_bot):
            counter += 1
    return counter

def pick_corner_find_uncovered_pixel(p_imgpoints, counter, t, pixel):
    
    all_corner = []
    save_discard = []
    for i in range(counter - t):
        all_corner.append( [p_imgpoints[0+49*i], p_imgpoints[6+49*i], p_imgpoints[48+49*i], p_imgpoints[42+49*i]] )
    poly = all_corner[-1]
    line = geometry.LineString(poly)
    polygon = geometry.Polygon(line)
    for k in range(len(pixel)):
        point= geometry.Point(pixel[k])
        if polygon.contains(point) == True:
            save_discard.append(pixel[k])
            pixel[k] = -1
    new_pixel = []
    for n in range(len(pixel)):
        if pixel[n] != -1:
            new_pixel.append(pixel[n])
            
    return new_pixel, save_discard      #這裡的new_pixel為還沒被任何一張覆蓋的pixel, save_discard為這張照片所覆蓋的pixel

def show_block(_width, _height):

    side_num1 = 4                                      #長邊(預設x)切成幾塊
    block_length1 = max(_width, _height)//side_num1 
    side_num2 = (min(_width, _height)//block_length1)+1  #短邊(預設y)切成幾塊
    block_length2 = min(_width, _height)//side_num2

    if _width < _height:
        side_num1, side_num2 = side_num2, side_num1
        block_length1, block_length2 = block_length2, block_length1  

    block=[]
    for k in range(side_num2):
        for l in range(side_num1):
            block.append([l*block_length1, (l+1)*block_length1, k*block_length2, (k+1)*block_length2]) #append([x左,x右,y上,y下])
    
    #print('block[x左,x右,y上,y下]: ', block)    #由左至右由上至下
    return side_num1, side_num2, block_length1, block_length2, block

def scatter_hist(imgpoints, _width, _height, inverse=True): #draw the scatter graph using imgpoints
    pt = parse_imgpoints(imgpoints)
    x = []
    y = []
    values = []
    for item in pt:
        _x, _y = item
        x.append(_x)
        y.append(_y)

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom+height+spacing, width, 0.2]
    rect_histy = [left+width+spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter,xlim=(0, width),ylim=(0, height))     #axe is a sub-area in figure.
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.set_xlim(0, _width)
    ax.set_ylim(0, _height)
    if inverse:     #due to the different coordinate between images and 2-D coordinate(mainly on Y axis), we have to inverse y axis to get the correct result
        ax.invert_yaxis()
    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth)+1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    plt.show()

def draw_block(frame, block, block_cover):
    for i in range(len(block)):
        if block_cover[i] < 0.3:
            cv2.rectangle(frame, (block[i][0], block[i][3]), (block[i][1], block[i][2]), color = (0, 0, 255), thickness = 1)
        else:
            cv2.rectangle(frame, (block[i][0], block[i][3]), (block[i][1], block[i][2]), color = (0, 255, 0), thickness = 2)