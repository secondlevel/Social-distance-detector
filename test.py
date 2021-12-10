import cv2
import numpy as np
import math

def show_block1(_width, _height):
    side_num1 = 4                                      #長邊(預設x)切成幾塊
    block_length1 = max(_width, _height)//side_num1 
    side_num2 = min(_width, _height)//block_length1+1  #短邊(預設y)切成幾塊
    block_length2 = min(_width, _height)//side_num2
    if _width < _height:
        hold_num = side_num1
        hold_length = block_length1
        side_num1 = side_num2
        block_length1 = block_length2
        side_num2 = hold_num
        block_length2 = hold_length

    block=[]
    for k in range(side_num2):
        for l in range(side_num1):
            if l == side_num1-1 and k == side_num2-1:
                block.append([l*block_length1, _width, k*block_length2, _height])
            elif l == side_num1-1:
                block.append([l*block_length1, _width, k*block_length2, (k+1)*block_length2])
            elif k == side_num2-1:
                block.append([l*block_length1, (l+1)*block_length1, k*block_length2, _height])
            else:
                block.append([l*block_length1, (l+1)*block_length1, k*block_length2, (k+1)*block_length2]) #append([x左,x右,y上,y下])
    #print('block[x左,x右,y上,y下]: ', block)    #由左至右由上至下
    return side_num1, side_num2, block_length1, block_length2, block

def show_block2(_width, _height):

    side_num1 = 4                                      #長邊(預設x)切成幾塊
    block_length1 = max(_width, _height)//side_num1 
    side_num2 = min(_width, _height)//block_length1+1  #短邊(預設y)切成幾塊
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

def test1(_wid0th, _width, _hei0ght, _height):
    
    spacing = 2
    pixel_width = math.floor(((_width-_wid0th-2)/spacing)+1)
    pixel_height = math.floor(((_height-_hei0ght-2)/spacing)+1)
    
    pixel=[]
    for m in range(pixel_width):
        for n in range(pixel_height):
            pixel.append([(m+0.5)*spacing+_wid0th, (n+0.5)*spacing+_hei0ght])
    #print(np.shape(pixel), len(pixel))  #(3072, 2) 3072
    return pixel

def test2(_wid0th, _width, _hei0ght, _height):
    
    pixel=[]
    spacing = 2
    pixel_width = math.floor(((_width-_wid0th-2)/spacing)+1)
    pixel_height = math.floor(((_height-_hei0ght-2)/spacing)+1)
    
    # (1, 1) ----> first point, (width-1, height-1) ----> last point
    for m in range(pixel_width):
        for n in range(pixel_height):
            pixel.append([1+m*spacing, 1+n*spacing])

    return pixel

def collect_image():
    a = []
    corner_x, corner_y = 7, 7
    gray = cv2.imread("image/1.jpg", 0)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)  #find if the image have chessboard inside
    if ret:
        a.append(corners)
        # a.append(corners)

    return a

def parse_imgpoints1(imgpoints):     #sub_function used by dimension statistic
    # print("before:",np.array(imgpoints).shape)
    pts = np.array(imgpoints).squeeze(axis=None)
    # print("pts.shape:", pts.shape)
    # print(pts.shape)
    # print("after:",pts.shape)
    if len(imgpoints) == 1:
        a = 1
        b, _ = pts.shape
    else:
        a, b, _ = pts.shape
    pts = np.resize(pts,(a*b,2)).tolist()    #resize to the format we want
    
    return pts

def parse_imgpoints2(imgpoints):     #sub_function used by dimension statistic
    
    # print("before:",np.array(imgpoints).shape)
    pts = np.array(imgpoints).squeeze(axis=None)
    # print("after:",pts.shape)

    if len(imgpoints) == 1:
        return pts.tolist()
    else:
        a, b, _ = pts.shape
        return np.resize(pts,(a*b,2)).tolist() #resize to the format we want

if __name__ == "__main__":
    _width, _height = 640, 480
    pixel1 = test1(0, _width, 0, _height)
    pixel2 = test2(0, _width, 0, _height)
    _1, _2, _3 , _4, block1 = show_block1(_width, _height)
    _1, _2, _3 , _4, block2 = show_block2(_width, _height)

    img = collect_image()
    img_preprocess1 = parse_imgpoints1(img)
    img_preprocess2 = parse_imgpoints2(img)

    # print(pixel1)
    # print(pixel2)
    # print(pixel1==pixel2)
    print(np.array(img_preprocess1).shape)
    print(np.array(img_preprocess2).shape)
    # print(pixel)
    # print(block1)
    # print(block2)
    # print(block1==block2)