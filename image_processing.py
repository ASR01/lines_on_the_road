import cv2
import numpy as np
import matplotlib.pyplot as plt




def filter_colours(img, low_filter = (0,200,0), high_filter = (120,255,120)):
    # This filter will filter out the ccolor oiut of white.

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #low_filter = (0,200,0)
    #high_filter = (180, 255, 255)

    colour_mask = cv2.inRange(img_hsv, low_filter, high_filter)
    colour_mask = cv2.bitwise_not(colour_mask)
    colour_mask = np.stack((colour_mask,) * 3, axis=-1)
    
#    img_hsv[colour_mask != 0] = [0,0,0]
    img_bgr = cv2.bitwise_and(img, colour_mask)
    #img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    
    return img_bgr
    return colour_mask
    
def define_ROI(img, corners):

    mask = np.zeros_like(img)
    white = (255,255,255)

    cv2.fillPoly(mask, pts = [corners], color = white)
    
    
    img = cv2.bitwise_and(img, mask)
    
    return img

def blurring(img, kernel):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blurred

def edging(img, low, high):
    edged = cv2.Canny(img, low, high, None, 3)
    return edged

def detect_lines(img, angle):
    limit = np.tan(angle)
    img_c = img.copy()
    lines = cv2.HoughLinesP(img_c, 0.75, np.pi / 180 , threshold=10, minLineLength=10, maxLineGap=30)
    img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(len(lines)):
            x0, y0, x1, y1 = lines[i][0]
            if x1 != x0:
                tangent = ((y1-y0)/(x1-x0))
            if abs(tangent) > limit:
                cv2.line(img_c, (x0, y0), (x1, y1), (255,0,0), 3, cv2.LINE_AA)
    rl, ll =checking_lines(lines, angle)
    
    return img_c, rl, ll

def checking_lines(lines, angle):
    #print(len(lines))
    left_lines = []
    right_lines = []
    limit = np.tan(angle)
            
    for i in range(len(lines)):
        x0, y0, x1, y1 = lines[i][0]
        if x1 != x0:
            tangent = ((y1-y0)/(x1-x0))
        if abs(tangent) > limit:
            if tangent > 0:
                left_lines.append([x0, y0, x1, y1])
            elif tangent < 0:
                right_lines.append([x0, y0, x1, y1])
        #print(lines[i][0], tangent, limit)
    #print('right', right_lines)
    #print('left',left_lines)
     
            
    return right_lines, left_lines

def unifying_lines(lin):
       
    tan = []
    con = []
    long  = [] 
    
    for i in range(len(lin)):
        #print(i, lin[i])
        x0, y0, x1, y1  = lin[i]
        if x1 !=x0:
            tangent = (y1-y0)/(x1-x0)
            b = y1 - tangent*x1
            length = np.sqrt((y1-y0)**2+(x1-x0)**2)
            if length > 20:
                tan.append(tangent)
                con.append(b)
                long.append(length)
                
    
    # add more weight to longer lines
    if len(tan) > 0:
        final_tangent= sum(np.array(long) * np.array(tan)) /np.sum(long) 
        final_b = sum(np.array(long) * np.array(con)) /np.sum(long)
    else:
        final_tangent = None
        lane = None
        final_b = None
    return final_tangent, final_b

def write_lines(img, left, right):

	height = img.shape[0]
	blank = np.zeros_like(img)
    
    #left points
	if left != None:
		left_y0 = int(height)
		left_y1 = int(round((height * 0.7),0))
		left_x0 = round((left_y0 - left[1])/left[0])
		left_x1 = round((left_y1 - left[1])/left[0])
		cv2.line(blank, (left_x0,left_y0), (left_x1,left_y1),  (255,0,0), 15)

    #right points
	if right != None:	
		right_y0 = int(height)
		right_y1 = int(round((height * 0.7),0))
		right_x0 = round(int(right_y0 - right[1])/right[0])
		right_x1 = round(int(right_y1 - right[1])/right[0])
		cv2.line(blank, (right_x0,right_y0), (right_x1,right_y1),  (255,0,0), 15)

    
       
    
	return cv2.addWeighted(img, 0.9, blank, 0.95, 0.0)
    
def process_img(img, type = 0):
    
    #type = 
    img_show = img[:,:,::-1]

    # Cropping Params
    row = img.shape[0]
    col = img.shape[1]
    angle = 20*(np.pi/180) #(Tangent in degrees)


    up_left = [col* 0.35, row * 0.50]
    up_right = [col* 0.65, row * 0.5]
    bottom_left = [col * 0, row * 1]
    bottom_right = [col* 1,row * 1]
    corners = np.array([up_left, up_right, bottom_right, bottom_left], dtype = 'int32')
    # Blurring Pars
    blur_kernel = 15
    sigma = 0
    #Edging pars
    low = 50
    high = 150

    #Img_processing
    filtered = filter_colours(img)
    blurred = blurring(filtered, blur_kernel)
    edged = edging(blurred, low, high)
    detection_ROI = define_ROI(edged, corners)

    # detect the lines, returns the image and the left and right oriented lines
    lines_detected, rl, ll = detect_lines(detection_ROI, angle)
    #group the lines
    #print(ll, rl)
    right = unifying_lines(rl)
    left = unifying_lines(ll)
    #print(left, right)
    lines_refined = write_lines(img_show, left, right)






    # display for test
    if type == 1:
        fig = plt.figure(figsize = (15,10))
        fig.add_subplot(3,3,1)
        plt.title('Original')
        plt.imshow(img_show)
        fig.add_subplot(3,3,2)
        plt.title('Color Filter')
        plt.imshow(filtered)
        fig.add_subplot(3,3,3)
        plt.title('Blurred')
        plt.imshow(blurred, cmap = 'gray')
        fig.add_subplot(3,3,4)
        plt.title('Edge')
        plt.imshow(edged, cmap = 'gray')
        fig.add_subplot(3,3,5)
        plt.title('Region of Interest')
        plt.imshow(detection_ROI, cmap = 'gray')
        fig.add_subplot(3,3,6)
        plt.title('Lines')
        plt.imshow(lines_detected)
        fig.add_subplot(3,3,7)
        plt.title('Lines refined')
        plt.imshow(lines_refined, cmap = 'gray')
        plt.show()

    return lines_refined 


def write_ROI(img, corners):
	#img_show = img[:,:,::-1]
    img_show = img
 
    up_left, up_right, bottom_right, bottom_left = corners

    
	# print(bottom_left, bottom_right, up_left, up_right)
	# print((up_left[0],up_left[1]), (up_right[0], up_right[1]),  (255,0,0), 15)
    blank = np.zeros_like(img_show)
    #print(left_x0,left_y0,left_x1,left_y1, '\n')
    #print(right_x0,right_y0,right_x1,right_y1, '\n')
       
    cv2.line(blank, (up_left[0],up_left[1]), (up_right[0], up_right[1]),  (0,255,0), 5)
    cv2.line(blank, (up_left[0],up_left[1]), (bottom_left[0], bottom_left[1]),  (0,255,0), 5)
    cv2.line(blank, (up_right[0],up_right[1]), (bottom_right[0], bottom_right[1]),  (0,255,0), 5)
    cv2.line(blank, (bottom_left[0],bottom_left[1]), (bottom_right[0], bottom_right[1]),  (0,255,0), 5)
    
    return cv2.addWeighted(img_show, 0.9, blank, 0.95, 0.0)




def main():
    pass    
    #test img
    # img = cv2.imread('./test_images/solidWhiteCurve.jpg')
    # print(img.shape)
    

    # process_img(img, 1)


if __name__ == '__main__':
    main()
