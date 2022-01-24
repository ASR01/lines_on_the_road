# This is the Streamlit version of the app.

from multiprocessing import Barrier
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_processing as ip

st.set_page_config(layout="wide")
st.header('CV for Driving')

st.write('load you image')
opencv_image = None
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

else:
   	img = cv2.imread('./test_images/solidWhiteCurve.jpg')


st.sidebar.text('Color Filters')
with st.sidebar.expander("Low Filter Change"):
	st.text('Low filter')
	h0 = st.slider ('H', min_value = 0, max_value = 180, value = 0)
	s0 = st.slider ('S', min_value = 0, max_value = 255, value = 200)
	v0 = st.slider ('V', min_value = 0, max_value = 255, value = 0)

with st.sidebar.expander("High Filter Change"):
		
	st.text('High filter')
	h1 = st.slider ('H', min_value = 0, max_value = 180, value = 180)
	s1 = st.slider ('S', min_value = 0, max_value = 255, value = 255)
	v1 = st.slider ('V', min_value = 0, max_value = 255, value = 255)

st.sidebar.text('Blurring')
with st.sidebar.expander("Kernel Parameters"):
	st.text('Kernel Size')
	blur_kernel = st.slider ('Kernel', min_value = 1, max_value = 29, value = 15, step = 2)

st.sidebar.text('Edging')
with st.sidebar.expander("Edging Filters"):
	edging_low = st.slider ('Low', min_value = 0, max_value = 255, value = 50)
	edging_high= st.slider ('High', min_value = 0, max_value = 255, value = 150)
 
st.sidebar.text('ROI')
with st.sidebar.expander('Region of Interest'):
	st.text('Percentage related to the total heigh, width')
	upper_border = st.slider('Upper Border', min_value = 0, max_value = 100, value = 45)
	bottom_border = st.slider('Lower Border', min_value = 0, max_value = 100, value = 0)
	left_border = st.slider('Left Border', min_value = 0, max_value = 100, value = 35)
	right_border = st.slider('Right Border', min_value = 0, max_value = 100, value = 35) 

st.sidebar.text('Minimum Angle')

with st.sidebar.expander('Minimum angle'):

	st.text('Percentage related to the total heigh, width')
	angle = st.slider('Angle', min_value = 0, max_value = 180,  value = 20)


img_show = img[:,:,::-1]

col1, col2, col3 = st.columns(3)

with col1:
	st.text('Original')
	st.image(img_show)

with col2:
	st.text('Color Filtered')

	filtered = ip.filter_colours(img_show, (h0, s0, v0), (h1, s1, v1))
	st.image(filtered)
	

with col3:
	st.text('Blurred')
	
	blurred = ip.blurring(filtered, blur_kernel)
	st.image(blurred)
	
col11, col12, col13 = st.columns(3)

with col11:
	st.text('Edged')
	edged = ip.edging(blurred, edging_low, edging_high)
	st.image(edged)

with col12:
	st.text('ROI')
	row = img.shape[0]
	col = img.shape[1]
	up_left = [col*left_border/100, row * upper_border/100]
	up_right = [col *(1-right_border/100), row * upper_border/100]
	bottom_left = [0, row * (1 - bottom_border/100)]
	bottom_right = [col, row * (1 - bottom_border/100)]
	corners = np.array([up_left, up_right, bottom_right, bottom_left], dtype = 'int32')

	write_ROI = ip.write_ROI(img, corners)
	st.image(write_ROI)


with col13:
	st.text('Detection ROI')
	detection_ROI = ip.define_ROI(edged, corners)
	st.image(detection_ROI)


 
#	st.image(detection_ROI)


col21, col22 = st.columns(2)

with col21:
	st.text('Lines Detected')

	lines_detected, rl, ll = ip.detect_lines(detection_ROI, angle*np.pi/180)
	st.image(lines_detected)

with col22:
	st.text('Lines Refined')
	right = ip.unifying_lines(rl)
	left = 	ip.unifying_lines(ll)
	lines_refined = ip.write_lines(img_show, left, right)
	st.image(lines_refined)
