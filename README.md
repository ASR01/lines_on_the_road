# Lines On The Road

The purpose of this project is to create a module for automated driving.

The first step of automatic driving is to detect the lines where the car is running and keep it this way.

It is intender to make this recognition with tools of the OpenCv library and not using Neural Networks.


## Activities.

1. The lines are white and sometimes yellow (when there are provisional). So filtering of this colors will help the system to get the lines.
2. The lines are not going to the end of the horizon, so there is a ROI in front of our car that has to be used and the rest filtered out.
3. Then when the image left we should able to reduce the noise.
	- Blur
	
	- Canny.
	
	- Identify the lines
	
	- Filter some lines out.
	
	- Draw our own line
	
	  

## Modules

​	Two applications  have been made. 

1. An Streamlit application that allow us to play wit the parameters in order to identify the best ones.

![](C:\Users\ander\Google Drive\GitHub Projects\9_lines _on_the _road\doc_images\Image.png)

Because of the Streamlit Particularities the Image Processing module has been adapted too.

So we have an Streamlit application called **app.py** that can help us to load static images and work with them to find out the best parameters, and the one that processes video that is called **runtime.py**. This application is no Streamlit application and has to be called via the python interpreter. 



![](C:\Users\ander\Google Drive\GitHub Projects\9_lines _on_the _road\doc_images\Image2.png)

