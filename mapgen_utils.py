import numpy as np
import cv2
from PIL import Image

def RGB_to_CV2(x,mode="save"):
	xx=np.array(x)
		
	if len(xx.shape) == 3:
		return xx[:, :, ::-1]
	else:
		if np.any(xx < 0):
			xx=np.abs(xx)
			
		if xx.max() > 1.:
			return xx
		else:
			if mode == "save":
				return xx * 255
			if mode == "show":
				return xx
			else:
				raise Exception("display option incorrect")
def scale_by(m,s):
	xdim,ydim = m.shape
	xs=int(round(xdim*s))
	ys=int(round(ydim*s))
	return np.array(Image.fromarray(m).resize((xs,ys),resample=Image.NEAREST))

def scale_to(m,dims):
	xdim,ydim = dims
	return np.array(Image.fromarray(m).resize((xdim,ydim),resample=Image.NEAREST))
	
def show_greyscale_map(self,m,t):
		cv2.imshow("map",np.abs(np.array(m)))
		cv2.waitKey(t)
		
def pick_submatrix(m,x,y):
	cst = int(x + y * ydim)
	m.take([[cst-4,cst-3,cst-2],[cst-1,cst,cst+1],[cst+2,cst+3,cst+4]],mode="wrap")

def dilate(m):
	return np.ceil(gaussian_filter(np.abs(m),3,mode="wrap"))
		
		
def cube_to_rgb_im(r=None,g=None,b=None):
	
	rb = np.any(r==None)
	gb = np.any(g==None)
	bb = np.any(b==None)
	
	if rb and gb and bb:
		raise Exception("Trying to create Null image")
	
	if rb and gb:
			r=np.zeros(b.shape)
			g=r
	
	if rb and bb:
			r=np.zeros(g.shape)
			b=r
	
	if bb and gb:
			b=np.zeros(r.shape)
			g=g
	
	return np.transpose(np.array([r,g,b]),(1,2,0))
	
def get_coord_layer(shape,mode="L"):
	xdim,ydim=shape
	text_img = np.zeros(shape)
	v=np.arange(round(xdim*.1),round(xdim*.9),round(xdim/10))
	for x in v:
		for y in v:
			cv2.putText(text_img,
				str((x,y)),
				(x,y),
				fontFace= cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=.5,
				color=(1,1,1))
	if mode=="L":
		return text_img
	if mode=="RGB":
		return np.stack([text_img,text_img,text_img],axis=2)		