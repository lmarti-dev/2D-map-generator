import random as rd
from PIL import Image
from PIL import ImageColor
from PIL import ImageChops
import cv2
import numpy as np
import scipy as sci
from scipy.ndimage import gaussian_filter
import datetime
import mapgen_utils as mpu
import WaterGenerator as wage
import ForestPainter as fopa
import capture_video as capvid
rd.seed()
xdim=1000
ydim=1000


terrain_layers=["f7eecb","fef6d7","e4f679","c4db43","a4bf0d","92a047","899064","9e9e9e","cfcfcf","e7e7e7","ffffff"]
water_layers=["263d92","344ca3","3b53ac","425ab4","465eb9","4962bd","5069c5","5d77d5"]

class TerrainDrawer:
	
	def __init__(self,xdim,ydim,tls,wls):
		
		self.resolution = 0
		self.elevation_map=np.zeros((xdim,ydim))
		self.water_map=np.zeros((xdim,ydim))
		self.mapimg = Image.new("RGB", (xdim,ydim))
		self.terrain_layers=tls
		self.water_layers=wls
	

	def show_img(self,t):
		matmap=mpu.RGB_to_CV2(self.mapimg_as_matrix(),"show")
		cv2.imshow('Map',matmap)
		cv2.waitKey(t)


	
	
	def draw_elevation_map(self):
		tempimg = Image.fromarray(self.elevation_map)
		self.mapimg=tempimg

	
	
# ===================== TERRAIN ==============================================
	def seed_terrain(self,sz,height):
		
		xdim,ydim = self.elevation_map.shape
		x=rd.randint(0,xdim)
		y=rd.randint(0,ydim)
		
		if not sz == 1 :
			for i in range(sz):
				for j in range(sz):
					self.elevation_map[np.mod(round(x-sz/2+i),xdim),np.mod(round(y-sz/2+j),ydim)]=height
		else:
			self.elevation_map[x,y]=1
	
	def erode(self, lam,scale,imb):
		oxdim,oydim= self.elevation_map.shape
		
		self.elevation_map = mpu.scale_by(self.elevation_map,scale)
		
		xdim,ydim = self.elevation_map.shape
		
		temp_elevation_map = self.elevation_map.copy()
		for x in range(xdim):
			for y in range(ydim):
				for i in (-1,1):
					for j in (-1,1):
						u = np.mod(x+i,xdim)
						v = np.mod(y+j,ydim)
						
						if self.elevation_map[x,y] > self.elevation_map[u,v]:
							if np.random.randint(0,2) == 1:
								temp_elevation_map[x,y] = temp_elevation_map[x,y] - 1/imb * lam
								temp_elevation_map[u,v] = temp_elevation_map[u,v] + imb * lam
		
		temp_elevation_map=np.clip(temp_elevation_map,0,1)
		
		self.elevation_map = temp_elevation_map
		
		self.elevation_map = mpu.scale_to(self.elevation_map,(oxdim,oydim))
	
# ===================== GENERAtE ==============================================
	
	def generate(self):
		sN=60
		for i in range(sN):
			self.seed_terrain(np.random.randint(3,88),1)
			self.draw_elevation_map()
			self.show_img(1)
			print("Seeding {} out of {}".format(i+1,sN))
		eN=10
		self.resolution=-.4
		scales=np.logspace(-1.3,self.resolution,eN)
		imb=np.linspace(.7,2,eN)
		lam=np.linspace(.2,.01,eN)
		#vid = capvid.mapVideoWriter("erode",dims=(1000,1000),framerate=2)
		for i in range(eN):
			self.erode(lam[i],scales[i],imb[i])
			self.draw_elevation_map()
			self.show_img(20)
			print("Eroding {} out of {}".format(i+1,eN))
			#vid.write_frame(self.elevation_map)
		#vid.end_recording()
		
		rN=30
		if np.any(self.elevation_map < .1):
			water_generator=wage.WaterGenerator(self.elevation_map,.1)
			water_generator.generate_rivers(rN,self.resolution)
			self.water_map = water_generator.render_water_map()
			
		self.mapimg = self.colorize()
		self.show_img(20)
		print("Colorized")
		
		fp = fopa.ForestPainter(self.elevation_map,self.water_map,.2,.4,"sprites/tree_sprite_12x12.png",scale=.5)
		forest_im = fp.paint()
		hm=.9
		
		mask = np.array(np.ceil(np.array(forest_im.convert("L")))).astype(float)
		self.mapimg=Image.composite(self.mapimg,forest_im,Image.fromarray(255-255*mask).convert("1"))
		
		self.show_img(20)
		print("Forest Added")
		
		self.mapimg = ImageChops.subtract(self.mapimg,self.shadow_rgb())
		self.show_img(20)
		print("Shadow added")
		
		while(1):
			self.show_img(0)
			k = cv2.waitKey(1)
			if k == -1:
				break
		
	def compute_shadow(self):
		a=5
		b=5
		dz= 1
		xdim,ydim = self.elevation_map.shape
		
		return dz * np.clip(np.roll(self.elevation_map,(a,b),axis=(0,1))-self.elevation_map,0,1)

	def mapimg_as_matrix(self):
		return np.array(self.mapimg)


		
	def shadow_rgb(self):
		x = self.compute_shadow()
		#cv2.imshow("bw",np.array(x))
		#cv2.waitKey(100)
		y = Image.fromarray(np.round(255*x)).convert('RGB')
		#cv2.imshow("rgb",np.array(y))
		#cv2.waitKey(100)
		return y
	
	def color_from_layers(self,hm,cm):
		indices=np.clip(np.round(hm*(len(cm)-1)),0,len(cm)-1).astype(int)
		
		colors = np.transpose(self.layers_to_rgb(cm),(1,0))
		
		# colors.shape =  rgb x palette length
		r_channel=np.take(colors[0],indices)
		g_channel=np.take(colors[1],indices)
		b_channel=np.take(colors[2],indices)
		return Image.fromarray(np.transpose(np.array([r_channel,g_channel,b_channel]),(1,2,0)).astype(np.uint8)).convert("RGB")
		
	def colorize(self):
		terrain = self.color_from_layers(self.elevation_map,self.terrain_layers)
		water = self.color_from_layers(self.elevation_map,self.water_layers)
		
		mask = Image.fromarray((255-255*np.abs(self.water_map*(1-self.elevation_map/255))))
		#cv2.imshow("mask",np.array(mask))
		#cv2.waitKey(0)
		
		#cv2.imshow("water",np.array(water))
		#cv2.imshow("terrain",np.array(terrain))
		#cv2.waitKey(300)
		
		#print(terrain.size)
		#print(water.size)
		#print(mask.size)
		
		img = Image.composite(terrain.convert("RGB"),water.convert("RGB"),mask.convert("L"))
		
		return img
		
	def to_bw(self):
		return self.mapimg.convert('LA')
		
	def rgb_from_hex(self,color):
		#returns tuple
		if color.startswith("#"):
			return ImageColor.getrgb(color)
		else:
			return ImageColor.getrgb("#"+color)
	
	def rgb_array_from_hex(self,color):
		return list(self.rgb_from_hex(color))
	
	def layers_to_rgb(self,layers):
		colors=[]
		for i in layers:
			colors.append(self.rgb_array_from_hex(i))
		return colors
	
	def save(self,m2s,filename="default"):
		x = datetime.datetime.now().strftime("%a%d_%H-%M-%S")
		matmap=mpu.RGB_to_CV2(m2s)
		if filename == "default":
			filenamer = 'isle'+x+".png"
		else:
			filenamer = filename+".png"
		cv2.imwrite(filenamer, matmap)
		print("Saved image as {}".format(filenamer))
		
	def get_dz(self,m,smooth=0):
		if smooth == 0:
			smooth=1/(10**self.resolution)
		if smooth > 1:
			m=gaussian_filter(m,smooth,mode="wrap")
		return np.array(np.gradient(m))
	
	def wrap(self,x,d):
		return np.mod(x,d)




isle = TerrainDrawer(xdim,ydim,terrain_layers,water_layers)
isle.generate()

isle.save(isle.mapimg)



