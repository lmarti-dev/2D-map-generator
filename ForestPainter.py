import cv2
import numpy as np
from PIL import Image
import capture_video as capvid

class ForestPainter:
	def __init__(self,emap,wmap,lo,hi,filename,scale=1):
		self.elevation_map=emap
		self.water_map=wmap
		self.lo=lo
		self.hi=hi
		self.default_sprite=self.load_sprite(filename,scale)
	
	def load_sprite(self,filename,scale):
		im = cv2.cvtColor(cv2.imread(filename,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGBA)
		
		szx,szy,_ = im.shape
		if scale != 1:
			im=cv2.resize(im,(round(szx*scale),round(szy*scale)),interpolation=cv2.INTER_NEAREST)
			
		print(im.shape)
		return im
		
	def generate_sprite(self,sz=5):
		vals=np.arange(self.palette)
		A=np.zeros(sz,sz)
		L=np.zeros(sz,sz)
		m = np.floor(sz/2).astype(int)
		
		for i in range(m+1):
			L[i,m]=vals[0]
		
			
	def paint(self,spacing=3):
		xdim,ydim=self.elevation_map.shape
		szx,szy,_=self.default_sprite.shape
		
		forest_layer = np.array(self.elevation_map < self.hi) * np.array(self.elevation_map > self.lo)
		
		
		forest_im=Image.fromarray(np.zeros((xdim,ydim,3)),mode="RGBA")
		
		sprite_gs=255*np.array(np.ceil(np.array((np.array(cv2.cvtColor(self.default_sprite, cv2.COLOR_RGB2GRAY))))/255)).astype(np.uint8)
		print("Tree shape:")
		print(sprite_gs)
		mask = Image.fromarray(sprite_gs,mode="L")
		
		#cv2.imshow("sprite",cv2.resize(np.array(mask), (600,600),interpolation=cv2.INTER_NEAREST))
		#cv2.waitKey(0)
		if np.all(forest_layer==0):
			return None
			
		for x in range(1,xdim,spacing):
			for y in range(ydim-1,0,-spacing):
				if np.random.normal(1,2)>0.9 and forest_layer[x,y] and np.abs(self.water_map[x,y])<1:
					u = np.mod(x-round(szx/2),xdim)
					v = np.mod(y-round(szy/2),ydim)
					
					forest_im.paste(Image.fromarray(self.default_sprite,mode="RGBA"),box=(v,u),mask=mask)
					#cv2.imshow("sprite",np.array(forest_im))
					#cv2.waitKey(1)
					
					#print("{},{}".format(x,y))
		#vid.end_recording()
		return forest_im.convert("RGB")
"""
pal=["72412E","8C9D2D","A8B562"]

emap=cv2.imread("emap.png",0)/255
fp = ForestPainter(emap,0.3,0.4,"tree_sprite_12x12.png",scale=.5)

forest = fp.paint()
mask = np.array(np.ceil(np.array(forest.convert("L")))).astype(float)

#cv2.imshow("mask",mask)
#cv2.waitKey(-1)
comp=np.array(Image.composite(Image.fromarray(255*emap),forest,Image.fromarray(255-255*mask).convert("1")))
cv2.imwrite("forest.png",comp)
cv2.imshow("sprite",comp)
#cv2.imshow("sprite",cv2.resize(fp.default_sprite, (600,600),interpolation=cv2.INTER_NEAREST))
cv2.waitKey(-1)

"""