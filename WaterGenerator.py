import random as rd
from PIL import Image
from PIL import ImageColor
from PIL import ImageChops
import cv2
import numpy as np
import scipy as sci
from scipy.ndimage import gaussian_filter
import mapgen_utils as mpu
import capture_video as capvid

class WaterGenerator:
	def generate_oceans(self,wl):
		oceans = np.array(np.nonzero(self.elevation_map<wl)).astype(int)
		water_map = np.zeros(self.elevation_map.shape)
		water_map[oceans[0],oceans[1]] = -1
		return water_map 
		
	def __init__(self,elevation_map,wl):
		self.wl=wl
		self.elevation_map=elevation_map
		self.water_map=self.generate_oceans(wl)
	
	
	
	def show_lakes_rivers(self,wmap,emap,origin,goal,t,figname="im"):
		point=np.zeros(emap.shape)
		point2=np.zeros(emap.shape)
		x,y=goal
		orx,ory=origin
		point[x-6:x+6,y-6:y+6]=255
		point2[orx-6:orx+6,ory-6:ory+6]=255
		comp=mpu.get_coord_layer(emap.shape,"RGB")+mpu.cube_to_rgb_im(r=emap,g=point+point2,b=np.array(wmap==1))
		
		global vid
		vid.write_frame(comp)
			
		#cv2.imshow(figname,comp)
		#cv2.waitKey(t)
		
		
	def show_ray_search(self,emap,rsmap,t,cir,rays,figname="im"):
			xdim,ydim = emap.shape
			text_img=np.zeros((xdim,ydim))
			
			for i in range(len(rays)):
				cv2.putText(text_img,
					str(rays[i]),
					(cir[1,i],cir[0,i]),
					fontFace= cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=.5,
					color=(1,1,1))
			
			comp = np.transpose(np.array([np.zeros((xdim,ydim)),emap/255,rsmap]+text_img),(1,2,0))
			
			global vid
			vid.write_frame(comp)
			
			#cv2.imshow(figname,comp)
			#cv2.waitKey(t)
			
			
			
	def coord_circle(self,r,coord,N):
		x,y=coord
		theta=np.arange(np.pi*N)
		return np.array([x + np.round(r*np.cos(theta)),y + np.round(r*np.sin(theta))]).astype(int)

	def coord_circle_rays(self,r,coord,rays,N):
		if type(rays) != np.ndarray:
			raise Exception("Rays are {}, not np.array".format(type(rays)))
		x,y=coord
		theta=2*np.pi*rays/N
		return np.array([x + np.round(r*np.cos(theta)),y + np.round(r*np.sin(theta))]).astype(int)
		
	def find_peak_from_shore(self,coords,ev_thre_per=.9):
		xdim,ydim = self.water_map.shape
		x,y=coords
		N=50
		rays=np.arange(N)
		state = np.zeros(N)
		pre_cir=np.array([np.zeros(N),np.zeros(N)])
		rmin=5
		rmax=500
		
		ev_thre=ev_thre_per*self.elevation_map.max()
		
		ray_state_map = np.zeros((xdim,ydim))
		for r in range(rmin,rmax):
				
			cir = self.coord_circle_rays(r,coords,rays,N)
			cir_i = self.coord_circle_rays(r-1,coords,rays,N)
			cir[0]=np.mod(cir[0],xdim)
			cir[1]=np.mod(cir[1],ydim)
			cir_i[0]=np.mod(cir_i[0],xdim)
			cir_i[1]=np.mod(cir_i[1],ydim)
			
			
				#print(back_in_water)
			
			evs=self.elevation_map[cir[0],cir[1]]
			best_ev = np.argmax(evs)
			
			#print("best elevation: {}, for ray {}".format(evs[best_ev],rays[best_ev]))
			
			bool_ev= np.any(evs >= ev_thre)
			if bool_ev and evs.size!=0:
				win_ray=np.argsort(evs)[-1]
				#print("Peak found at ({},{})".format(cir[0,win_ray],cir[1,win_ray]))
				return(cir[0,win_ray],cir[1,win_ray])
			
			
			gr_rays = np.where(self.water_map[cir[0],cir[1]] != 0)
			prev_gr_rays = np.where(self.water_map[cir_i[0],cir_i[1]] == 0)
			
			back_in_water=np.intersect1d(prev_gr_rays,gr_rays)
			if back_in_water.size != 0:
				rays = np.delete(rays,back_in_water)
				if rays.size == 0:
					return None
			
			ray_state_map[cir[0],cir[1]]=1
			#self.show_ray_search(self.elevation_map,ray_state_map,10,cir,rays)
		return None
			
			
	def find_sea(self,coords):
		
		xdim,ydim = self.water_map.shape
		temp_wm = self.water_map.copy()
		x,y=coords
		for r in range(5,xdim/4,2):
			coor=coord_circle(r,(x,y),20)
			
			coor[0]=np.mod(coor[0],xdim)
			coor[1]=np.mod(coor[1],xdim)
			
			temp_wm[coor[0],coor[1]]=1
			
			#self.show_lakes_rivers(temp_wm,10)
			
			if np.any(self.water_map[coor[0],coor[1]]<0):
				ncoor=np.where(self.water_map[coor[0],coor[1]]<0)
				u=coor[0,ncoor[0][0]]
				v=coor[1,ncoor[0][0]]
				th = 10*np.sqrt((u-x)**2 + (v-y)**2)
				if np.sum(self.elevation_map[coor[0,u:x+1],coor[1,y:v+1]]) < th:
					return (u,v)

		
	def get_random_shore_point(self):
		line=np.array(np.where(self.elevation_map<self.wl))
		if line == np.array([]) or line.size==0:
			return np.unravel_index(np.argmin(self.elevation_map),self.elevation_map)
		
		
		ii = np.random.randint(0,line.shape[1])

		return (line[0,ii],line[1,ii])


	def flow_river(self,coord,scale):
			
			ev_thre_per=.9
			goal=self.find_peak_from_shore(coord,ev_thre_per)
			while goal is None:
				goal=self.find_peak_from_shore(coord,ev_thre_per)
				ev_thre_per=ev_thre_per**2
			oxr,oyr=goal
			
			oxdim,oydim = self.water_map.shape
			
			self.water_map = mpu.scale_by(self.water_map,scale)
			
			
			x,y=np.round(np.array(coord)*scale).astype(int)
			xr,yr=np.round(np.array((oxr,oyr))*scale).astype(int)
			ox,oy = x,y
			xdim,ydim = self.water_map.shape
			
			u=x
			v=y
			ui=xr
			vi=yr
			
			temp_wmap = self.water_map.copy()
			
			Niter=0
			Nitermax=1e5
			
			
			# xr,yr indiquent le sommet à chercher
			'''
			def rand_in_seeking_range(x,xdim,seeking_range):
				return np.mod(np.random.randint(x-seeking_range,x+seeking_range),xdim).astype(int)
			'''
			dist_to_summit = 100
			
			while dist_to_summit > 5 and Niter < Nitermax:
				
				dist_to_summit = np.sqrt((xr-x)**2 + (yr-y)**2)
				
				'''
				seeking_range = round(initial_sr*(((Nitermax-Niter+1)/Nitermax)**4))
				if np.random.randint(0,100)>90:
					xr=rand_in_seeking_range(x,xdim,seeking_range)
					yas=np.argsort(self.elevation_map)
					yr=yas[xr,-1]
					print("Seeking new summit at x: {}, y: {}, seeking range: {}".format(xr,yr,seeking_range))
				'''
				
				
				
				# ===== FLIPPY
				
				flipx=1
				flipy=1
				
				
				if abs(xr-x) > .5 * xdim:
					flipx=-1
				
				if abs(yr-y) > .5*ydim:
					flipy=-1
				
				# ========= MOMENTUM"
				mom=0.5
				
				mom_rand=.2
				mom_dd=.1
				sigma=3
				ddx=0
				ddy=0
				mmx = np.random.normal(scale=sigma)
				mmy = np.random.normal(scale=sigma)
				u=u+mom*flipx*np.sign(xr-x)+mom_rand*mmx + mom_dd*mmx
				v=v+mom*flipy*np.sign(yr-y)+mom_rand*mmy + mom_dd*mmy
				ddx=mmx
				ddy=mmy
				
				ui=np.mod(np.clip(np.round(u),x-1,x+1).astype(int),xdim)
				vi=np.mod(np.clip(np.round(v),y-1,y+1).astype(int),ydim)
				#if np.mod(Niter,100)==0:
				#	print("dx: {}, dy: {}".format(mom*(xr-x),mom*(yr-y)))
				#	print("Current elevation (%): {}".format(self.elevation_map[ui,vi]/np.max(self.elevation_map)))
				if not temp_wmap[ui,vi]==-1:
					temp_wmap[ui,vi]=1
				
				'''
				if x!=ui or y!=vi:
					self.show_lakes_rivers(mpu.scale_to(temp_wmap,(oxdim,oydim)),self.elevation_map,coord,(oxr,oyr),1)
					print("(x: {},y: {}) -- (xr: {}, yr: {}) -- d: {}".format(x,y,xr,yr,dist_to_summit))
					print("(u: {:.1f},v: {:.1f}) -- (ui: {}, vi: {})".format(u,v,ui,vi))
				
					print("Distance in x: {}, and x half-dim: {}".format(abs(xr-x),.5 * xdim))
					print("Distance in y: {}, and y half-dim: {}".format(abs(yr-y),.5 * ydim))
				'''
				
				if Niter>xdim/(10*mom*mom_rand) and self.water_map[ui,vi]==-1:
					return np.zeros((oxdim,oydim))
				
				x=ui
				y=vi
				
				Niter+=1
				
				#print("Iteration {}".format(Niter))
				
				
				
				
			return mpu.scale_to(temp_wmap,(oxdim,oydim))
			
	def generate_rivers(self,eN,resolution):
		
		
		i=0
		xdim,ydim=self.water_map.shape
		
		while i < eN:
			x,y = self.get_random_shore_point()
			updated_wm=self.flow_river((x,y),10**resolution)
			if not np.all(updated_wm==0):
				self.water_map=updated_wm.copy()	
				i+=1
			else:
				self.water_map = mpu.scale_to(self.water_map,(xdim,ydim))
			
			print("Adding rivers {} out of {}".format(i,eN))
			
		#vid.end_recording()
	def render_water_map(self):
		return self.water_map

'''
elevation_map=cv2.imread("emap.png",0)

xdim,ydim = elevation_map.shape

vid = capvid.mapVideoWriter(dims=(xdim,ydim),filename="rivers")


rg = WaterGenerator(elevation_map,.1)
rg.generate_rivers(3,-.4)
'''


