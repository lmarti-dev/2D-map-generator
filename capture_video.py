import numpy as np
import cv2


class mapVideoWriter:
	
	def __init__(self,filename="output",extension=".avi",dims=(640,400),framerate=60.0):
		# dimension of frame is xdim * ydim * colors * length
		self.out = cv2.VideoWriter(filename+extension,-1, framerate, dims)
	
	def write_frame(self,frame):
		self.out.write(frame)
		
	def end_recording(self):
		self.out.release()
		cv2.destroyAllWindows()
