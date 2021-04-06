#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from random import *
import time
def addinputcolor(r,g,b,c):
	c.append(r)
	c.append(g)
	c.append(b)

def getcolor(p):
	
	s = len(p) / 3
	j = randint(0,s-1)
	#gimp.message('getcolor: j: '+str(j) + 'p: ' + str(p[3*j:3*j+3]))
	return tuple(p[3*j:3*j+3])

def gettuple(p,i):
	s = len(p) / 3
	return tuple(p[3*i:3*i+3])


def getpalette(a,b,c,l=None,n=None):
	x=a
	y=b
	z=c
	u1 = find1comp(x,y,z)
	u2 = find2comp(x,y,z)
	u3 = find3comp(x,y,z)
	u4 = find10comp(x,y,z)
	u5 = find2close(x,y,z)
	u6 = findngr(x,y,z,n)
	x=x/255
	y=y/255
	z=z/255
	addinputcolor(x,y,z,u1)
	addinputcolor(x,y,z,u2)
	addinputcolor(x,y,z,u3)
	addinputcolor(x,y,z,u4)
	addinputcolor(x,y,z,u5)
	addinputcolor(x,y,z,u6)
	if l == None:
		l=randint(0,3)
	if l == 0:
		return u1
	if l == 1:
		return u2
	if l == 2:
		return u3
	if l == 3:
		return u4
	if l == 4:
		return u5
	else:
		return u6

def HUE2RGB( p, q,  t):
	if t<0.0:
		t = t+1.
	if t>1.0:
		t = t-1.
	if t<1./6.:
		return p+(q-p)*6.*t
	if t<1./2.:
		return q
	if t<2./3.:
		return p+(q-p)*(2./3.-t)*6.
	else:
		return p

def RGB2HSL( r,  g,  b):
	r1 = float(r)/255.
	g1 = float(g)/255.
	b1 = float(b)/255.
	
	mi=min(r1,g1,b1)
	ma=max(r1,g1,b1)
	L = (ma + mi)/2.
	if ma == mi:
		H = 0.
		S = 0.
	else:
		if L<=.5:
			S=(ma-mi)/(ma+mi)
		elif L >.5:
			S=(ma-mi)/(2.0-ma-mi)
		H=0
		if ma == r1:
			H=(g1-b1)/(ma-mi) + (6. if g < b else 0.)
		elif ma == g1:
			H=(b1-r1)/(ma-mi)+2.0
		elif ma == b1:
			H=(r1-g1)/(ma-mi)+4.0
		H = H/6.
	H = H*360
	if H<0.0:
		H = H +360
	return [round(H,4),round(S,4),round(L,4)]
def HSL2RGB(H,  S,  L):
	r=0.0
	g=0.0
	b=0.0
	if S == 0.0:
		r=L
		g=L
		b=L
	else:
		tmp1 = 0.
		tmp2 = 0.
		if L < .5:
			tmp1 = L * (1.0+S)
		elif L>=.5:
			tmp1 = L+S-L*S
		tmp2 = 2*L-tmp1
		H = H/360
		r = HUE2RGB(tmp2, tmp1, H +1./3.)*255
		g = HUE2RGB(tmp2, tmp1, H)*255
		b = HUE2RGB(tmp2, tmp1, H-1./3.)*255
	return [round(r),round(g),round(b)]

def find1comp(r,g,b):
	[h,s,l] = RGB2HSL(r,g,b)
	h1 = abs(h + 180) % 360
	[r1,g1,b1] = HSL2RGB(h1,s,l)
	return [round(r1)/255.,round(g1)/255.,round(b1)/255.]

def find2comp( r,  g,  b):
	[h,s,l] = RGB2HSL(r,g,b)
	h1 = abs(h + 120) % 360
	h2 = abs(h + 240) % 360
	[r1,g1,b1] = HSL2RGB(h1,s,l)
	[r2,g2,b2] = HSL2RGB(h2,s,l)
	return [round(r1)/255.,round(g1)/255.,round(b1)/255.,round(r2)/255.,round(g2)/255.,round(b2)/255.]

def find3comp(r,g,b):
	[h,s,l] = RGB2HSL(r,g,b)
	h1 = abs(h + 90) % 360
	h2 = abs(h + 180) % 360
	h3 = abs(h + 270) % 360
	[r1,g1,b1] = HSL2RGB(h1,s,l)
	[r2,g2,b2] = HSL2RGB(h2,s,l)
	[r3,g3,b3] = HSL2RGB(h3,s,l)
	return [round(r1)/255.,round(g1)/255.,round(b1)/255.,round(r2)/255.,round(g2)/255.,round(b2)/255.,
			round(r3)/255.,round(g3)/255.,round(b3)/255.]
			
def find10comp(r,g,b):
	[h,s,l] = RGB2HSL(r,g,b)
	p = []
	ht = h
	for i in range(10):
		ht = abs(ht + 75) % 360
		p.append(HSL2RGB(ht,s,l)[0])
		p.append(HSL2RGB(ht,s,l)[1])
		p.append(HSL2RGB(ht,s,l)[2])
	p[:] = [round(xx)/255 for xx in p]
	return p
	
def find2close(r,g,b):
	[h,s,l] = RGB2HSL(r,g,b)
	h1 = abs(h + 60) % 360
	h2 = abs(h + 300) % 360	
	[r1,g1,b1] = HSL2RGB(h1,s,l)
	[r2,g2,b2] = HSL2RGB(h2,s,l)
	return [round(r1)/255.,round(g1)/255.,round(b1)/255.,round(r2)/255.,round(g2)/255.,round(b2)/255.]

def findngr(r,g,b,n=None):
	[h,s,l] = RGB2HSL(r,g,b)
	if n == None:
		n=2
	p = []
	ht = h
	for i in range(n):
		ht = abs(ht + 137.5) % 360
		p.append(HSL2RGB(ht,s,l)[0])
		p.append(HSL2RGB(ht,s,l)[1])
		p.append(HSL2RGB(ht,s,l)[2])
	p[:] = [round(xx)/255 for xx in p]
	return p

