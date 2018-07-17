import cv2 as cv
import numpy as np
from ctypes import *
import time

''' conf: please do not change this '''
# canvas width and height 
w, h = 800, 600
# focus distance
f = 100
'''---------------------------------'''
tools = cdll.LoadLibrary('./libtools.so')
pi = 3.1415926

def seed(t):
    tools.seed(c_uint32(t))

def copy(world_depth_new, world_color_new, world_depth, world_color):
    color_ptr = cast(world_color.ctypes.data, POINTER(c_uint8))
    depth_ptr = cast(world_depth.ctypes.data, POINTER(c_float))
    color_ptr_new = cast(world_color_new.ctypes.data, POINTER(c_uint8))
    depth_ptr_new = cast(world_depth_new.ctypes.data, POINTER(c_float))
    tools.copy(depth_ptr_new, color_ptr_new, depth_ptr, color_ptr, c_int(h), c_int(w)) 

def create():
    world_color = np.zeros([h, w, 3], dtype=np.uint8)
    world_depth = np.zeros([h, w], dtype=np.float32)
    if not world_color.flags['CONTIGUOUS']:
        world_color = np.ascontiguous(world_color, world_color.dtype)
    if not world_depth.flags['CONTIGUOUS']:
        world_depth = np.ascontiguous(world_depth, world_depth.dtype)
    return world_color, world_depth

def generate(
    world_color_new, 
    world_depth_new, 
    world_color, 
    world_depth):
    color_ptr = cast(world_color.ctypes.data, POINTER(c_uint8))
    depth_ptr = cast(world_depth.ctypes.data, POINTER(c_float))
    color_ptr_new = cast(world_color_new.ctypes.data, POINTER(c_uint8))
    depth_ptr_new = cast(world_depth_new.ctypes.data, POINTER(c_float))
    # initialize the real world
    tools.rand_world(
        depth_ptr_new,
        color_ptr_new,
        depth_ptr, 
        color_ptr,
        c_int(h), 
        c_int(w), 
        c_float(450), 
        c_float(500),
        c_float(f),
        c_int(0))
    '''
    tools.set_world_depth(
        depth_ptr,
        c_int(h),
        c_int(w),
        c_float(450),
        c_float(500),
        c_float(f)
    )
    tools.set_pixel_map(
        color_ptr,
        depth_ptr,
        c_int(h),
        c_int(w),
        c_int(30),
        c_int(30)
    )'''

def transform(
    world_color_new, 
    world_depth_new, 
    world_color, 
    world_depth,
    dx,
    dy,
    dz):
    color_ptr = cast(world_color.ctypes.data, POINTER(c_uint8))
    depth_ptr = cast(world_depth.ctypes.data, POINTER(c_float))
    color_ptr_new = cast(world_color_new.ctypes.data, POINTER(c_uint8))
    depth_ptr_new = cast(world_depth_new.ctypes.data, POINTER(c_float))
    # transform the 3d world
    tools.transform(
        depth_ptr,
        depth_ptr_new,
        color_ptr,
        color_ptr_new,
        c_int(h),
        c_int(w),
        c_float(f),
        c_float(dx),
        c_float(dy),
        c_float(dz))

def rotateX(
    world_color_new, 
    world_depth_new, 
    world_color, 
    world_depth,
    theta):
    color_ptr = cast(world_color.ctypes.data, POINTER(c_uint8))
    depth_ptr = cast(world_depth.ctypes.data, POINTER(c_float))
    color_ptr_new = cast(world_color_new.ctypes.data, POINTER(c_uint8))
    depth_ptr_new = cast(world_depth_new.ctypes.data, POINTER(c_float))
    # transform the 3d world
    tools.rotateX(
        depth_ptr,
        depth_ptr_new,
        color_ptr,
        color_ptr_new,
        c_int(h),
        c_int(w),
        c_float(f),
        c_float(theta))

def rotateY(
    world_color_new, 
    world_depth_new, 
    world_color, 
    world_depth,
    theta):
    color_ptr = cast(world_color.ctypes.data, POINTER(c_uint8))
    depth_ptr = cast(world_depth.ctypes.data, POINTER(c_float))
    color_ptr_new = cast(world_color_new.ctypes.data, POINTER(c_uint8))
    depth_ptr_new = cast(world_depth_new.ctypes.data, POINTER(c_float))
    # transform the 3d world
    tools.rotateY(
        depth_ptr,
        depth_ptr_new,
        color_ptr,
        color_ptr_new,
        c_int(h),
        c_int(w),
        c_float(f),
        c_float(theta))
   
def test():
    tools.seed(123)
    world_color, world_depth = create()
    world_color_new, world_depth_new = create()
    generate(
        world_color_new, 
        world_depth_new,
        world_color, 
        world_depth)
    i=0
    paused = 0
    v = 10
    t_beg = time.clock()
    cntr = 0
    while True:
        cntr += 1
        key = cv.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        if key==ord('p'):
            paused = 1 - paused
        if paused:
            continue
        #transform(world_color_new, world_depth_new, world_color, world_depth, i, 0, 0)
        #rotateX(world_color_new, world_depth_new, world_color, world_depth, i/1800.0*pi)
        rotateY(world_color_new, world_depth_new, world_color, world_depth, i/1800.0*pi)
        i += v
        if i==1800:
            v = -v
        elif i==-1800:
            v = -v
        else:
            pass
        cv.imshow('3D World', world_color_new)
        
    t_end = time.clock()
    print int(1.0*cntr/(t_end-t_beg)), 'FPS'
    cv.destroyAllWindows()

def build_net():
    pass
    
test()
