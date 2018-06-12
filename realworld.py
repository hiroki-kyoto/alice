# camera capture
# using opencv2 in python 2.7

import cv2 as cv
import numpy as np
from ctypes import *
import time

def main():
    w, h = 800, 600
    f = 100
    tools = cdll.LoadLibrary('./libtools.so')
    world_color = np.zeros([h, w, 3], dtype=np.uint8)
    world_depth = np.zeros([h, w], dtype=np.float32)
    world_color_new = np.zeros([h, w, 3], dtype=np.uint8)
    world_depth_new = np.zeros([h, w], dtype=np.float32)
    
    if not world_color.flags['CONTIGUOUS']:
        world_color = np.ascontiguous(world_color, world_color.dtype)
    color_ptr = cast(world_color.ctypes.data, POINTER(c_uint8))
    if not world_depth.flags['CONTIGUOUS']:
        world_depth = np.ascontiguous(world_depth, world_depth.dtype)
    depth_ptr = cast(world_depth.ctypes.data, POINTER(c_float))
    
    if not world_color_new.flags['CONTIGUOUS']:
        world_color_new = np.ascontiguous(world_color_new, world_color_new.dtype)
    color_ptr_new = cast(world_color_new.ctypes.data, POINTER(c_uint8))
    if not world_depth_new.flags['CONTIGUOUS']:
        world_depth_new = np.ascontiguous(world_depth_new, world_depth_new.dtype)
    depth_ptr_new = cast(world_depth_new.ctypes.data, POINTER(c_float))
    
    # initialize the real world
    tools.set_world_depth(
        depth_ptr, 
        c_int(h), 
        c_int(w), 
        c_float(450), 
        c_float(500),
        c_float(f))
    # initialize the pixel map
    tools.set_pixel_map(
        color_ptr,
        depth_ptr,
        c_int(h),
        c_int(w),
        c_int(50),
        c_int(50))
    
    i=0
    paused = 0
    t_beg = time.clock()
    while True:
        key = cv.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        if key==ord('p'):
            paused = 1 - paused
        if paused:
            continue
        # transform the 3d world
        tools.transform(
            depth_ptr,
            depth_ptr_new,
            color_ptr,
            color_ptr_new,
            c_int(h),
            c_int(w),
            c_float(f),
            c_float(i),
            c_float(i),
            c_float(-0.1*i))
        
        i += 1   
        cv.imshow('3D World', world_color_new)
        
    t_end = time.clock()
    print 1.0*i/(t_end-t_beg), ' FPS'
    cv.destroyAllWindows()

main()
