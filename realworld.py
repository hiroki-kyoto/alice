# camera capture
# using opencv2 in python 2.7

import cv2 as cv
import numpy as np
from ctypes import *

def main():
    w, h = 800, 600
    tools = cdll.LoadLibrary('./libtools.so')
    world_color = np.zeros([h, w, 3], dtype=np.uint8)
    world_depth = np.zeros([h, w], dtype=np.float32)
    
    if not world_color.flags['CONTIGUOUS']:
        world_color = np.ascontiguous(world_color, world_color.dtype)
    color_ptr = cast(world_color.ctypes.data, POINTER(c_uint8))
    if not world_depth.flags['CONTIGUOUS']:
        world_depth = np.ascontiguous(world_depth, world_depth.dtype)
    depth_ptr = cast(world_depth.ctypes.data, POINTER(c_float))
    
    # initialize the real world
    tools.set_world_depth(
        depth_ptr, 
        c_int(h), 
        c_int(w), 
        c_float(450), 
        c_float(500),
        c_float(100))
    # initialize the pixel map
    tools.set_pixel_map(
        color_ptr,
        depth_ptr,
        c_int(h),
        c_int(w),
        c_int(50),
        c_int(50))

    while True:
        
        cv.imshow('3D World', world_color)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
    
    cv.destroyAllWindows()

main()
