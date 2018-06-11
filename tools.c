#include <math.h>
#include <string.h>

// A camera and a half of ball which is not solid 
// world_depth : the depth matrix
// h : height of world
// w : width of world
// r : radius of ball
// c : center of ball position from the origin
// f : focus distance
void set_world_depth(
    float * world_depth, 
    int h, 
    int w, 
    float r, 
    float c, 
    float f
){
    int i, j;
    float x, y;
    float cx, cy;
    float t;
    float delta;

    cx = w/2.0;
    cy = h/2.0;

    for(i=0; i<h; ++i){
        for(j=0; j<w; ++j){
            x = j - cx;
            y = i - cy;
            t = (x*x + y*y)/(f*f);
            delta = t*(r*r-c*c)+r*r;
            // if the ray does not hit any object
            if(delta<0){
                world_depth[i*w+j] = -1.0; // a negative value means infinity
            } else {
                world_depth[i*w+j] = 1.0/(t+1)*(c+sqrt(delta));
            }
        }
    }
}


void set_pixel_map(
    __uint8_t * color_ptr,
    float * depth_ptr,
    int h,
    int w,
    int uh,
    int uw
){
    int i, j;
    int row_color_idx, col_color_idx;
    
    col_color_idx = row_color_idx = 0;
    
    for(i=0; i<h; ++i){
        if(i%uh==0)
            row_color_idx = 1 - row_color_idx;
            col_color_idx = row_color_idx;
        for(j=0; j<w; ++j){
            if(j%uw==0)
                col_color_idx = 1 - col_color_idx;
            if(depth_ptr[i*w+j]>0){
                if(col_color_idx){
                    color_ptr[i*w*3+j*3] = 200; // b
                    color_ptr[i*w*3+j*3+1] = 0; // g
                    color_ptr[i*w*3+j*3+2] = 0; // r
                } else {
                    color_ptr[i*w*3+j*3] = 200; // b
                    color_ptr[i*w*3+j*3+1] = 200; // g
                    color_ptr[i*w*3+j*3+2] = 200; // r
                }
            } else {
                color_ptr[i*w*3+j*3] = 0;
                color_ptr[i*w*3+j*3+1] = 0;
                color_ptr[i*w*3+j*3+2] = 0;
            }
        }
    }
}

void transform(
    float * depth, 
    float * depth_new,
    __uint8_t * color, 
    __uint8_t * color_new,
    int h, 
    int w, 
    float f,
    float dx,
    float dy,
    float dz
){
    int i, j;
    int x, y;
    float z;
    
    for(i=0; i<h*w; ++i){
        depth_new[i] = -1;
        color_new[3*i] = 0;
        color_new[3*i+1] = 0;
        color_new[3*i+2] = 0;
    }
    
    for(i=0; i<h; ++i){
        for(j=0; j<w; ++j){
            z = depth[i*w+j];
            if(z-dz<=0 || z<=0) continue;
            x = (int)(((j-w/2)*z-dx*f)/(z-dz) + w/2);
            y = (int)(((i-h/2)*z-dy*f)/(z-dz) + h/2);
            z -= dz;
            if(x>=0 && x<w && y>=0 && y<h){
                if(depth_new[y*w+x]<=0 || z<depth_new[y*w+x]){
                    depth_new[y*w+x] = z;
                    color_new[y*w*3+x*3] = color[i*w*3+j*3];
                    color_new[y*w*3+x*3+1] = color[i*w*3+j*3+1];
                    color_new[y*w*3+x*3+2] = color[i*w*3+j*3+2];
                }
            }
        }
    }
}


void copy(
    float * depth, 
    float * depth_new, 
    __uint8_t * color,
    __uint8_t * color_new,
    int h, 
    int w
){
    memcpy(depth, depth_new, sizeof(float)*h*w);
    memcpy(color, color_new, sizeof(__uint8_t)*h*w*3);
}

int change_hair_color(__uint8_t * data, int h, int w, int c){
    int i, j, strides[2]; 
    if(c!=3){ return 1; }
    strides[0] = w*c;
    strides[1] = c;
    for(i=0; i<h; ++i){
        for(j=0; j<w; ++j){
            if(data[i*strides[0]+j*strides[1]]<30 &&
                data[i*strides[0]+j*strides[1]+1]<30 &&
                data[i*strides[0]+j*strides[1]+2]<30){
                data[i*strides[0]+j*strides[1]] = 0;
                data[i*strides[0]+j*strides[1]+1] = 200;
                data[i*strides[0]+j*strides[1]+2] = 200;
            }        
        }
    }
}
