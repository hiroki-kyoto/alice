#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void seed(__uint32_t t){
    srand(time(NULL));
}

int arg_max( 
    float * v, 
    int n){
    int i, m;
    m = 0;
    for(i=1; i<n; ++i){
        if(v[i]>v[m]){
            m = i;
        }
    }
    return m;
}

void copy(
    float * depth_new, 
    __uint8_t * color_new,
    float * depth,
    __uint8_t * color,
    int h, 
    int w
){
    memcpy(depth_new, depth, sizeof(float)*h*w);
    memcpy(color_new, color, sizeof(__uint8_t)*h*w*3);
}

void max_pool(
    float * depth_new,
    __uint8_t * color_new,
    float * depth, 
    __uint8_t * color,
    int h,
    int w,
    int i, 
    int j){
    int up, down, left, right;
    int idx[5][2];
    float v[5];
    int mid;
    up = fmin(i+1, h-1);
    down = fmax(i-1, 0);
    right = fmin(j+1, w-1);
    left = fmax(j-1, 0);
    // all neighbors
    idx[0][0] = up;
    idx[0][1] = j;
    v[0] = depth[up*w+j]; 
    idx[1][0] = i;
    idx[1][1] = left;
    v[1] = depth[i*w+left];
    idx[2][0] = i;
    idx[2][1] = right;
    v[2] = depth[i*w+right];
    idx[3][0] = down;
    idx[3][1] = j;
    v[3] = depth[down*w+j];
    idx[4][0] = i;
    idx[4][1] = j;
    v[4] = depth[i*w+j];
    
    mid = arg_max(v, 5);
    depth_new[i*w+j] = v[mid];
    color_new[i*w*3+j*3] = color[idx[mid][0]*w*3+idx[mid][1]*3];
    color_new[i*w*3+j*3+1] = color[idx[mid][0]*w*3+idx[mid][1]*3+1];
    color_new[i*w*3+j*3+2] = color[idx[mid][0]*w*3+idx[mid][1]*3+2];
}

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
                world_depth[i*w+j] = 1.0/(t+1)*(c-sqrt(delta));
            }
        }
    }
}

// randomize the depth matrix for the world
void rand_world(
    float * depth,
    __uint8_t * color,
    float * depth_new,
    __uint8_t * color_new,
    int h,
    int w, 
    float r,
    float c, 
    float f,
    int smooth_level
){
    int i, j, k, n, s, t;
    int up, down, left, right;
    float * depth_tmp;
    __uint8_t * color_tmp;
    
    for(i=0; i<h; ++i){
        for(j=0; j<w; ++j){
            /*depth[i*w+j] = rand()%256;
            color[i*w*3+j*3] = depth[i*w+j];
            color[i*w*3+j*3+1] = depth[i*w+j];
            color[i*w*3+j*3+2] = depth[i*w+j];*/
            if(i>=100&&i<200&&j>=100&&j<200){
                depth[i*w+j] = 100;
                color[i*w*3+j*3] = 200;
                color[i*w*3+j*3+1] = 100;
                color[i*w*3+j*3+2] = 50;
            }else if(i>=100&&i<200&&j>=200&&j<300){
                depth[i*w+j] = 200;
                color[i*w*3+j*3] = 50;
                color[i*w*3+j*3+1] = 100;
                color[i*w*3+j*3+2] = 200;
            }else{
                depth[i*w+j] = -1;
                color[i*w*3+j*3] = 0;
                color[i*w*3+j*3+1] = 0;
                color[i*w*3+j*3+2] = 0;
            }
        }
    }
    
    // make the world smoother
    for(k=0; k<smooth_level; ++k){
        for(i=0; i<h; ++i){
            for(j=0; j<w; ++j){
                max_pool(depth_new, color_new, depth, color, h, w, i, j); 
            }
        }
        // switch buffer
        depth_tmp = depth;
        color_tmp = color;
        depth = depth_new;
        color = color_new;
        depth_new = depth_tmp;
        color_new = color_tmp;
    }
    // make the two copies the same
    copy(depth_new, color_new, depth, color, h, w);
    /* Mente-Carlo Algorithm *//*
    n = 50*w*h;
    for(k=0; k<n; ++k){
        s = rand();
        t = rand();
        if(k%2){
            i = s%h;
            j = t%w;
        } else {
            i = t%h;
            j = s%w;
        }
        
        max_pool(depth, color, h, w, i, j);
    }*/
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

// color_ptr : RGB matrix pointer in 1-D array
// depth_ptr : depth matrix pointer in 1-D array
void rand_pixel_map(
    __uint8_t * color_ptr,
    float * depth_ptr,
    int h,
    int w
){
    int i, j, k;
    int up, down, left, right;
    
    for(i=0; i<h; ++i){
        for(j=0; j<w; ++j){
            color_ptr[i*w*3+j*3] = rand()%256;
            color_ptr[i*w*3+j*3+1] = rand()%256;
            color_ptr[i*w*3+j*3+2] = rand()%256;
        }
    }
    
    for(k=0; k<5; ++k){
        for(i=0; i<h; ++i){
            for(j=0; j<w; ++j){
                up = fmin(i+1, h);
                down = fmax(i-1, 0);
                right = fmin(j+1, w);
                left = fmax(j-1, 0);
                color_ptr[i*w*3+j*3] = (color_ptr[up*w*3+left*3] + color_ptr[up*w*3+j*3] + color_ptr[up*w*3+right*3] + color_ptr[i*w*3+left*3] + color_ptr[i*w*3+right*3] + color_ptr[down*w*3+left*3] + color_ptr[down*w*3+j*3] + color_ptr[down*w*3+right*3])/8.0;
                color_ptr[i*w*3+j*3+1] = (color_ptr[up*w*3+left*3+1] + color_ptr[up*w*3+j*3+1] + color_ptr[up*w*3+right*3+1] + color_ptr[i*w*3+left*3+1] + color_ptr[i*w*3+right*3+1] + color_ptr[down*w*3+left*3+1] + color_ptr[down*w*3+j*3+1] + color_ptr[down*w*3+right*3+1])/8.0;
                color_ptr[i*w*3+j*3+2] = (color_ptr[up*w*3+left*3+2] + color_ptr[up*w*3+j*3+2] + color_ptr[up*w*3+right*3+2] + color_ptr[i*w*3+left*3+2] + color_ptr[i*w*3+right*3+2] + color_ptr[down*w*3+left*3+2] + color_ptr[down*w*3+j*3+2] + color_ptr[down*w*3+right*3+2])/8.0;
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
    int i, j, k;
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
            if(z-dz<=0||z<=0) continue;
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


// Rotate along axis X
void rotateX(
    float * depth, 
    float * depth_new,
    __uint8_t * color, 
    __uint8_t * color_new,
    int h, 
    int w, 
    float f,
    float theta
){
    int i, j, k;
    int x, y;
    float z;
    float ratio;
    float sinval, cosval;
    
    for(i=0; i<h*w; ++i){
        depth_new[i] = -1;
        color_new[3*i] = 0;
        color_new[3*i+1] = 0;
        color_new[3*i+2] = 0;
    }
    
    // prepare dictionary for fast computation
    sinval = sin(theta);
    cosval = cos(theta);
    
    for(i=0; i<h; ++i){
        ratio = f/((i-h/2)*sinval+f*cosval);
        y = (int)(ratio*((i-h/2)*cosval-f*sinval)+ h/2);
        for(j=0; j<w; ++j){
            // update the z offset
            z = depth[i*w+j]*(cosval+sinval*(i-h/2)/f);
            if(z<=0||depth[i*w+j]<=0) continue; // exceeds the view
            x = (int)(ratio*(j-w/2) + w/2);
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
