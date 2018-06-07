#include <math.h>

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
