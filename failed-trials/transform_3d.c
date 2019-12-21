#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bitmap.h"

#define SCREEN_X 1920
#define SCREEN_Y 1080
#define PI 3.1415926
#define SUBPIXEL 1

// have to be both on range of [-1,1]
float cosine(float x){
    return x*cos(32.0*PI*x)/2;
}

// the equation to draw
int equation_satisified(
    float (*f)(float), 
    float x, 
    float y, 
    float pen_width
){
    float ry;
    ry = (*f)(2.0*x/SCREEN_X - 1); // in range of [-1, 1]
    if(fabs((ry+1)*SCREEN_Y/2-y)<=pen_width)
        return 1;
    else
        return 0;
}


// example for bmp image rendering
void test_draw_curve()
{
	pixel data[SCREEN_X*SCREEN_Y];
	for ( int i=0; i<SUBPIXEL*SCREEN_Y; i++)
	{
		for ( int j=0; j<SUBPIXEL*SCREEN_X; j++)
		{
			if (equation_satisified(cosine, 1.0*j/SUBPIXEL, 1.0*i/SUBPIXEL, 8.0))
				data[i/SUBPIXEL*SCREEN_X+j/SUBPIXEL] = 0x00ff00;
			else
				data[i/SUBPIXEL*SCREEN_X+j/SUBPIXEL] = 0x000000;
		}
	}
	// save pixel array into file
	draw_by_pixel_array( 
		LEFT_BOTTOM, 
		data, 
		SCREEN_X, 
		SCREEN_Y, 
		"demo.bmp"
	);
}



int main(){
    test_draw_curve();
}
