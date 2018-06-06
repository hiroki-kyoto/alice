#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bitmap.h"

#define SCREEN_X 800
#define SCREEN_Y 600
#define PI 3.1415926

// have to be both on range of [-1,1]
float cosine(float x){
    return cos(8.0*PI*x)/4;
}

// the equation to draw
int equation_satisified(
    float (*f)(float), 
    int x, 
    int y, 
    int pen_width
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
	for ( int i=0; i<SCREEN_Y; i++)
	{
		for ( int j=0; j<SCREEN_X; j++)
		{
			if (equation_satisified(cosine, j, i, 2))
				data[i*SCREEN_X+j] = 0x00ff00;
			else
				data[i*SCREEN_X+j] = 0x000000;
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
