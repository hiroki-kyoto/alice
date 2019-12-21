// BITMAP.H
#ifndef BITMAP_H
#define BITMAP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned int pixel;
typedef unsigned char byte;
typedef unsigned short word;
typedef unsigned int dword;

typedef enum E_COORD_TYPE{
    LEFT_TOP = 0,
    LEFT_BOTTOM = 1
} __coord_t;

// BMP HEADER
typedef struct tagBITMAPFILEHEADER {
    // BMP file header (14 bytes)
    word bfType;			// file type 
    dword bfSize;			// File Size 
    word bfReserved1;		// Must be 0
    word bfReserved2;   	// Must be 0 
    dword bfOffBits;		// Offset bit
    // BMP information head (40 bytes)
    dword biSize;			// header size
    dword biWidth;			// image width
    dword biHeight;			// image height
    word biPlanes;			// must be 1
    word biBitCount;		// bits/pixel
    dword biCompression;	// compression type
    dword biSizeImage;		// image body size
    dword biXPelsPerMeter;	// horizontal resolution (pixels/meter)
    dword biYPelsPerMeter;	// vertical resolution (pixels/meter)
    dword biClrUsed;		// colour index
    dword biClrImportant;	// colour index count
    // IF THERE'S PLATE USED, THEN HERE SHOULD BE 4*NumColrs Bytes
    // COLOR TABLE	
}BMPHeader;

// check platform
int check_platform(){
    if(sizeof(byte)!=1){
        printf("Error: size of type (unsigned char) is not 1!\n");
        return 0;
    }
    if(sizeof(word)!=2){
        printf("Error: size of type (unsigned short) is not 2!\n");
        return 0;
    }
    if(sizeof(dword)!=4){
        printf("Error: size of type (unsigned int) is not 4!\n");
        return 0;
    }
    return 1;
}

// member function
void init_bmp(
    BMPHeader* bmp_p, 
    unsigned long width, 
    unsigned long height, 
    int depth
    ){
    if(!check_platform())
        return;
    bmp_p-> bfType = 0x4D42;
    unsigned long real_width = ((3*width-1)/4+1)*4; // storage alignment
    bmp_p-> bfSize = 0x36 + real_width*height;
    bmp_p-> bfReserved1 = 0x0;
    bmp_p-> bfReserved2 = 0x0;
    bmp_p-> bfOffBits = 0x36;
    // BMP INFORMATION HEADER
    bmp_p-> biSize = 0x28;
    bmp_p-> biWidth = width;
    bmp_p-> biHeight = height;
    bmp_p-> biPlanes = 0x01;
    bmp_p-> biBitCount = depth;
    bmp_p-> biCompression = 0;
    bmp_p-> biSizeImage = bmp_p->bfSize - 0x36;
    bmp_p-> biXPelsPerMeter = 0;
    bmp_p-> biYPelsPerMeter = 0;
    bmp_p-> biClrUsed = 0;
    bmp_p-> biClrImportant = 0;
};

void write_header(FILE * fp, BMPHeader * hd){
    if(!check_platform())
        return;
    // BMP file header (14 bytes)
    fwrite((char*)&hd->bfType, sizeof(word), 1, fp);
    fwrite((char*)&hd->bfSize, sizeof(dword), 1, fp);
    fwrite((char*)&hd->bfReserved1, sizeof(word), 1, fp);
    fwrite((char*)&hd->bfReserved2, sizeof(word), 1, fp);
    fwrite((char*)&hd->bfOffBits, sizeof(dword), 1, fp);
    // BMP information head (40 bytes)
    fwrite((char*)&hd->biSize, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biWidth, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biHeight, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biPlanes, sizeof(word), 1, fp);
    fwrite((char*)&hd->biBitCount, sizeof(word), 1, fp);
    fwrite((char*)&hd->biCompression, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biSizeImage, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biXPelsPerMeter, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biYPelsPerMeter, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biClrUsed, sizeof(dword), 1, fp);
    fwrite((char*)&hd->biClrImportant, sizeof(dword), 1, fp);
}

void read_header(FILE * fp, BMPHeader * hd){
    if(!check_platform())
        return;
    // file header (14 bytes)
    fread((char*)&hd->bfType, sizeof(word), 1, fp);
    fread((char*)&hd->bfSize, sizeof(dword), 1, fp);
    fread((char*)&hd->bfReserved1, sizeof(word), 1, fp);
    fread((char*)&hd->bfReserved2, sizeof(word), 1, fp);
    fread((char*)&hd->bfOffBits, sizeof(dword), 1, fp);
    // BMP information head (40 bytes)
    fread((char*)&hd->biSize, sizeof(dword), 1, fp);
    fread((char*)&hd->biWidth, sizeof(dword), 1, fp);
    fread((char*)&hd->biHeight, sizeof(dword), 1, fp);
    fread((char*)&hd->biPlanes, sizeof(word), 1, fp);
    fread((char*)&hd->biBitCount, sizeof(word), 1, fp);
    fread((char*)&hd->biCompression, sizeof(dword), 1, fp);
    fread((char*)&hd->biSizeImage, sizeof(dword), 1, fp);
    fread((char*)&hd->biXPelsPerMeter, sizeof(dword), 1, fp);
    fread((char*)&hd->biYPelsPerMeter, sizeof(dword), 1, fp);
    fread((char*)&hd->biClrUsed, sizeof(dword), 1, fp);
    fread((char*)&hd->biClrImportant, sizeof(dword), 1, fp);
}

void draw_by_pixel_array (
    __coord_t ct, // declare coord type
    pixel *	data, // pixel array
    int width, // width of image
    int height, // height of image
    const char * file_name // image name
    ){
    if(!check_platform())
        return;
    BMPHeader bitmap_h;
    unsigned short depth = 0x18;
    init_bmp(&bitmap_h, width, height, depth);
    FILE * fp = fopen(file_name, "w+b");
    // write header information into bmp file
    write_header(fp, &bitmap_h);
    // write image pixel data into file
    byte seg = 0x00;
    int real_width = ((3*width-1)/4+1)*4;
    int edge_n = real_width - 3*width;
    if(ct == LEFT_TOP){
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++)
                fwrite((char*)(data+((height-1-i)*width+j)), 3*sizeof(byte), 1, fp);
            for(int j=0; j<edge_n; j++)
                fwrite((char*)(&seg), sizeof(byte), 1, fp);
        }
    }else if(ct == LEFT_BOTTOM){
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++)
                fwrite((char*)(data+(i*width+j)), 3*sizeof(byte), 1, fp);
            for(int j=0; j<edge_n; j++)
                fwrite((char*)(&seg), sizeof(byte), 1, fp);
        }
    }else{
        printf("BAD COORD TYPE GIVEN!\n");
    }
    fclose(fp);
}

void save_data_as_bitmap(
    unsigned char** mat, 
    int width, 
    int height, 
    const char* file_name
    ){
    if(!check_platform())
        return;
    BMPHeader bitmap_h;
    unsigned short depth = 0x18;
    init_bmp(&bitmap_h, width, height, depth);
    FILE *fp = fopen(file_name, "w+b");
    // fill header
    // be careful with the memory alignment
    write_header(fp, &bitmap_h);
    // write data to colour matrix in file
    // be careful! data are put in the file upside down
    unsigned char seg = 0x00;
    // be careful with the disk alignment
    int real_width = ((3*width-1)/4+1)*4;
    int edge_n = real_width - 3*width;
    for(int i=0; i<height; i++){
        for(unsigned long j=0;j<width;j++){
            fwrite((char*)&mat[height-1-i][j], sizeof(unsigned char), 1, fp);
            fwrite((char*)&mat[height-1-i][j], sizeof(unsigned char), 1, fp);
            fwrite((char*)&mat[height-1-i][j], sizeof(unsigned char), 1, fp);
        }
        for(int j=0;j<edge_n;j++)
            fwrite((char*)&seg, sizeof(unsigned char), 1, fp);
    }
	// close file 
    fclose(fp);
}

int GetImageInfo(const char * fn, int * w_ptr, int * h_ptr){
    if(!check_platform())
        return 0;
    FILE * fp = fopen(fn, "r+b");
    BMPHeader hd;
    read_header(fp, &hd);
    *w_ptr = hd.biWidth;
    *h_ptr = hd.biHeight;
    return 1;
}

// read information from bitmap file
void read_info_from_bitmap(char * file_name)
{
    if(!check_platform())
        return;
    FILE * fp = fopen(file_name, "r+b");
    BMPHeader hd;
    read_header(fp, &hd);
    printf("Image information:\n");
    // BMP file header (14 bytes)
    printf("bfType: %x\n", hd.bfType);
    printf("bfSize: %u\n", hd.bfSize);
    printf("bfReserved1: %x\n", hd.bfReserved1);
    printf("bfReserved2: %x\n", hd.bfReserved2);
    printf("bfOffBits: %x\n", hd.bfOffBits);
    // BMP information head (40 bytes)
    printf("biSize: %u\n", hd.biSize);
    printf("biWidth: %u\n", hd.biWidth);
    printf("biHeight: %u\n", hd.biHeight);
    printf("biPlanes: %d\n", hd.biPlanes);
    printf("biBitCount: %d\n", hd.biBitCount);
    printf("biCompression: %u\n", hd.biCompression);
    printf("biSizeImage: %u\n", hd.biSizeImage);
    printf("biXResolution: %u\n", hd.biXPelsPerMeter);
    printf("biYResolution: %u\n", hd.biYPelsPerMeter);
    printf("biClrUsed: %u\n", hd.biClrUsed);
    printf("biClrImportant: %u\n", hd.biClrImportant);
}

#endif
