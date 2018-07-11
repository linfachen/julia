#define __KERNEL__

#include <stdio.h>
#include "CComplex.h"
#include "hsl2rgb.hpp"


#define MAX_COLOR 128

COLOR_RGB color_mat[MAX_COLOR];

static void init_color_mat(COLOR_RGB * colormat,float h1=137.0,float h2=30.0)
{
    COLOR_HSL hslA;
    COLOR_HSL hslB;    
    for(int i=0;i<MAX_COLOR/2;i++){
        hslA={h1,100.0f,i*200.0f/MAX_COLOR};
        hslB={h2,100.0f,i*200.0f/MAX_COLOR}; 
        HSLtoRGB(&hslA,&colormat[i]);
        HSLtoRGB(&hslB,&colormat[MAX_COLOR-i-1]);
    } 
}



__global__ void julia(unsigned char * out, float c_rel,float c_im,float x0, float x1,float y0,float y1, int w, int h,int step,COLOR_RGB * colormat)
{
    float delta_x=(x1-x0)/(w-1);
    float delta_y=(y1-y0)/(h-1);
    int x=16*blockIdx.x+threadIdx.x;
    int y=24*blockIdx.y+threadIdx.y;
    int n=1000;
    
    CComplex z(x0+x*delta_x,y0+y*delta_y);
	CComplex c(c_rel, c_im);
    for (int i=0;i<step;i++){
       if(z.mod()<4){
           z=z*z+c;
       }
       else{
            n=i;
            break;
       }
    }
    *(out+(x*h+y)*3)  =(colormat+n%MAX_COLOR)->r;
    *(out+(x*h+y)*3+1)=(colormat+n%MAX_COLOR)->g;
    *(out+(x*h+y)*3+2)=(colormat+n%MAX_COLOR)->b;  
}



int main(int argc,char **argv)
{
    
    if(argc!=3){
        printf("you must give 2 args");
        return 0;
    }
    
    float x,y;
    
    x=atof(argv[1]);
    y=atof(argv[2]);
    
    
    
    int thread_x=16;
    int thread_y=24;   
    int block_x=64;
    int block_y=64;    
    
    int w=thread_x*block_x;
    int h=thread_y*block_y;
    int pic_size=thread_x*thread_y*block_x*block_y*3;
    
    
    
    
    unsigned char *out=(unsigned char *) malloc(pic_size);
    if(!out){
        printf("task is fail!\n");
    }    
    memset(out,0,pic_size);
    
    init_color_mat(color_mat);
    
//    printf("r=%d\n",color_mat[2].r);
//    printf("g=%d\n",color_mat[2].g);   
//    printf("b=%d\n",color_mat[2].b);    
    
    unsigned char * dev_out;
    COLOR_RGB * colormat;
    
    cudaMalloc((void**)&dev_out, pic_size);
    cudaMalloc((void**)&colormat, sizeof(COLOR_RGB)*MAX_COLOR);
    cudaMemcpy(dev_out, out, pic_size, cudaMemcpyHostToDevice);
    cudaMemcpy(colormat, color_mat, sizeof(COLOR_RGB)*MAX_COLOR, cudaMemcpyHostToDevice);    
    
    dim3 block(block_x,block_y),thread(thread_x,thread_y,1);
 
    julia<<<block,thread>>>(dev_out,x,y,-1,1,-1.5,1.5,w,h,1000,colormat);
    
    cudaMemcpy(out, dev_out, pic_size, cudaMemcpyDeviceToHost);
  
    //write data to file
    char filename[200];
    sprintf(filename,".//julia_%0.3f_%0.3f.bin",x,y);
    FILE * f=fopen(filename,"wb");
    if(f){
        fwrite(out,1,pic_size,f);
        fclose(f);
    }  
    
    
    cudaFree(dev_out);
    cudaFree(colormat);    
    free(out);
}
