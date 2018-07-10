#define __KERNEL__

#include "MyComplex.h"





__global__ void julia(unsigned char * out, float c_rel,float c_im,float x0, float x1,float y0,float y1, int w, int h,int step)
{
    float delta_x=(x1-x0)/(w-1);
    float delta_y=(y1-y0)/(h-1);
    int x=16*blockIdx.x+threadIdx.x;
    int y=16*blockIdx.y+threadIdx.y;
    
    
    MyComplex z(x0+x*delta_x,y0+y*delta_y);
	MyComplex c(c_rel, c_im);
    for (int i=0;i<step;i++){
       if(z.mod()>2){
           *(out+(x*w+y)*3)=255;
       }else{
            z=z*z+c;   
       }
    }   
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
    int thread_y=16;   
    int block_x=32;
    int block_y=32;    
    
    int w=thread_x*block_x;
    int h=thread_y*block_y;
    int pic_size=thread_x*thread_y*block_x*block_y*3;
    
    
    
    
    unsigned char *out=(unsigned char *) malloc(pic_size);
    if(!out){
        printf("task is fail!\n");
    }
    
    memset(out,0,pic_size);
    


    
    unsigned char * dev_out;
    cudaMalloc((void**)&dev_out, pic_size);
    
    cudaMemcpy(dev_out, out, pic_size, cudaMemcpyHostToDevice);
    
    dim3 block(block_x,block_y),thread(thread_x,thread_y,1);
 
    julia<<<block,thread>>>(dev_out,x,y,-1,1,-1,1,w,h,1000);
    
    
    cudaMemcpy(out, dev_out, pic_size, cudaMemcpyDeviceToHost);
  
    char filename[200];
    sprintf(filename,".//julia_%0.3f_%0.3f.bin",x,y);
    FILE * f=fopen(filename,"wb");
    if(f){
        fwrite(out,1,pic_size,f);
        fclose(f);
    }  
    
    
    cudaFree(dev_out);
    free(out);
}
