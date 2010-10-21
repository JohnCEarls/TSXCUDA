/*=================================================================
 *
 *  nvtspmex.cu
 *  Author: Andrew Magis
 *  Calculate TSP scores on the GPU
 *  Inputs: Class 1 data, Class 2 data, N (size of cross-validation)
 *  Outputs: TSP primary scores, TSP secondary scores, TSP upper and lower bounds for CV
 *
 *
 *=================================================================*/

#include <math.h>
//#include "mex.h"
#include <vector>
#include <iostream>
#include "nvtsp.cuh"
using std::cout;
using std::endl;
#ifndef __NVTSP_CU_
#define __NVTSP_CU_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif
void DisplayDeviceProperties(int device) {

    cudaDeviceProp deviceProp;
    memset(&deviceProp, 0, sizeof (deviceProp));
	
	printf("-----\n");
	
    if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, device)) {
		printf("Device Name\t\t\t\t%s\n", deviceProp.name);
		printf("Total Global Memory\t\t\t%ld KB\n",deviceProp.totalGlobalMem / 1024);
		printf("Maximum threads per block\t\t%d\n", deviceProp.maxThreadsPerBlock);
		printf("Can map to host memory\t\t%d\n", deviceProp.canMapHostMemory);
	    printf("Number processors\t\t%d\n", deviceProp.multiProcessorCount);
	    printf("Compute Mode\t\t%d\n", deviceProp.computeMode);
        printf("Shared Memory per Block\t\t%d\n", deviceProp.sharedMemPerBlock/1024);
        printf("Max Grid Size[0]\t\t%d\n", deviceProp.maxGridSize[0]);	
        printf("Timeout \t\t%d\n", deviceProp.kernelExecTimeoutEnabled);  	
    } else {
        printf("\n%s", cudaGetErrorString(cudaGetLastError()));
    }
	
	printf("------\n");				
		
}

#define THREADS 16
#define ABSMACRO(X) ( ((X)<0) ? (-(X)) : (X) )
#define ABSBINARYMACRO(X) ( ((X)<0) ? (0) : (1) )
#define MINMACRO(X,Y) ( ((X)<(Y)) ? (X) : (Y) )
#define MAXMACRO(X,Y) ( ((X)>(Y)) ? (X) : (Y) )

__global__ void tspKernel(float *d_class1, float *d_class2, unsigned int n1, unsigned int n2, unsigned int m, unsigned int cvn, float *primary, float *secondary, float *lower, float *upper, int *vote) {
	
    float class1_score = 0.f;
	float class2_score = 0.f;
	float class1_rank = 0.f;
	float class2_rank = 0.f;	
	float temp_lower1, temp_lower2;
	float temp_upper1, temp_upper2;
	
	float n1_invert = __fdividef(1.f, (float)n1);
	float n2_invert = __fdividef(1.f, (float)n2);
	
	//We are only building a diagonal matrix here, so return if I am part of the diagonal
	//or below the diagonal
	if ((blockIdx.x*blockDim.x+threadIdx.x) > (blockIdx.y*blockDim.y+threadIdx.y)) {
	
		//Pointers to correct memory location for class1
		float *data1 = &d_class1[(blockIdx.x*blockDim.x + threadIdx.x)];
		float *data2 = &d_class1[(blockIdx.y*blockDim.y + threadIdx.y)];

		for (int i = 0; i < n1*m; i+=m) {
			//if (data1[i] <= data2[i]) {
			//	class1_score += 1.f;
			//}	
			class1_score += signbit(data1[i]-data2[i]);
			class1_rank += (float)(data1[i]-data2[i]);
		}
		temp_lower1 = __fdividef(class1_score-(float)cvn, (float)(n1-cvn));
		temp_upper1 = __fdividef(class1_score, (float)(n1-cvn));
		class1_score = class1_score * n1_invert;
		class1_rank = class1_rank * n1_invert;

		//Pointers to correct memory location for class2
		data1 =  &d_class2[(blockIdx.x*blockDim.x + threadIdx.x)];
		data2 =  &d_class2[(blockIdx.y*blockDim.y + threadIdx.y)];
	
		for (int i = 0; i < n2*m; i+=m) {
			//if (data1[i] <= data2[i]) {
			//	class2_score += 1.f;
			//}		
			class2_score += signbit(data1[i]-data2[i]);
			class2_rank += (float)(data1[i]-data2[i]);
		}
		temp_lower2 = __fdividef(class2_score, (float)(n2-cvn));
		temp_upper2 = __fdividef(class2_score-(float)cvn, (float)(n2-cvn));
		class2_score = class2_score * n2_invert;
		class2_rank = class2_rank * n2_invert;
		
		temp_lower1 = ABSMACRO(temp_lower1 - class2_score);
		temp_lower2 = ABSMACRO(class1_score - temp_lower2); 
		temp_upper1 = ABSMACRO(temp_upper1 - class2_score);
		temp_upper2 = ABSMACRO(class1_score - temp_upper2);
	
	}

	//Write the result to global memory
	primary[(blockIdx.x*blockDim.x + threadIdx.x)*m + (blockIdx.y*blockDim.y + threadIdx.y)] = ABSMACRO(class1_score-class2_score);
	secondary[(blockIdx.x*blockDim.x + threadIdx.x)*m + (blockIdx.y*blockDim.y + threadIdx.y)] = ABSMACRO(class1_rank-class2_rank);
	lower[(blockIdx.x*blockDim.x + threadIdx.x)*m + (blockIdx.y*blockDim.y + threadIdx.y)] = MINMACRO(temp_lower1, temp_lower2);
	upper[(blockIdx.x*blockDim.x + threadIdx.x)*m + (blockIdx.y*blockDim.y + threadIdx.y)] = MAXMACRO(temp_upper1, temp_upper2);
	vote[(blockIdx.x*blockDim.x + threadIdx.x)*m + (blockIdx.y*blockDim.y + threadIdx.y)] = ABSBINARYMACRO(class1_score-class2_score);

}
/**

If I am lucky this is the only bit I'm going to have to rewrite
**/
void nvwrapper( std::vector<double> & data, int dsSize, std::vector<int> & classSizes){

    size_t f_size = sizeof(float);
   // gonna have to pass this in as a parameter, but not interested in doing that  the moment.
	unsigned int cvn = 5;//(unsigned int)(cvn_temp[0]);

	//m is the number of rows (genes)
	//n is the number of chips (samples)
	unsigned int m1 = dsSize;
	unsigned int n1 = classSizes[0];
	unsigned int m2 = dsSize;
	unsigned int n2 = classSizes[1];
	
	//Create a padded m which is multiple of THREADS
	unsigned int m;
	if (m1 % THREADS == 0) {
		m = m1;
	} else {
		m = ((int)(m1 / THREADS) + 1) * THREADS;
	}

    /**Five outputs required (TSP primary scores, TSP secondary scores, lower bounds, upper bounds, vote)
    **/

    
    //build_output_containers
    float ** TSPPrimaryScores, ** TSPSecondaryScores,** lowerbounds,** upperbounds, ** vote;
    float ** help_allocation[5];
   for(int i=0;i<5;i++){
        help_allocation[i] = new float*[m1];
        for(int j=0;j<m1;j++){
            help_allocation[i][j] = new float[m1];
        }
    }
    TSPPrimaryScores = help_allocation[0];
    TSPSecondaryScores=help_allocation[1];
    lowerbounds=help_allocation[2];
    upperbounds=help_allocation[3];
    vote=help_allocation[4];
 
    int gpu_mem;	
    //the size our vectors 
    unsigned long int class_size[2] = {m*n1*f_size, m*n2*f_size};
	unsigned long int result_size = m*m * f_size;

    //sum of gpu memory we need
    unsigned long int necessary_gpu_memory = class_size[0] + class_size[1] + (5*result_size);
    //check  available memory
    availableMemory(0, necessary_gpu_memory);

    //Allocate space on the GPU to store the input data
    float * d_class[2];
    for(int i=0;i<2;i++){
        if( cudaMalloc( (void**)&d_class[i], class_size[i] ) != cudaSuccess){
            cout << "Memory allocating failure on GPU" << endl;
            exit(1);
        }
    }

    //Allocate space on the GPU to store the output data
	float * d_s[5];
    for(int i=0;i<5;i++){
        if( cudaMalloc( (void**)&d_s[i], result_size ) != cudaSuccess){
            cout << "Memory allocating failure on GPU" << endl;
            exit(1);
        }

    } 
   

    //Allocate space on the host to store the input data
    float * h_class[2];
    int data_i = 0;
    for(int i=0;i<2;i++){
        if(cudaMallocHost((void**)&h_class[i], class_size[i] ) != cudaSuccess){
            cout << "Memory allocating failure on HOST-input" << endl;
            exit(1);
        }
        //where data aligns assign data, where padding assign zero
        int h_class_elements = class_size[i]/f_size;
        for(int pad_i=0; pad_i < h_class_elements; pad_i++){
            if(pad_i % m < m1){
                if( data_i >= data.size()){
                    cout << data_i <<endl;
                    cout << pad_i << endl;
                    cout << m1 << endl;
                    cout << m << endl;
                    cout << class_size[i];
                }
                h_class[i][pad_i] = (float)data.at( data_i);
                
                data_i++;
            } else {
                h_class[i][pad_i] = 0.0;
            }
        }
    }

    //Allocate space on the host to store the output data
	float * h_s[5];
    for(int i=0;i<5;i++){
        if( cudaMallocHost( (void**)&h_s[i], result_size ) != cudaSuccess){
            cout << "Memory allocating failure on HOST-output" << endl;
            exit(1);
        }

    } 
						
	//Copy data to the GPU
    for(int i=0;i<2;i++){
	    if ( (cudaMemcpy(d_class[i], h_class[i], class_size[i], cudaMemcpyHostToDevice) != cudaSuccess)){
    		cout << "Error copying data to the GPU." << endl;
            exit(1);
        }
	}	
	//Set the dimension of the blocks and grids
	dim3 dimBlock(THREADS, THREADS);
	dim3 dimGrid(m/THREADS, m/THREADS);
    //here we go	
	tspKernel<<<dimGrid, dimBlock>>>(d_class[0], d_class[1], n1, n2, m, cvn, d_s[0], d_s[1], d_s[2], d_s[3], (int*)d_s[4]);
	//wait for it
    cudaThreadSynchronize();
		
	//Copy the memory back
    for(int i = 0; i<5;i++){
	    if (cudaMemcpy(h_s[i], d_s[i], result_size, cudaMemcpyDeviceToHost) != cudaSuccess){
		    cout << "Error copying data from the GPU.";	
            exit(0);
        }
    }
		
	//Finally, copy the padded array data into the output matrix
    for (int help_i = 0; help_i < 5; help_i++){ 
        float ** out = help_allocation[help_i]; 
        float * padded = h_s[help_i];
        for (int i = 0; i < m1; i++) {
            memcpy(out[i], padded, m1*sizeof(float));
            padded += m;        
        }
    }
	
    //Clear up memory on the device
    for(int i=0;i<5;i++){
	    if(i<2) cudaFree(d_class[i]);
	    cudaFree(d_s[i]); 
    }
	//Clear up memory on the host
    for(int i=0;i<5;i++){
	    if(i<2) cudaFree(h_class[i]);
	    cudaFree(h_s[i]); 
    }

    //TEMP*********************************************
    //just cleanup memory for now, worry about passing back later
    for(int i=0;i<5;i++){
        for(int j=0;j<m1;j++){
            delete [] help_allocation[i][j];
        }
        delete [] help_allocation[i];
    }
 
}

/**
returns the number of bytes of memory available on dev.
FYI, if you only have 1 gpu dev is just 0
**/
void availableMemory(int dev, unsigned long int necessary_gpu_memory){


    size_t available_gpu_memory, total_gpu_memory;
    CUcontext ctx;
    cuInit(dev);

    cuCtxCreate(&ctx, 0, dev);
    cuMemGetInfo(&available_gpu_memory, &total_gpu_memory);
    cuCtxDetach(ctx);
    if(necessary_gpu_memory >available_gpu_memory 	){
        cout << "not enough memory to run this calculation"<<endl;
        cout << "Necessary: ";
        cout << necessary_gpu_memory<<endl;
        cout << "Available: ";
        cout << available_gpu_memory<<endl;
        exit(1);
    }
}
