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
void nvwrapper(std::vector<double> & data, int dsSize, std::vector<int> & classSizes ){

	
	DisplayDeviceProperties(0);

	//Time the execution of this function
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    cudaEventSynchronize(start_event);
	float time_run;
	/**	
	//Error check
	if (nrhs != 3) {
		mexErrMsgTxt("Three inputs required (class 1 ranks, class 2 ranks, N-fold cross-validation).");
	}
	if (nlhs != 5) {
		mexErrMsgTxt("Five outputs required (TSP primary scores, TSP secondary scores, lower bounds, upper bounds, vote)");
	}
   
    // The input must be a noncomplex single.
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Class1 Input must be a noncomplex single.");
    }
	if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("Class2 Input must be a noncomplex single.");
    }		
    **/	
	//Get the size of cross-validation we want
    // gonna have to pass this in.
	unsigned int cvn = 5;//(unsigned int)(cvn_temp[0]);
	printf("Size of cross-validation is is %u\n", cvn);

	//m is the number of rows (genes)
	//n is the number of chips (samples)
	unsigned int m1 = dsSize;//mxGetM(prhs[0]);
	unsigned int n1 = classSizes[0];//mxGetN(prhs[0]);
	unsigned int m2 = dsSize;//mxGetM(prhs[1]);
	unsigned int n2 = classSizes[0];//mxGetN(prhs[1]);
    /**
	if (m1 != m2) {
		mexErrMsgTxt("Number of genes for class 1 != class 2\n");
	}
    **/	
		
	//Create a padded m which is multiple of THREADS
	unsigned int m;
	if (m1 % THREADS == 0) {
		m = m1;
	} else {
		m = ((int)(m1 / THREADS) + 1) * THREADS;
	}
	printf("Class1 Ranks: [%d, %d] Class2 Ranks: [%d, %d]\n", m1, n1, m2, n2);
	printf("Thread Dimension: %d Padded length: %d\n", THREADS, m);

    /**Five outputs required (TSP primary scores, TSP secondary scores, lower bounds, upper bounds, vote)
    **/
	// Create an mxArray for the output data - this is automatically zeroed out
    //near as I can tell we are creating 5 ngenes x ngenes arrays, four floats and one int
    /**
    May be able to get away with using vectors
    **/
    float ** TSPPrimaryScores = new float*[m1];
    float ** TSPSecondaryScores = new float*[m1];
    float ** lower_bounds = new float*[m1];
    float ** upper_bounds = new float*[m1];
    int **  vote = new int*[m1];
     for(int i=0; i<m1 ; i++){
        TSPPrimaryScores[i] = new float[m1];
        TSPSecondaryScores[i] = new float[m1];
        lower_bounds[i] = new float[m1];
        upper_bounds[i] = new float[m1];
        vote[i] = new int[m1];
        /**
        Dont think this is necessary
        for(int j = 0; j<m1; j++){
            TSPPrimaryScores[i][j] = 0.0;
            TSPSecondaryScores[i][j] = 0.0; 
            lower_bounds[i][j] = 0.0;
            upper_bounds[i][j] = 0.0;
            vote[i][j] = 0;
        }**/
    } 
    /**
	plhs[0] = mxCreateNumericMatrix(m1, m1, mxSINGLE_CLASS, mxREAL);	
	plhs[1] = mxCreateNumericMatrix(m1, m1, mxSINGLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericMatrix(m1, m1, mxSINGLE_CLASS, mxREAL);	
	plhs[3] = mxCreateNumericMatrix(m1, m1, mxSINGLE_CLASS, mxREAL);
	plhs[4] = mxCreateNumericMatrix(m1, m1, mxINT32_CLASS, mxREAL);
	**/
	unsigned long int class1_size = m*n1 * sizeof(float);
	unsigned long int class2_size = m*n2 * sizeof(float);
	unsigned long int result_size = m*m * sizeof(float);
	
	//Allocate space on the GPU to store the input data
	float *d_class1, *d_class2;
	if ( (cudaMalloc( (void**)&d_class1, class1_size ) != cudaSuccess ) ||  cudaMalloc( (void**)&d_class2, class2_size )  != cudaSuccess ){
            cout << "Memory allocating failure on the GPU.";
    }
			
	//Allocate space on the GPU to store the output data
	float *d_s1, *d_s2, *d_s3, *d_s4, *d_s5;	
    if( ( cudaMalloc( (void**)&d_s1, result_size )  != cudaSuccess )
    || ( cudaMalloc( (void**)&d_s2, result_size )  != cudaSuccess )
    || ( cudaMalloc( (void**)&d_s3, result_size )  != cudaSuccess )
    || ( cudaMalloc( (void**)&d_s4, result_size )  != cudaSuccess )
    || ( cudaMalloc( (void**)&d_s5, result_size )  != cudaSuccess )){
			cout << "Memory allocating failure on the GPU.";
    }
			
	//Reallocate space for the data on the host with zeroed out padding
	float *h_class1, *h_class2, *h_s1, *h_s2, *h_s3, *h_s4, *h_s5;
	if ((cudaMallocHost((void**)&h_class1, class1_size) != cudaSuccess) 
	|| (cudaMallocHost((void**)&h_class2, class2_size) != cudaSuccess)
	|| (cudaMallocHost((void**)&h_s1, result_size) != cudaSuccess) 
	|| (cudaMallocHost((void**)&h_s2, result_size) != cudaSuccess) 
	|| (cudaMallocHost((void**)&h_s3, result_size) != cudaSuccess) 
	|| (cudaMallocHost((void**)&h_s4, result_size) != cudaSuccess) 
	|| (cudaMallocHost((void**)&h_s5, result_size) != cudaSuccess) ){
	    cout <<	"Memory allocating failure on the host.";	
    }
		
	//Zero out the memory on the host
	memset(h_class1, 0, class1_size);
	memset(h_class2, 0, class2_size);
	
	//Copy over data to new padded array location on host
    //k back to near as I can tell
    //this appears to be copying the data into the GPU
    //prob a good time to make our dataFloatArray
    //may not have to do this.
    float * mtemp = new float[data.size()];

    for (int i = 0; i<data.size();i++){
        mtemp[i] = (float)data.at(i);//might as well be explicit
    }
	float *temp = h_class1;
	//float *mtemp = (float*)mxGetData(prhs[0]);
	for (int i = 0; i < n1; i++) {
		memcpy(temp, mtemp, m1*sizeof(float));
		mtemp += m1;
		temp += m;
	}	
	temp = h_class2;

	for (int i = 0; i < n2; i++) {
		memcpy(temp, mtemp, m1*sizeof(float));
		mtemp += m1;
		temp += m;
	}		
							
	//Copy data to the GPU
	if ( (cudaMemcpy(d_class1, h_class1, class1_size, cudaMemcpyHostToDevice) != cudaSuccess) || (cudaMemcpy(d_class2, h_class2, class2_size, cudaMemcpyHostToDevice) != cudaSuccess)){
		cout << "Error copying memory to the GPU.";
    }
		
	//Set the dimension of the blocks and grids
	dim3 dimBlock(THREADS, THREADS);
	dim3 dimGrid(m/THREADS, m/THREADS);
	
	printf("Scheduling [%d %d] threads in [%d %d] blocks\n", THREADS, THREADS, m/THREADS, m/THREADS);
	tspKernel<<<dimGrid, dimBlock>>>(d_class1, d_class2, n1, n2, m, cvn, d_s1, d_s2, d_s3, d_s4, (int*)d_s5);
	cudaThreadSynchronize();
		
	//Copy the memory back
	if ((cudaMemcpy(h_s1, d_s1, result_size, cudaMemcpyDeviceToHost) != cudaSuccess) ||
	 (cudaMemcpy(h_s2, d_s2, result_size, cudaMemcpyDeviceToHost) != cudaSuccess) 
	|| (cudaMemcpy(h_s3, d_s3, result_size, cudaMemcpyDeviceToHost) != cudaSuccess) 
	|| (cudaMemcpy(h_s4, d_s4, result_size, cudaMemcpyDeviceToHost) != cudaSuccess) 
	|| (cudaMemcpy(h_s5, d_s5, result_size, cudaMemcpyDeviceToHost) != cudaSuccess) )
		cout << "Error copying memory from the GPU.";	
		
	float *gpu_output1 = h_s1, *gpu_output2 = h_s2, *gpu_output3 = h_s3, *gpu_output4 = h_s4, *gpu_output5 = h_s5;

/**	float *matlab_output1 = (float*) mxGetData(plhs[0]);		
	float *matlab_output2 = (float*) mxGetData(plhs[1]);
	float *matlab_output3 = (float*) mxGetData(plhs[2]);
	float *matlab_output4 = (float*) mxGetData(plhs[3]);	
	float *matlab_output5 = (float*) mxGetData(plhs[4]);
**/	
	//Finally, copy the padded array data into the output matrix
	for (int i = 0; i < m1; i++) {
		memcpy(TSPPrimaryScores, gpu_output1, m1*sizeof(float));
		memcpy(TSPSecondaryScores, gpu_output2, m1*sizeof(float));
		memcpy(lower_bounds, gpu_output3, m1*sizeof(float));
		memcpy(upper_bounds, gpu_output4, m1*sizeof(float));
		memcpy(vote, gpu_output5, m1*sizeof(float));			
		TSPPrimaryScores += m1; TSPSecondaryScores += m1; lower_bounds += m1; upper_bounds += m1; vote += m1;
		gpu_output1 += m; gpu_output2 += m; gpu_output3 += m; gpu_output4 += m; gpu_output5 += m;
	}		
	
    /**
    TODO
    Memory cleanup and pushing data into output
    

    **/
	
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event); // block until the event is actually recorded
	cudaEventElapsedTime(&time_run, start_event, stop_event);
	printf("Finished running nvTSP in %.6f seconds\n", time_run / 1000.0);
	cudaEventRecord(start_event, 0);
    cudaEventSynchronize(start_event);	
	
	//Clear up memory on the device
	cudaFree(d_class1);
	cudaFree(d_class2);
	cudaFree(d_s1); 
	cudaFree(d_s2);
	cudaFree(d_s3);
	cudaFree(d_s4);
	cudaFree(d_s5);
	
	//Clear up memory on the host
	cudaFreeHost(h_class1);
	cudaFreeHost(h_class2);
	cudaFreeHost(h_s1); 
	cudaFreeHost(h_s2);
	cudaFreeHost(h_s3);
	cudaFreeHost(h_s4);	
	cudaFreeHost(h_s5);
		
}

