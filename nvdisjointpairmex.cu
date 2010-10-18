/*=================================================================
 *
 *  nvdisjointpairmex.c
 *  Author: Andrew Magis
 *  Get list of all disjoint pairs in a large square matrix above a certain
 *  cutoff or a fixed number N
 *  Inputs: 2D matrix, cutoff, 0 for max Num or 1 for min cutoff
 *  Outputs: sorted disjoint pairs, index i, index j
 *=================================================================*/

#include <math.h>
#include "mex.h"
#include <vector>

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
#define REDUCTION_THREADS 128
__global__ void maxKernel(float *d_tsp, unsigned int m, unsigned int m1, float *maxValue, unsigned int *maxIndex, float *d_baddata) {

    __shared__ float sdata[REDUCTION_THREADS];
	__shared__ float sIndex[REDUCTION_THREADS];
	float s_maxValue = -1e-6;
	unsigned int s_index = 0;
	
	if (d_baddata[blockIdx.x] != 0) {
                maxValue[blockIdx.x] = 0.f;
                maxIndex[blockIdx.x] = 0.f;
		return;
 	}

	float *g_idata;
	for (unsigned int i = 0; i < m; i+=REDUCTION_THREADS) {
	
		//Set shared memory to be zero
		sdata[threadIdx.x] = 0.f;
		sIndex[threadIdx.x] = 0.f;
	
		// Go to correct loation in memory 
		g_idata = d_tsp + m*blockIdx.x + i;
		
		//Check to see if we will overshoot the actual data
		int WA = m-i > REDUCTION_THREADS ? REDUCTION_THREADS : m-i;
		
		if (threadIdx.x < WA) {
			sdata[threadIdx.x] = g_idata[threadIdx.x];
			sIndex[threadIdx.x] = m1*blockIdx.x + i + threadIdx.x;
		}
		__syncthreads();
			
		// do reduction in shared mem
		for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
			if (threadIdx.x < s) {
				if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
					sdata[threadIdx.x] = sdata[threadIdx.x + s];
					sIndex[threadIdx.x] = sIndex[threadIdx.x + s];
				}
			}
			__syncthreads();
		}
		
		// Keep track of largest element of this round
		if (threadIdx.x == 0) {
			if (sdata[0] > s_maxValue) {
				s_maxValue = sdata[0];
				s_index = sIndex[0];
			}
		}
		
	}
	
	if (threadIdx.x == 0) {
		maxValue[blockIdx.x] = s_maxValue;
		maxIndex[blockIdx.x] = s_index;
	}
	
}

__global__ void clearKernel(float *d_tsp, unsigned int m, unsigned int row, unsigned int col, float *d_baddata) {

	for (unsigned int i = 0; i < m; i+=REDUCTION_THREADS) {
		
		// Go to correct loation in memory 
		float *col_loc = d_tsp + m*col + i + threadIdx.x;
		float *row_loc = d_tsp + m*(threadIdx.x+i) + row;
		
		//Check to see if we will overshoot the actual data
		int WA = m-i > REDUCTION_THREADS ? REDUCTION_THREADS : m-i;
		
		if (threadIdx.x < WA) {
			*col_loc = 0.f;
			*row_loc = 0.f;
		}
		__syncthreads();
	}
	d_baddata[col] = 1.f;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) { 
	
	DisplayDeviceProperties(0);

	//Time the execution of this function
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    cudaEventSynchronize(start_event);
	float time_run;
		
	//Error check
	if (nrhs != 3) {
		mexErrMsgTxt("Three inputs required (2D matrix, max/cutoff, toggle).");
	}
	if (nlhs != 3) {
		mexErrMsgTxt("Three outputs required (list of max disjoint scores, index i, index j");
	}
    // The input must be a noncomplex single.
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Class1 Input must be a noncomplex single.");
    }
				
	//Get the cutoff
	double *stop_temp = (double*)mxGetPr(prhs[1]);
	float stop = (float)stop_temp[0];
	
	//Get the toggle that tells us if the user wants the top N disjoint pairs
	//or all the disjoint pairs with scores above a certain cutoff
	double *toggle_temp = (double*)mxGetPr(prhs[2]);
	unsigned int toggle = (unsigned int)toggle_temp[0];	

	if (toggle == 0) {
		printf("Requested the top %d disjoint pairs\n", (int)stop);
	} else {
		printf("Requested all disjoint pairs with scores > %.3f\n", stop);
	}	
	
	unsigned int m1 = mxGetM(prhs[0]);
	unsigned int n1 = mxGetN(prhs[0]);
	if (m1 != n1) {
		mexErrMsgTxt("Input matrix must be square");
	}	
		
	//Create a padded m which is multiple of THREADS
	unsigned int m;
	if (m1 % THREADS == 0) {
		m = m1;
	} else {
		m = ((int)(m1 / THREADS) + 1) * THREADS;
	}
	printf("Matrix Size: [%d, %d] ", m1, n1);
	printf("Thread Dimension: %d Padded length: %d\n", THREADS, m);

	unsigned long int matrix_size = m*m * sizeof(float);
	
	//Allocate space on the GPU to store the input data
	float *d_matrix;
	if ( cudaMalloc( (void**)&d_matrix, matrix_size ) != cudaSuccess )
       	mexErrMsgTxt("Memory allocating failure on the GPU.");		
	
	//Reallocate space for the data with zeroed out padding
	float *h_matrix;
	if (cudaMallocHost((void**)&h_matrix, matrix_size) != cudaSuccess) 
		mexErrMsgTxt("Memory allocating failure on the host.");
			
	//Zero out this memory
	memset(h_matrix, 0, matrix_size);
	
	//Copy over data to new padded array location
	float *temp = h_matrix;
	float *mtemp = (float*)mxGetData(prhs[0]);
	for (int i = 0; i < n1; i++) {
		memcpy(temp, mtemp, m1*sizeof(float));
		mtemp += m1;
		temp += m;
	}	
									
	//Copy data to the GPU
	if (cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		mexErrMsgTxt("Error copying memory to the GPU.");

	//Allocate space on the GPU and host for some vectors to identify 
	//used rows and columns
	float *d_maxValues, *h_maxValues, *d_maxValue, *h_maxValue;
	unsigned int *d_maxIndices, *d_maxIndex, *h_maxIndices, *h_maxIndex;
	if ( cudaMalloc( (void**)&d_maxValues, m*sizeof(float))  != cudaSuccess )
       	mexErrMsgTxt("Memory allocating failure on the GPU.");
	if ( cudaMalloc( (void**)&d_maxValue, sizeof(float))  != cudaSuccess )
       	mexErrMsgTxt("Memory allocating failure on the GPU.");	
	if ( cudaMalloc( (void**)&d_maxIndices, m*sizeof(float))  != cudaSuccess )
       	mexErrMsgTxt("Memory allocating failure on the GPU.");
	if ( cudaMalloc( (void**)&d_maxIndex, sizeof(float))  != cudaSuccess )
       	mexErrMsgTxt("Memory allocating failure on the GPU.");	
	if (cudaMallocHost((void**)&h_maxValues, m*sizeof(float)) != cudaSuccess) 
		mexErrMsgTxt("Memory allocating failure on the host.");	
	if (cudaMallocHost((void**)&h_maxValue, sizeof(float)) != cudaSuccess) 
		mexErrMsgTxt("Memory allocating failure on the host.");	
	if (cudaMallocHost((void**)&h_maxIndices, m*sizeof(float)) != cudaSuccess) 
		mexErrMsgTxt("Memory allocating failure on the host.");	
	if (cudaMallocHost((void**)&h_maxIndex, sizeof(float)) != cudaSuccess) 
		mexErrMsgTxt("Memory allocating failure on the host.");	

	float *h_baddata, *d_baddata, *h_baddata_single, *d_baddata_single;
    if (cudaMallocHost((void**)&h_baddata, m1*sizeof(float)) != cudaSuccess)
        mexErrMsgTxt("Memory allocating failure on the host.");
    if (cudaMallocHost((void**)&h_baddata_single, sizeof(float)) != cudaSuccess)
        mexErrMsgTxt("Memory allocating failure on the host.");
	memset(h_baddata, 0, m1*sizeof(float));
	memset(h_baddata_single, 0, sizeof(float));
 	if ( cudaMalloc( (void**)&d_baddata, m1*sizeof(float))  != cudaSuccess )
    	mexErrMsgTxt("Memory allocating failure on the GPU.");
    if ( cudaMalloc( (void**)&d_baddata_single, sizeof(float))  != cudaSuccess )
        mexErrMsgTxt("Memory allocating failure on the GPU.");
    if (cudaMemcpy(d_baddata, h_baddata, m1*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        mexErrMsgTxt("Error copying memory to the GPU.");
    if (cudaMemcpy(d_baddata_single, h_baddata_single, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        mexErrMsgTxt("Error copying memory to the GPU.");
		
				
	dim3 dimBlockMax(REDUCTION_THREADS, 1, 1);
	dim3 dimGridMax(m, 1, 1);
	
	std::vector<float> v_tsp;
	std::vector<int> v_row;
	std::vector<int> v_col;

	h_maxValue[0] = 1.f;	
	
	if (toggle == 0) {
	
		for (int z = 0; z < (int)stop; z++) {
	
			maxKernel<<<dimGridMax, dimBlockMax>>>(d_matrix, m, m1, d_maxValues, d_maxIndices, d_baddata);
			cudaThreadSynchronize();
			
			if (cudaMemcpy(h_maxValues, d_maxValues, m*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");
			if (cudaMemcpy(h_maxIndices, d_maxIndices, m*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");		
			
			maxKernel<<<1, dimBlockMax>>>(d_maxValues, m, m1, d_maxValue, d_maxIndex, d_baddata_single);
		
			if (cudaMemcpy(h_maxValue, d_maxValue, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");
			if (cudaMemcpy(h_maxIndex, d_maxIndex, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");	
			
			//Convert index into row/column indices
			int index = h_maxIndices[h_maxIndex[0]];
			int col = (int)floor(index/m1);
			int row = index % m1;
		
			//Add these values to the vectors
			v_tsp.push_back(h_maxValues[h_maxIndex[0]]);
			v_row.push_back(row);
			v_col.push_back(col);

			//Clear this row and column
			clearKernel<<<1, dimBlockMax>>>(d_matrix, m, row, col, d_baddata);	
			clearKernel<<<1, dimBlockMax>>>(d_matrix, m, col, row, d_baddata);			
		
		}
	
	} else {
	
		while (h_maxValue[0] > stop) {

			maxKernel<<<dimGridMax, dimBlockMax>>>(d_matrix, m, m1, d_maxValues, d_maxIndices, d_baddata);
			cudaThreadSynchronize();
			
			if (cudaMemcpy(h_maxValues, d_maxValues, m*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");
			if (cudaMemcpy(h_maxIndices, d_maxIndices, m*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");		
			
			maxKernel<<<1, dimBlockMax>>>(d_maxValues, m, m1, d_maxValue, d_maxIndex, d_baddata_single);
		
			if (cudaMemcpy(h_maxValue, d_maxValue, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");
			if (cudaMemcpy(h_maxIndex, d_maxIndex, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
				mexErrMsgTxt("Error copying memory from the GPU.");	
			
			//Convert index into row/column indices
			int index = h_maxIndices[h_maxIndex[0]];
			int col = (int)floor(index/m1);
			int row = index % m1;
		
			//Add these values to the vectors
			v_tsp.push_back(h_maxValues[h_maxIndex[0]]);
			v_row.push_back(row);
			v_col.push_back(col);

			//Clear this row and column
			clearKernel<<<1, dimBlockMax>>>(d_matrix, m, row, col, d_baddata);	
			clearKernel<<<1, dimBlockMax>>>(d_matrix, m, col, row, d_baddata);		
		
		}
		
		//Remove the last element
		if (v_tsp.size() > 0) {
			v_tsp.pop_back();
			v_row.pop_back();
			v_col.pop_back();
		}
	}
						
	//Create the output for the top scoring pairs
 	plhs[0] = mxCreateNumericMatrix(v_tsp.size(), 1, mxSINGLE_CLASS, mxREAL);
 	plhs[1] = mxCreateNumericMatrix(v_tsp.size(), 1, mxINT32_CLASS, mxREAL);
 	plhs[2] = mxCreateNumericMatrix(v_tsp.size(), 1, mxINT32_CLASS, mxREAL);	
	
	float *maxscores = (float*) mxGetData(plhs[0]);
	int *indexi = (int*) mxGetData(plhs[1]);
	int *indexj = (int*) mxGetData(plhs[2]);
	
	for (int i = 0; i < v_tsp.size(); i++) {
		maxscores[i] = v_tsp[i];
		indexi[i] = v_row[i]+1;
		indexj[i] = v_col[i]+1;
	}

	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event); // block until the event is actually recorded
	cudaEventElapsedTime(&time_run, start_event, stop_event);
	printf("Finished getting max values in %.6f seconds\n", time_run / 1000.0);	
		
	//Clear up memory on the device
	cudaFree(d_matrix);
	cudaFree(d_maxValues);
	cudaFree(d_maxValue);
	cudaFree(d_maxIndex);
	cudaFree(d_maxIndices);
	cudaFree(d_baddata);
	cudaFree(d_baddata_single);
	
	//Clear up memory on the host
	cudaFreeHost(h_matrix);
	cudaFreeHost(h_maxValues);
	cudaFreeHost(h_maxValue);	
	cudaFreeHost(h_maxIndices);
	cudaFreeHost(h_maxIndex);
	cudaFreeHost(h_baddata);
	cudaFreeHost(h_baddata_single);
		
}

