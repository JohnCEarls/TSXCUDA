/*=================================================================
 *
 *  nvtst.c
 *  Author: Andrew Magis
 *  Calculate TSP scores on the GPU
 *
 *
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

#define THREADS 8
#define REDUCTION_THREADS 128
#define ABSMACRO(X) (((X)<0)?(-(X)):(X))

//Kernel running on the GPU
__global__ void tstKernel(float *d_class1, float *d_class2, unsigned int n1, unsigned int n2, unsigned int m, unsigned int zcoord, float *d_s1) {
		
	//Declare shared memory variables and zero them out
	__shared__ float sclass1_scores[6*THREADS*THREADS];
	__shared__ float sclass2_scores[6*THREADS*THREADS];
	float *class1_scores = &sclass1_scores[6*(threadIdx.x*THREADS+threadIdx.y)];
	float *class2_scores = &sclass2_scores[6*(threadIdx.x*THREADS+threadIdx.y)];
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		class1_scores[i] = 0.f;
		class2_scores[i] = 0.f;
	}
	
	//Pre-calculate the inverse of the two class lengths 
	float n1inverse = 1.f / (float)n1;
	float n2inverse = 1.f / (float)n2;
	
	//Shared memory array for each thread to store its own data
	__shared__ float stemp[6*THREADS*THREADS];
	float *temp = &stemp[6*(threadIdx.x*THREADS+threadIdx.y)];
		
	//We are only building a diagonal matrix here, so return if I am part of the diagonal
	//or below the diagonal
	if (((blockIdx.x*blockDim.x+threadIdx.x) > (blockIdx.y*blockDim.y+threadIdx.y)) &&
  	   ((blockIdx.y*blockDim.y+threadIdx.y) > zcoord)) {
	
		//Pointers to correct memory location for class1
		float *data1 = &d_class1[(blockIdx.x*blockDim.x + threadIdx.x)];
		float *data2 = &d_class1[(blockIdx.y*blockDim.y + threadIdx.y)];
		float *data3 = &d_class1[zcoord];

		//Registers to read from shared memory
		float sdata1, sdata2, sdata3;
		
		for (int i = 0; i < n1*m; i+=m) {
		
			//Set temp array to 0
			#pragma unroll
			for (int j = 0; j < 6; j++) {
				temp[j] = 0.f;
			}		
			float icount = 0.f;
		
			//Copy all the data into registers first
			sdata1 = data1[i]; sdata2 = data2[i]; sdata3 = data3[i];
		
			if ((sdata1 <= sdata2) && (sdata2 <= sdata3)) {
				temp[0] = 1.f;
				icount += 1.f;
			}
			if ((sdata1 <= sdata3) && (sdata3 <= sdata2)) {
				temp[1] = 1.f;
				icount += 1.f;
			}
			if ((sdata2 <= sdata1) && (sdata1 <= sdata3)) {
				temp[2] = 1.f;
				icount += 1.f;
			}
			if ((sdata2 <= sdata3) && (sdata3 <= sdata1)) {
				temp[3] = 1.f;
				icount += 1.f;
			}
			if ((sdata3 <= sdata1) && (sdata1 <= sdata2)) {
				temp[4] = 1.f;
				icount += 1.f;
			}
			if ((sdata3 <= sdata2) && (sdata2 <= sdata1)) {
				temp[5] = 1.f;
				icount += 1.f;
			}			
			
			//After we have computed all cases, if there was a tie, 
			//divide (won't happen very often)
			if (icount > 1.f) {
				#pragma unroll
				for (int j = 0; j < 6; j++) {
					temp[j] = __fdividef(temp[j], icount);
				}		
			}
			
			//Now add the result for each case to the final scores
			#pragma unroll
			for (int j = 0; j < 6; j++) {
				class1_scores[j] += temp[j];
			}
		}
		
		//At the end, scale the class1 scores by the number of elements
		#pragma unroll
		for (int i = 0; i < 6; i++) {
			class1_scores[i] *= n1inverse;
		}

		//Pointers to correct memory location for class2
		data1 = &d_class2[(blockIdx.x*blockDim.x + threadIdx.x)];
		data2 = &d_class2[(blockIdx.y*blockDim.y + threadIdx.y)];
		data3 = &d_class2[zcoord];
		
		for (int i = 0; i < n2*m; i+=m) {
		
			//Set temp array to 0
			#pragma unroll
			for (int j = 0; j < 6; j++) {
				temp[j] = 0.f;
			}		
			float icount = 0.f;
		
			//Copy all the data into registers first
			sdata1 = data1[i]; sdata2 = data2[i]; sdata3 = data3[i];
		
			if ((sdata1 <= sdata2) && (sdata2 <= sdata3)) {
				temp[0] = 1.f;
				icount += 1.f;
			}
			if ((sdata1 <= sdata3) && (sdata3 <= sdata2)) {
				temp[1] = 1.f;
				icount += 1.f;
			}
			if ((sdata2 <= sdata1) && (sdata1 <= sdata3)) {
				temp[2] = 1.f;
				icount += 1.f;
			}
			if ((sdata2 <= sdata3) && (sdata3 <= sdata1)) {
				temp[3] = 1.f;
				icount += 1.f;
			}
			if ((sdata3 <= sdata1) && (sdata1 <= sdata2)) {
				temp[4] = 1.f;
				icount += 1.f;
			}
			if ((sdata3 <= sdata2) && (sdata2 <= sdata1)) {
				temp[5] = 1.f;
				icount += 1.f;
			}			
			
			//After we have computed all cases, if there was a tie, 
			//divide (won't happen very often)
			if (icount > 1.f) {
				#pragma unroll
				for (int j = 0; j < 6; j++) {
					temp[j] = __fdividef(temp[j], icount);
				}		
			}
			
			//Now add the result for each case to the final scores
			#pragma unroll
			for (int j = 0; j < 6; j++) {
				class2_scores[j] += temp[j];
			}
		}
		
		//At the end, scale the class1 scores by the number of elements
		#pragma unroll
		for (int i = 0; i < 6; i++) {
			class2_scores[i] *= n2inverse;
		}
	}

	//Finally, sum the result
	float sum = 0.f;
	#pragma unroll	
	for (int i = 0; i < 6; i++) {
		sum += (float)ABSMACRO(class1_scores[i]-class2_scores[i]);
	}
	
	//Write the result to global memory
	d_s1[(blockIdx.x*blockDim.x + threadIdx.x)*m + (blockIdx.y*blockDim.y + threadIdx.y)] = sum;
}

__global__ void maxKernel(float *d_tsp, unsigned int m, unsigned int m1, float *maxValue, unsigned int *maxIndex) {

    __shared__ float sdata[REDUCTION_THREADS];
	__shared__ float sIndex[REDUCTION_THREADS];
	float s_maxValue = -1e-6;
	unsigned int s_index = 0;

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
			sdata[threadIdx.x] = ABSMACRO(g_idata[threadIdx.x]);
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
		mexErrMsgTxt("Three inputs required (class 1 ranks, class 2 ranks, cutoff).");
	}
	if (nlhs != 3) {
		mexErrMsgTxt("One outputs only at this time.");
	}
    // The input must be a noncomplex single.
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Class1 Input must be a noncomplex single.");
    }
	if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("Class2 Input must be a noncomplex single.");
    }		
	
	//Get the cutoff
	double *cutoff = mxGetPr(prhs[2]);
	printf("Cutoff is %.3f\n", (float)*cutoff);

	//m is the number of rows (genes)
	//n is the number of chips (samples)
	unsigned int m1 = mxGetM(prhs[0]);
	unsigned int n1 = mxGetN(prhs[0]);
	unsigned int m2 = mxGetM(prhs[1]);
	unsigned int n2 = mxGetN(prhs[1]);
	if (m1 != m2) {
		mexErrMsgTxt("Number of genes for class 1 != class 2\n");
	}	
		
	//Create a padded m which is multiple of THREADS
	unsigned int m;
	if (m1 % THREADS == 0) {
		m = m1;
	} else {
		m = ((int)(m1 / THREADS) + 1) * THREADS;
	}
	printf("Class1 Ranks: [%d, %d] Class2 Ranks: [%d, %d]\n", m1, n1, m2, n2);
	printf("Thread Dimension: %d Padded length: %d\n", THREADS, m);

	// Create an mxArray for the output data - this is automatically zeroed out
	const mwSize dims[3] = {m1, m1, m1};
	plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericMatrix(1, 3, mxINT32_CLASS, mxREAL);		

	unsigned long int class1_size = m*n1 * sizeof(float);
	unsigned long int class2_size = m*n2 * sizeof(float);
	unsigned long int result_size_gpu = m*m * sizeof(float);
	
	//Allocate space on the GPU to store the input data
	float *d_class1, *d_class2, *d_s1;
	if ( cudaMalloc( (void**)&d_class1, class1_size ) != cudaSuccess )
       	mexErrMsgTxt("Memory allocating failure on the GPU.");
	if ( cudaMalloc( (void**)&d_class2, class2_size )  != cudaSuccess )
        mexErrMsgTxt("Memory allocating failure on the GPU.");
    if ( cudaMalloc( (void**)&d_s1, result_size_gpu )  != cudaSuccess )
        mexErrMsgTxt("Memory allocating failure on the GPU.");
	
	//Reallocate space for the data with zeroed out padding
	float *h_class1, *h_class2, *h_s1;
	if (cudaMallocHost((void**)&h_class1, class1_size) != cudaSuccess) 
		mexErrMsgTxt("Memory allocating failure on the host.");
	if (cudaMallocHost((void**)&h_class2, class2_size) != cudaSuccess)
		mexErrMsgTxt("Memory allocating failure on the host.");
	if (cudaMallocHost((void**)&h_s1, result_size_gpu*m) != cudaSuccess) 
		mexErrMsgTxt("Memory allocating failure on the host.");
						
	//Zero out this memory
	memset(h_class1, 0, class1_size);
	memset(h_class2, 0, class2_size);
	memset(h_s1, 0, result_size_gpu*m);
	
	//Copy over data to new padded array location
	float *temp = h_class1;
	float *mtemp = (float*)mxGetData(prhs[0]);
	for (int i = 0; i < n1; i++) {
		memcpy(temp, mtemp, m1*sizeof(float));
		mtemp += m1;
		temp += m;
	}	
	temp = h_class2;
	mtemp = (float*)mxGetData(prhs[1]);
	for (int i = 0; i < n2; i++) {
		memcpy(temp, mtemp, m1*sizeof(float));
		mtemp += m1;
		temp += m;
	}		
									
	//Copy data to the GPU
	if (cudaMemcpy(d_class1, h_class1, class1_size, cudaMemcpyHostToDevice) != cudaSuccess)
		mexErrMsgTxt("Error copying memory to the GPU.");
	if (cudaMemcpy(d_class2, h_class2, class2_size, cudaMemcpyHostToDevice) != cudaSuccess)
		mexErrMsgTxt("Error copying memory to the GPU.");
		
	//Set the dimension of the blocks and grids
	dim3 dimBlock(THREADS, THREADS);
	dim3 dimGrid(m/THREADS, m/THREADS);	
	printf("Scheduling [%d %d] threads in [%d %d] blocks for %d executions\n", THREADS, THREADS, m/THREADS, m/THREADS, m1);
		
	//We will overlap execution with data processing with streams
	cudaStream_t *stream = new cudaStream_t[m1];
	for (unsigned int i = 0; i < m1; i++) {
		cudaStreamCreate(&stream[i]);
	}		
	
	//Call the kernel
	for (unsigned int i = 0; i < m1; i++) {
		tstKernel<<<dimGrid, dimBlock, 0, stream[i]>>>(d_class1, d_class2, n1, n2, m, i, d_s1);
	}
	
	//Copy the memory back
	for (unsigned int i = 0; i < m1; i++) { 
		if (cudaMemcpyAsync(&h_s1[i*m*m], d_s1, result_size_gpu, cudaMemcpyDeviceToHost, stream[i]) != cudaSuccess) 
			mexErrMsgTxt("Error copying memory to the host.");
	}
		
	//Make sure all copies are complete before continuing
	cudaThreadSynchronize();
	
	//Destroy the streams
	for (int i = 0; i < m1; i++) {
		cudaStreamDestroy(stream[i]);
	}
	delete[] stream;
	
	/*
	//No streams
	for (int i = 0; i < m1; i++) {
	
		tstKernel<<<dimGrid, dimBlock, 0>>>(d_class1, d_class2, n1, n2, m, i, d_s1);
		cudaThreadSynchronize();

		if (cudaMemcpy(&h_s1[i*m*m], d_s1, result_size_gpu, cudaMemcpyDeviceToHost) != cudaSuccess) 
			mexErrMsgTxt("Error copying memory to the host.");
		
	}*/
	
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event); // block until the event is actually recorded
	cudaEventElapsedTime(&time_run, start_event, stop_event);
	printf("Finished running nvTST in %.6f seconds\n", time_run / 1000.0);
	cudaEventRecord(start_event, 0);
    cudaEventSynchronize(start_event);	
		
	//Get pointer to the matlab output array
	float *matlab_output1 = (float*) mxGetData(plhs[0]);

	
	//Temp output
	/*
	float *gpu_output1 = h_s1;
	for (int i = 0; i < 512; i++) {
		printf("%d %.3f\n", i, gpu_output1[i]);
	}*/
	
	//Finally, copy the padded array data into the output matrix
	for (int i = 0; i < m1; i++) {
		float *gpu_output1 = &h_s1[i*m*m];
		for (int j = 0; j < m1; j++) {
			memcpy(matlab_output1, gpu_output1, m1*sizeof(float));
			matlab_output1 += m1; 
			gpu_output1 += m; 
		}	
	}
	
	/*
	
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
				
	dim3 dimBlockMax(REDUCTION_THREADS, 1, 1);
	dim3 dimGridMax(m, 1, 1);

	maxKernel<<<dimGridMax, dimBlockMax>>>(d_s1, m, m1, d_maxValues, d_maxIndices);
	cudaThreadSynchronize();
		
	if (cudaMemcpy(h_maxValues, d_maxValues, m*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
		mexErrMsgTxt("Error copying memory from the GPU.");
	if (cudaMemcpy(h_maxIndices, d_maxIndices, m*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
		mexErrMsgTxt("Error copying memory from the GPU.");		
		
	maxKernel<<<1, dimBlockMax>>>(d_maxValues, m, m1, d_maxValue, d_maxIndex);
	
	if (cudaMemcpy(h_maxValue, d_maxValue, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
		mexErrMsgTxt("Error copying memory from the GPU.");
	if (cudaMemcpy(h_maxIndex, d_maxIndex, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) 
		mexErrMsgTxt("Error copying memory from the GPU.");	
		
	//Convert index into row/column indices
	int index = h_maxIndices[h_maxIndex[0]];
	int col = (int)floor(index/m1);
	int row = index % m1;
		
		//Add these values to the vectors
		//v_tsp.push_back(h_maxValues[h_maxIndex[0]]);
		//v_row.push_back(row);
		//v_col.push_back(col);

		//Clear this row and column
		//clearKernel<<<1, dimBlockMax>>>(d_s1, m, row, col, d_baddata);	
		//clearKernel<<<1, dimBlockMax>>>(d_s1, m, col, row, d_baddata);			
	float *maxtsp = (float*) mxGetData(plhs[1]);
	int *maxindices = (int*) mxGetData(plhs[2]);

	//Copy the data back
	maxtsp[0] = h_maxValues[h_maxIndex[0]];
	maxindices[0] = row+1;
	maxindices[1] = col+1; 
	*/
		
	//Clear up memory on the device
	cudaFree(d_class1);
	cudaFree(d_class2);
	cudaFree(d_s1); 
	cudaFreeHost(h_class1);
	cudaFreeHost(h_class2);
	cudaFreeHost(h_s1); 
		
}


