#ifndef NVTSP_CUH
#define NVTSP_CUH
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
void nvwrapper(std::vector<double> & data, int dsSize, std::vector<int> & classSizes );
void availableMemory(int dev, unsigned long int needed_memory);
#endif

