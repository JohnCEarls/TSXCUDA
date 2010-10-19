#include "NVTSP_pywrapper.h"

void runNVTSP(std::vector<double> & data, int nGenes, std::vector<int> & classSizes ){
    nvwrapper( data, nGenes, classSizes );
}
