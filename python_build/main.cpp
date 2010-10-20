#include "nvtsp.cuh"
#include <vector>
using std::vector;

int main(){
    int nGenes = 100;
    int class1 = 10;
    int class2 = 10;
    vector<double> data(nGenes*class1+nGenes*class2);
    for( int i=0; i<nGenes*(class1+class2); i++){
        data.at(i) = 1.0;
    }
    vector<int> classSizes(2);
    classSizes[0] = 10;
    classSizes[1] = 10;
    nvwrapper(data, nGenes, classSizes);

}
