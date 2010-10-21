#include "nvtsp.cuh"
#include <vector>
#include <iostream>
#include <stdlib.h>
using std::cout;
using std::endl;
using std::vector;

int main(int argc, char * argv[]){
    int num;
    for(int i=1; i<argc; i++){
        num = atoi(argv[i]);
        cout << num << endl;
    }
    int nGenes = num;
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
