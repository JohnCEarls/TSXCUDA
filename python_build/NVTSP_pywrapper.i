%module nvtsp
#define PySwigIterator nvtsp_PySwigIterator
%{
    #include "NVTSP_pywrapper.h"
%}

%include stl.i
namespace std {
    %template(IntVector) vector<int>;
    %template(DoubleVector) vector<double>;
    %template(BoolVector) vector<bool>;
}

%include "NVTSP_pywrapper.h"


