/*=================================================================
 *
 *  tiedrankmex.cpp
 *  Author: Andrew Magis
 *  Calculate tiedrank on matrix of data.
 *
 * Inputs: matrix of data, ranks are calculated by column
 * Outputs: tiedranks of the data
 *=================================================================*/

#include "mex.h"
#include "tiedrank.h"
#include <vector>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) { 
		
	if (nrhs != 1) {
		mexErrMsgTxt("Only one input (matrix of data).");
	}
	if (nlhs != 1) {
		mexErrMsgTxt("Only one output (ranks of data)");
	}
		
    // The input must be a noncomplex single.
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Expression input must be a noncomplex single.");
    }
			
	//m is the number of rows (genes)
	//n is the number of chips (samples)
	int m1 = mxGetM(prhs[0]);
	int n1 = mxGetN(prhs[0]);
		
	// Create an mxArray for the output data
	plhs[0] = mxCreateNumericMatrix(m1, n1, mxSINGLE_CLASS, mxREAL);
	
	// Retrieve the input data 
	float *data = (float*)mxGetPr(prhs[0]);
	
	// Create a pointer to the output data
	float *output = (float*)mxGetPr(plhs[0]);

	std::vector< std::pair<float, int> > wilcox_scores;
	
	//Main loop, looping over every column
	for (int i = 0; i < n1; i++) {
	
		std::vector<float> col(m1, 0);
		std::vector<float> ranks(m1, 0);
	
		//Copy the elements of this column into a vector
		for (int j = 0; j < m1; j++) {
			col[j] = data[m1*i+j];
		}
		
		//Pass the two vectors to tiedrank to calculate ranks
		tiedrank(col, ranks); 
		
		//Finally, put the ranks into the output matrix using the original indices
		for (int j = 0; j < m1; j++) {
			output[m1*i+j] = ranks[j];
		}
	}
}


