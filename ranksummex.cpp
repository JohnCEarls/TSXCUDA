/*=================================================================
 *
 *  ranksummex.cpp
 *  Author: Andrew Magis
 *  Calculate Wilcoxon rank sum test on matrix of gene expression data
 *
 * Inputs: expression data, list of class labels.  The algorithm
 * assumes that class label 0 is normal
 * Outputs: unsortd list of scores, sorted list of directional wilcoxon scores, 
 * and indices into original gene list
 *=================================================================*/

#include <math.h>
#include "mex.h"
#include <vector>
#include <algorithm>
#include "tiedrank.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) { 
		
	if (nrhs != 2) {
		mexErrMsgTxt("Only two inputs allowed (expression data, class labels).");
	}
	if (nlhs != 3) {
		mexErrMsgTxt("Only two outputs (unsorted ranksum scores, sorted ranksum scores, indices into original dataset)");
	}
		
    // The input must be a noncomplex single.
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Expression input must be a noncomplex single.");
    }
	if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("Class input must be a noncomplex single.");
    }			
				
	//m is the number of rows (genes)
	//n is the number of chips (samples)
	int m1 = mxGetM(prhs[0]);
	int n1 = mxGetN(prhs[0]);
	int m2 = mxGetM(prhs[1]);
	int n2 = mxGetN(prhs[1]);
	
	if (m2 != 1) {
		mexErrMsgTxt("Class labels must have a single row\n");
	}
	if (n1 != n2) {
		mexErrMsgTxt("Number of samples != number of class labels\n");
	}
		
	// Create an mxArray for the output data
	plhs[0] = mxCreateNumericMatrix(m1, 1, mxSINGLE_CLASS, mxREAL);	
	plhs[1] = mxCreateNumericMatrix(m1, 1, mxSINGLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericMatrix(m1, 1, mxUINT32_CLASS, mxREAL);
	
	// Retrieve the input data 
	float *data = (float*)mxGetPr(prhs[0]);
	float *classes = (float*)mxGetPr(prhs[1]);	
	
	// Create a pointer to the output data
	float *unsorted = (float*)mxGetPr(plhs[0]);
	float *wilcox = (float*)mxGetPr(plhs[1]);
	int *index = (int*)mxGetPr(plhs[2]);

	//Calculate the mean and standard deviation of the ranked distributions
	float na = 0.f, nb = 0.f;
	for (int i = 0; i < n2; i++) {
		nb += classes[i];
	}
	na = n2 - nb;
	//float mua = (na*(n2+1.f)) / 2.f;
	float mub = (nb*(n2+1.f)) / 2.f;
	float sigma = sqrt((na*nb*(n2+1.f)) / 12.f);
	
	std::vector< std::pair<float, int> > wilcox_scores;
	for (int i = 0; i < m1; i++) {
	
		//Define vectors for input and output of this data
		std::vector<float> col(n1, 0);
		std::vector<float> ranks(n1, 0);	
		
		//For each sample of this gene
		for (int j = 0; j < n1; j++) {
			col[j] = data[m1*j+i];
		}
				
		//Pass the two vectors to tiedrank to calculate ranks
		tiedrank(col, ranks); 
		
		//Now sum the ranks for each class
		float Ta = 0.f, Tb = 0.f;
		for (int j = 0; j < n2; j++) {
			if (classes[j] == 0) {
				Ta += ranks[j];
			} else {
				Tb += ranks[j];
			}
		}
				
		//Finally calculate the test statistic for this gene
		//do not calculate test statistic for normal, only for other
		/*
		float za = 0.f;
		if (Ta > mua) {
			za = (Ta - mua - 0.5f) / sigma;
		} else if (Ta < mua) {
			za = (Ta - mua + 0.5f) / sigma;
		}
		*/
		
		
		float zb = 0.f;
		if (Tb > mub) {
			zb = (Tb - mub - 0.5f) / sigma;
		} else if (Tb < mub) {
			zb = (Tb - mub + 0.5f) / sigma;
		}	
		
		//printf("Tb = %.3f, mub = %.3f, sigma = %.3f\n", Tb, mub, sigma);
		
		/*
		//printf("Za: %.3f Zb = %.3f\n", za, zb);
		for (int j = 0; j < n1; j++) {
			printf("<%.3f %.3f> Rank: %.3f\n", gene[j].first, gene[j].second, ranks[j]);
		}
		*/
		
		//Add this to the vector of scores along with the indices into the genes
		wilcox_scores.push_back(std::pair<float, int>(zb, i+1));
		
	}
	
	if (wilcox_scores.size() != m1) {
		mexErrMsgTxt("Error! Incorrect number of wilcox scores calculated\n");
	}
	
	//Copy the data back 
	for (int i = 0; i < m1; i++) {
		unsorted[i] = wilcox_scores[i].first;
	}
	
	//At the end of the primary loop, sort the zscores
	std::sort(wilcox_scores.rbegin(), wilcox_scores.rend());
	
	//Copy the data back into the output arrays
	for (int i = 0; i < m1; i++) {
		wilcox[i] = wilcox_scores[i].first;
		index[i] = wilcox_scores[i].second;
	}
	
}


