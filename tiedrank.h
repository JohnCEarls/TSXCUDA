/*=================================================================
 *
 *  tiedrank.h
 *  Author: Andrew Magis
 *  Calculate tiedrank on vector of data. Output vector is
 *  passed in by reference.  All tiedranks are averaged.
 *
 * Inputs: reference to vector of floats
 * Outputs: void
 *=================================================================*/

#include <vector>
#include <algorithm>
void tiedrank(std::vector<float> &data, std::vector<float> &ranks) {

	//Clear the ranks, in case it has anything in it
	ranks.clear();
	ranks.resize(data.size(), 0);

	//Vector of pairs to store the indices
	std::vector< std::pair<float, int> > col;

	//Inner loop, looping over every row
	for (int j = 0; j < data.size(); j++) {
	
		//Copy the element into the pair template with index
		std::pair<float, int> element(data[j], j);
	
		//Add this pair to the vector
		col.push_back(element);
		
	}
		
	//Sort the vector using quicksort
	std::sort(col.begin(), col.end());
	std::vector<float> temp;

	int crank = 1, tied = 1;
	float average = 1.f;
	temp.push_back(crank);
	for (int j = 1; j < col.size(); j++) {
	
		//Compare this one to the previous one
		if (col[j].first == col[j-1].first) {
			tied++;
			average += ++crank;
			temp.push_back(-1);
		} else {
		
			//If they are not equal, and tied == 0, do nothing but add in
			//the current rank
			if (tied == 1) {
				average = ++crank;
				temp.push_back(crank);
			
			//Otherwise, go back through and average the ranks for all tied elements
			} else {
				average /= (float)tied;
				for (int k = temp.size()-tied; k <= temp.size()-1; k++) {
					temp[k] = average;
				}
				average = (float)++crank;
				temp.push_back(crank);
				tied = 1;
			}
		}
	}
	
	//Check to make sure the final one is not tied
	if (tied > 1) {
		average /= (float)tied;		
		for (int k = temp.size()-tied; k <= temp.size()-1; k++) {
			temp[k] = average;
		}		
	}
	
	/*
	for (int j = 0; j < col.size(); j++ ) {
		printf("%d: %.3f  %d  %.3f\n", j, col[j].first, col[j].second, temp[j]);
	}*/
		
	//Finally, put the ranks into the output vector using the original indices
	for (int j = 0; j < col.size(); j++) {
		ranks[col[j].second] = temp[j];
	}
}



