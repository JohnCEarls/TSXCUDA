function classifiers = classify_cuda(tsp_struct, k)
%CLASSIFY_CUDA Takes the output of a run of TSP and finds the best classifiers
%
%   [RESULT] = TSP_CUDA(TSP_STRUCT) assumes the class identifiers are in the first row
%   of the DATA matrix and removes it.  DATA is then split based on the class
%   labels and the TSP algorithm is performed.  Rows are assumed to be probes and 
%   columns experiments.  The header row must contain only zeroes and ones for the 
%   two class labels.  Cross validation is assumed to be LOOCV (leave one out).
% 
%  TODO

% Check inputs
if (nargin < 1)
	error('Usage: [CLASSIFIERS] = CLASSIFY_CUDA(TSP_STRUCT)');
end


% Check to make sure tsp_struct is a struct
if ~isstruct(tsp_struct)
	error('Invalid input (should be tsp_struct format)');
end

% If k is 1, we just get the top scoring pair
[sorted, indexi, indexj] = nvdisjointpairmex(tsp_struct.primary, k, 0);

% If we have filtered the data by differential expression, we want to return 
% indices and names relative to the original data set
classifiers = [];

for j=1:k
	
	classifer = struct;
	if (tsp_struct.filter ~= 0)
		classifier.indexi = tsp_struct.indices(indexi(j));
		classifier.indexj = tsp_struct.indices(indexj(j));
		classifier.score = sorted(j);
		classifier.name1 = tsp_struct.probes(tsp_struct.indices(indexi(j)));
		classifier.name2 = tsp_struct.probes(tsp_struct.indices(indexj(j)));
	else
		classifier.indexi = indexi(j);
		classifier.indexj = indexj(j);
		classifier.score = sorted(j);
		classifier.name1 = tsp_struct.probes(indexi(j));
		classifier.name2 = tsp_struct.probes(indexj(j));
	end

	classifiers = [classifiers classifier];
end
	
% Get the top 2*k scoring pairs from the data
%[sorted, indexi, indexj] = nvdisjointpairmex(single(lower), 2*k, 0);