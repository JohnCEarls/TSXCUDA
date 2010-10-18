function result = tsp_cuda(data, labels, probes, N, filter)
%TSP_CUDA Implementation of the TSP algorithm on the CUDA GPU architecture
%
%   [RESULT] = TSP_CUDA(DATA) assumes the class identifiers are in the first row
%   of the DATA matrix and removes it.  DATA is then split based on the class
%   labels and the TSP algorithm is performed.  Rows are assumed to be probes and 
%   columns experiments.  The header row must contain only zeroes and ones for the 
%   two class labels.  Cross validation is assumed to be LOOCV (leave one out).
%
%   [RESULT] = TSP_CUDA(DATA, LABELS) splits the data based on the class labels
%   and performs the TSP algorithm. No header row is assumed. Rows are assumed to 
%   be probes and columns experiments. If DATA is an MxN matrix, LABELS must be 
%   a 1xN vector containing only zeros and ones for the two class labels. 
%   Cross-validation is assumed to be LOOCV (leave 1 out).
%
%	[RESULT] = TSP_CUDA(DATA, LABELS, PROBES) defines the probe names for each row
%	of the matrix.  Probes must be a cell_array of strings.  If probes is absent or {}, 
%   then a default set of gene names are created
%
%   [RESULT] = TSP_CUDA(DATA, LABELS, PROBES, N) is the same as above, but
%   upper and lower bounds for LNOCV (leaving out N samples) are calculated.
%   
%   [RESULT] = TSP_CUDA(DATA, LABELS, PROBES, N, FILTER) is the same as above, but 
%   the genes are sorted for differential expression using the Wilcoxon rank sum test
%   and only the top FILTER genes are used for the TSP calculations.  
%   

%RESULT in both cases is a struct containing the following fields:
%
%   primary: 	primary TSP score
%   secondary: 	secondary TSP score for breaking ties
%   lower: 	lower bounds for cross-validation optimization algorithm
%   upper: 	upper bounds for cross-validation optimization algorithm
%   vote: 	which class this score votes for (0=class1, 1=class2)

if (nargin < 5)
	% No filtering for differential expression
	filter = 0;
end
if (nargin < 4)
	% Use LOOCV for lower and upper bounds optimization algorithm
	N = 1;
end
if (nargin < 3)
	probes = {};
end
if (nargin < 2)
	% Get the labels from the data matrix first row
	labels = [];
end
if (nargin < 1)
	error('Usage: [RESULT] = TSP_CUDA(DATA, LABELS)');
end

% If the label set is empty, get the first row of the data matrix
if (isempty(labels))
	labels = data(1,:);
	data(1,:) = [];	
% Check to make sure the number of labels is ok
else (length(labels) ~= size(data, 2))
	error('Number of class labels does not match number of cols of data');
end

% Now check to make sure the labels are only zeros and ones
if (length(unique(labels)) > 2)
	error('Class labels must be only 0 or 1')
elseif find(unique(labels) ~= [0 1])
	error('Class labels must be only 0 or 1')
end

% If the probe set is empty, create a default set of probe names
if (isempty(probes))
	probes = cell(size(data, 1), 1);
	for j=1:size(data,1)
		probes{j} = ['probe', int2str(j)];
	end
% Otherwise, check that the probe list is the correct size
else 
	if (length(probes) ~= size(data, 1))
		error('Number of probe names does not match number of rows of data');
	end
end

% Now we have ensured all the data is okay.  Lets impute any missing data
if ~isempty(find(isnan(data)))
	fprintf('Input matrix contains NaNs, imputing...\n');
	data = knnimpute(data);
end

% Calculate the ranks of the data
ranks = tiedrankmex(single(data));

% If asked to filter for differential expression, do so
if (filter > 0)
	% Yes, this is calculating differential expression based on ranks.  This is how
	% it is done in Lin et al 2009
	[unsorted, wilcox, indices] = ranksummex(ranks, single(labels));
	ranks = ranks(indices(1:filter), :);
else
	% Make sure this variable is zero if it is not positive
	filter = 0;
end

% % Finally we can run the TSP algorithm on the GPU
result = struct;
[result.primary, result.secondary, result.lower, result.upper, result.vote] = nvtspmex(ranks(:, labels==0), ranks(:, labels==1), N); 

% Add the labels and probes to the structure
result.labels = labels;
result.probes = probes;
result.cvn = N;
result.filter = filter;

% If we have filtered for differential expression, put in indices to original data matrix
% for each of the filtered genes
if (filter > 0)
	result.indices = indices(1:filter);
end