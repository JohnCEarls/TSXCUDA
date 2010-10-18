function nvmex(cuFileName, CUDA_LIB_Location, Host_Compiler_Location, PIC_Option)
%NVMEX Compiles and links a CUDA file for MATLAB usage
%   NVMEX(FILENAME) will create a MEX-File (also with the name FILENAME) by
%   invoking the CUDA compiler, nvcc, and then linking with the MEX
%   function in MATLAB.

% This portion taken from nvmex.m from 2009 The MathWorks, Inc.
[path, filename, ext] = fileparts(cuFileName);

nvccCommandLine = [ ...
    '/usr/local/cuda/bin/nvcc --compile ' cuFileName ' ' Host_Compiler_Location ' ' ...
    ' -o ' filename '.o ' ...
    PIC_Option ...
    ' -I' matlabroot '/extern/include ' ...
    ];
	
mexCommandLine = ['mex (''' filename '.o'', ''-L' CUDA_LIB_Location ''', ''-lcudart'')'];

disp(nvccCommandLine);
status = system(nvccCommandLine);
if status < 0
    error 'Error invoking nvcc';
end

disp(mexCommandLine);
eval(mexCommandLine);

end
