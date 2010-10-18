%=================================================================
%
%  make.m
%  Author: Andrew Magis
%  Makefile for the MATLAB installation of tsp_cuda
%=================================================================

% Check the locations below to ensure they match your system configuration!

% This portion taken from nvmex.m from 2009 The MathWorks, Inc.
if ispc % Windows
    CUDA_LIB_Location = 'C:\CUDA\lib';
    Host_Compiler_Location = '-ccbin "C:\Program Files\Microsoft Visual Studio 9.0\VC\bin"';
    PIC_Option = '';
else % Mac and Linux (assuming gcc is on the path)
    CUDA_LIB_Location = '/usr/local/cuda/lib64/';
    Host_Compiler_Location = '';
    PIC_Option = ' --compiler-options -fPIC';
end
% End MathWorks code

% Build the c++ codes first
fprintf('Building the *.cpp files\n');
mex tiedrankmex.cpp
mex ranksummex.cpp

fprintf('Building the CUDA *.cu files\n')
fprintf('If this does not work, ensure the paths to the CUDA libraries \nand NVCC compiler are correct in make.m!\n\n\n');

nvmex('nvtspmex.cu', CUDA_LIB_Location, Host_Compiler_Location, PIC_Option);
nvmex('nvtstmex.cu', CUDA_LIB_Location, Host_Compiler_Location, PIC_Option);
nvmex('nvdisjointpairmex.cu', CUDA_LIB_Location, Host_Compiler_Location, PIC_Option);

fprintf('Deleting object files\n');
delete *.o
fprintf('Finished building tsp_cuda\n')
fprintf('Above MATLAB warnings about source files and 32-bit compatibility are typical\n');