julia:julia.cu CComplex.h
	nvcc julia.cu -o julia --std=c++11