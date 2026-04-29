#pragma once
#include "root.hpp"

#ifdef CUDA_ON
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include <thrust/async/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#endif