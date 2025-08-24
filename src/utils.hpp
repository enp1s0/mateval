#include <stdexcept>
#include <sstream>
#include <cuda_device_runtime_api.h>
#include <cuda.h>

#define CUDA_CHECK_ERROR(status) cuda_error_check(status, __FILE__, __LINE__, __func__)

inline void cuda_error_check(const cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
  if(error != cudaSuccess){
    std::stringstream ss;
    ss << cudaGetErrorString(error);
    ss << " [" << filename << ":" << line << " in " << funcname << "]";
    throw std::runtime_error(ss.str());
  }
}
