cmake_minimum_required(VERSION 3.18)
project(mateval_cuda LANGUAGES CXX CUDA)

set(BUILD_TEST OFF)

find_package(CUDAToolkit 10.0 REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Directories
set(INCDIR include)
set(SRCDIR src)
set(LIBDIR lib)

##########################################################################
# Library
##########################################################################

# Source files
file(GLOB HEADERS "${INCDIR}/mateval/*.hpp")

## CUDA
add_library(mateval_cuda STATIC
	src/comparison_cuda.cu
	src/minmax_cuda.cu
	src/norm_cuda.cu
	${HEADERS}
	)

# Include directories
target_include_directories(mateval_cuda PUBLIC ${INCDIR})
#target_link_libraries(mateval_cuda PRIVATE
#	CUDA::cublas
#	CUDA::cusolver
#	CUDA::curand
#	)

# CUDA (2/2)
set_target_properties(mateval_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mateval_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

## CUDA LATMS
add_library(mateval_cuda_latms STATIC
	src/latms_cuda.cu
	${HEADERS}
	)

# Include directories
target_include_directories(mateval_cuda_latms PUBLIC ${INCDIR})
target_link_libraries(mateval_cuda_latms PRIVATE
	CUDA::cublas
	CUDA::cusolver
	CUDA::curand
	)

# CUDA (2/2)
set_target_properties(mateval_cuda_latms PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mateval_cuda_latms PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Output
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIBDIR})

##########################################################################
# Installing
##########################################################################
install(TARGETS mateval_cuda
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/mateval
	)
install(FILES ${HEADERS}
        DESTINATION include/mateval)

##########################################################################
# Test
##########################################################################
if (${BUILD_TEST})
	find_package(OpenMP)
	find_package(LAPACK)
	# Directory
	set(TESTSRCDIR tests)

	# CU
	foreach(test cuda cuda_throughput cuda_operation cuda_norm)
		set(test_out ${test}.test)
		add_executable(${test_out} ${TESTSRCDIR}/${test}.cu)
		target_include_directories(${test_out} PRIVATE ${INCDIR})
		target_link_libraries(${test_out} PRIVATE mateval_cuda)
		target_link_libraries(${test_out} PRIVATE
			CUDA::cublas
			)
		if (OpenMP_CXX_FOUND)
			target_link_libraries(${test_out} PRIVATE OpenMP::OpenMP_CXX)
		endif()
	endforeach()

	if(LAPACK_FOUND)
		foreach(test cuda_latms)
			set(test_out ${test}.test)
			add_executable(${test_out} ${TESTSRCDIR}/${test}.cu)
			target_include_directories(${test_out} PRIVATE ${INCDIR})
			target_link_libraries(${test_out} PRIVATE mateval_cuda_latms)
			target_link_libraries(${test_out} PRIVATE
				CUDA::cublas
				LAPACK::LAPACK
				)
			if (OpenMP_CXX_FOUND)
				target_link_libraries(${test_out} PRIVATE OpenMP::OpenMP_CXX)
			endif()
		endforeach()
	endif()

	# CPP
	foreach(test comparison util norm)
		set(test_out ${test}.test)
		add_executable(${test_out} ${TESTSRCDIR}/${test}.cpp)
		target_include_directories(${test_out} PRIVATE ${INCDIR})
		if (OpenMP_CXX_FOUND)
			target_link_libraries(${test_out} PRIVATE OpenMP::OpenMP_CXX)
		endif()
	endforeach()
endif()
