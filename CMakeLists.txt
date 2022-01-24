cmake_minimum_required(VERSION 3.18)
project(mateval_cuda LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)
set(LIBDIR lib)

##########################################################################
# Library
##########################################################################

# Source files
file(GLOB SRCS "${SRCDIR}/*.cu")
file(GLOB HEADERS "${INCDIR}/mateval/*.hpp")

add_library(mateval_cuda STATIC
	${SRCS}
	${HEADERS}
	)

# Include directories
target_include_directories(mateval_cuda PUBLIC ${INCDIR})

# CUDA (2/2)
set_target_properties(mateval_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

# Directory
set(TESTSRCDIR tests)

# CU
foreach(test cuda cuda_throughput)
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

# CPP
foreach(test comparison util)
	set(test_out ${test}.test)
	add_executable(${test_out} ${TESTSRCDIR}/${test}.cpp)
	target_include_directories(${test_out} PRIVATE ${INCDIR})
	if (OpenMP_CXX_FOUND)
		target_link_libraries(${test_out} PRIVATE OpenMP::OpenMP_CXX)
	endif()
endforeach()
