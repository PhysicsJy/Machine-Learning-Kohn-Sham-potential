cmake_minimum_required(VERSION 3.14)
enable_testing()

set(proj_name tddft CXX)
project(${proj_name})

# list(APPEND CMAKE_PREFIX_PATH /opt/intel/oneapi/mkl/2021.4.0)
list(APPEND CMAKE_PREFIX_PATH /usr/local/opt/openblas)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)

set(fftw3_INCLUDE_DIRS /usr/local/include)
set(fftw3_LIBRARIES /usr/local/lib/libfftw3.a)
set(OPENBLAS_INCLUDE /usr/local/opt/openblas/include)
set(APPLE_VEC_LIB_DIR /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/)

set(BLA_VENDOR OpenBLAS)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(NOT proj_path)
	set(proj_src_path ${PROJECT_SOURCE_DIR})
endif(NOT proj_path)

include_directories(${OPENBLAS_INCLUDE})

set(LIB_SOURCE_FILES  lib/common.h
                      lib/grid.h
                      lib/ground_state.h
                      lib/inversion.h
                      lib/hamiltonian.h
                      lib/td_math.h
                      lib/inversion.cpp)

add_library(tddftlib ${LIB_SOURCE_FILES} )

if (Boost_FOUND)
    target_include_directories(tddftlib
                               PUBLIC ${Boost_INCLUDE_DIRS}
                               PUBLIC ${OPENBLAS_INCLUDE}
                              )
endif()

# test
find_package(Boost COMPONENTS unit_test_framework )
if (Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS})

    set(TEST_SOURCE_FILES tests/test_math.cpp)

    add_executable(tddft_tests ${TEST_SOURCE_FILES} )
    target_include_directories(tddft_tests PUBLIC ${OPENBLAS_INCLUDE})
    target_link_libraries(tddft_tests PUBLIC tddftlib LAPACK::LAPACK BLAS::BLAS OpenMP::OpenMP_CXX ${Boost_LIBRARIES} ${fftw3_LIBRARIES} )     

    add_executable(tddft_omp_tests tests/test_omp.cpp )
    target_include_directories(tddft_omp_tests PUBLIC ${OPENBLAS_INCLUDE})
    target_link_libraries(tddft_omp_tests tddftlib LAPACK::LAPACK BLAS::BLAS OpenMP::OpenMP_CXX ${Boost_LIBRARIES} ${fftw3_LIBRARIES} ) 
endif ()

# executable
set(SRC_FILES src/main.cpp)

add_executable(tddft ${SRC_FILES})
target_include_directories(tddft PUBLIC ${OPENBLAS_INCLUDE})
target_link_libraries(tddft tddftlib)

if (Boost_FOUND)
    target_link_libraries(tddft ${Boost_LIBRARIES}  LAPACK::LAPACK BLAS::BLAS OpenMP::OpenMP_CXX ${fftw3_LIBRARIES} )
endif()
