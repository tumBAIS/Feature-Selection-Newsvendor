cmake_minimum_required(VERSION 3.16)
project(newsvendor_features)
set(CMAKE_CXX_STANDARD 14)

# set CPLEX variables for FindCPLEX
if(MSVC)
    # requires CPLEX_ROOT to be set to the root directory of the CPLEX studio installation
    message(STATUS "CPLEX_ROOT is set to $ENV{CPLEX_ROOT}")
    set(CPLEX_CONCERT_INCLUDE_DIR "$ENV{CPLEX_ROOT}/concert/include")
    set(CPLEX_CONCERT_LIBRARY "$ENV{CPLEX_ROOT}/concert/lib/x64_windows_msvc14/stat_mda/concert.lib")
    set(CPLEX_ILO_LIBRARY "$ENV{CPLEX_ROOT}/cplex/lib/x64_windows_msvc14/stat_mda/ilocplex.lib")
    set(CPLEX_INCLUDE_DIR "$ENV{CPLEX_ROOT}/cplex/include")
    set(CPLEX_LIBRARY "$ENV{CPLEX_ROOT}/cplex/lib/x64_windows_msvc14/stat_mda/cplex2211.lib")
else()

endif()

# Set CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

# Include FindCPLEX.cmake
include(FindCPLEX.cmake)

# Set compiler flags
if(MSVC)
    # For MSVC, add any specific flags if necessary
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++14")
endif()

# Set source and include files
file(GLOB source_files "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB include_files
        "${PROJECT_SOURCE_DIR}/src/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        )

include_directories(include)

# Find CPLEX package
find_package(CPLEX REQUIRED cplex ilocplex concert)

# Include CPLEX directories
include_directories(${CPLEX_INCLUDE_DIRS})

# Add executable
add_executable(newsvendor_features ${source_files} ${include_files})

# Link libraries
target_link_libraries(
        newsvendor_features
        ${CPLEX_LIBRARY}
        ${CPLEX_CONCERT_LIBRARY}
        ${CPLEX_ILO_LIBRARY}
        ${CPLEX_LIBRARIES} -ldl
)

# Set compile definitions
set_property(TARGET newsvendor_features PROPERTY COMPILE_DEFINITIONS IL_STD)
