# - Try to find CPLEX
# Once done this will define
#  CPLEX_FOUND - System has Cplex
#  CPLEX_INCLUDE_DIRS - The Cplex include directories
#  CPLEX_LIBRARIES - The libraries needed to use Cplex

if (CPLEX_INCLUDE_DIR)
  # in cache already
  set(CPLEX_FOUND TRUE)
  set(CPLEX_INCLUDE_DIRS "${CPLEX_INCLUDE_DIR};${CPLEX_CONCERT_INCLUDE_DIR}" )
  set(CPLEX_LIBRARIES "${CPLEX_LIBRARY};${CPLEX_ILO_LIBRARY};${CPLEX_CONCERT_LIBRARY}" )
else (CPLEX_INCLUDE_DIR)

find_path(CPLEX_INCLUDE_DIR 
          NAMES ilcplex/cplex.h
          PATHS "$ENV{CPLEX_DIR}/cplex/include"
          )
          
find_path(CPLEX_CONCERT_INCLUDE_DIR 
          NAMES ilconcert/ilomodel.h
          PATHS "$ENV{CPLEX_DIR}/concert/include"
          )

find_library( CPLEX_LIBRARY 
              cplex
              PATHS "$ENV{CPLEX_DIR}/cplex/lib/x86-64_linux/static_pic" 
              )

find_library( CPLEX_ILO_LIBRARY 
              ilocplex
              PATHS "$ENV{CPLEX_DIR}/cplex/lib/x86-64_linux/static_pic" 
              )

find_library( CPLEX_CONCERT_LIBRARY 
              concert
              PATHS "$ENV{CPLEX_DIR}/concert/lib/x86-64_linux/static_pic" 
              )     
                 

set(CPLEX_INCLUDE_DIRS "${CPLEX_INCLUDE_DIR};${CPLEX_CONCERT_INCLUDE_DIR}" )
set(CPLEX_LIBRARIES "${CPLEX_LIBRARY};${CPLEX_ILO_LIBRARY};${CPLEX_CONCERT_LIBRARY}" )

# use c++ headers as default
set(CPLEX_COMPILER_FLAGS "-DIL_STD" CACHE STRING "Cplex Compiler Flags")
mark_as_advanced(CPLEX_INCLUDE_DIR CPLEX_LIBRARY )

endif(CPLEX_INCLUDE_DIR)
