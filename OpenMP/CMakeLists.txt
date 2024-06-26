cmake_minimum_required(VERSION 3.14)
project(hw2_1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(OpenMP REQUIRED)

# Group number
set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")

# Choose between serial.cpp and serial_naive.cpp
option(SERIAL_NAIVE "Use serial_naive.cpp instead of serial.cpp" OFF)


# Packaging system
if (NOT ${GROUP_NAME} STREQUAL None)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "cs267${GROUP_NAME}_hw2_1")
    if (SERIAL_NAIVE)
        install(FILES serial_naive.cpp openmp.cpp DESTINATION .)
    else()
        install(FILES serial.cpp openmp.cpp DESTINATION .)
    endif()
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif()

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()


if (SERIAL_NAIVE)
    add_executable(serial_naive main.cpp)
    target_sources(serial_naive PRIVATE serial_naive.cpp common.h)
else()
    add_executable(serial main.cpp)
    target_sources(serial PRIVATE serial.cpp common.h)
endif()

add_executable(openmp main.cpp openmp.cpp common.h)
target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)

# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
configure_file(job-openmp job-openmp COPYONLY)
