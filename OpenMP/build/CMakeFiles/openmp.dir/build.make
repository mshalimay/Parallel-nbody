# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mashalimay/HPPC/hw2/hw2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mashalimay/HPPC/hw2/hw2/build

# Include any dependencies generated for this target.
include CMakeFiles/openmp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/openmp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/openmp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/openmp.dir/flags.make

CMakeFiles/openmp.dir/main.cpp.o: CMakeFiles/openmp.dir/flags.make
CMakeFiles/openmp.dir/main.cpp.o: ../main.cpp
CMakeFiles/openmp.dir/main.cpp.o: CMakeFiles/openmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mashalimay/HPPC/hw2/hw2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/openmp.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/openmp.dir/main.cpp.o -MF CMakeFiles/openmp.dir/main.cpp.o.d -o CMakeFiles/openmp.dir/main.cpp.o -c /home/mashalimay/HPPC/hw2/hw2/main.cpp

CMakeFiles/openmp.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openmp.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mashalimay/HPPC/hw2/hw2/main.cpp > CMakeFiles/openmp.dir/main.cpp.i

CMakeFiles/openmp.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openmp.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mashalimay/HPPC/hw2/hw2/main.cpp -o CMakeFiles/openmp.dir/main.cpp.s

CMakeFiles/openmp.dir/openmp.cpp.o: CMakeFiles/openmp.dir/flags.make
CMakeFiles/openmp.dir/openmp.cpp.o: ../openmp.cpp
CMakeFiles/openmp.dir/openmp.cpp.o: CMakeFiles/openmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mashalimay/HPPC/hw2/hw2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/openmp.dir/openmp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/openmp.dir/openmp.cpp.o -MF CMakeFiles/openmp.dir/openmp.cpp.o.d -o CMakeFiles/openmp.dir/openmp.cpp.o -c /home/mashalimay/HPPC/hw2/hw2/openmp.cpp

CMakeFiles/openmp.dir/openmp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openmp.dir/openmp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mashalimay/HPPC/hw2/hw2/openmp.cpp > CMakeFiles/openmp.dir/openmp.cpp.i

CMakeFiles/openmp.dir/openmp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openmp.dir/openmp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mashalimay/HPPC/hw2/hw2/openmp.cpp -o CMakeFiles/openmp.dir/openmp.cpp.s

# Object files for target openmp
openmp_OBJECTS = \
"CMakeFiles/openmp.dir/main.cpp.o" \
"CMakeFiles/openmp.dir/openmp.cpp.o"

# External object files for target openmp
openmp_EXTERNAL_OBJECTS =

openmp: CMakeFiles/openmp.dir/main.cpp.o
openmp: CMakeFiles/openmp.dir/openmp.cpp.o
openmp: CMakeFiles/openmp.dir/build.make
openmp: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
openmp: /usr/lib/x86_64-linux-gnu/libpthread.a
openmp: CMakeFiles/openmp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mashalimay/HPPC/hw2/hw2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable openmp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/openmp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/openmp.dir/build: openmp
.PHONY : CMakeFiles/openmp.dir/build

CMakeFiles/openmp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/openmp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/openmp.dir/clean

CMakeFiles/openmp.dir/depend:
	cd /home/mashalimay/HPPC/hw2/hw2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mashalimay/HPPC/hw2/hw2 /home/mashalimay/HPPC/hw2/hw2 /home/mashalimay/HPPC/hw2/hw2/build /home/mashalimay/HPPC/hw2/hw2/build /home/mashalimay/HPPC/hw2/hw2/build/CMakeFiles/openmp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/openmp.dir/depend
