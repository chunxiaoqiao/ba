# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/q/CLionProjects/ba

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/q/CLionProjects/ba/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/ba.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ba.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ba.dir/flags.make

CMakeFiles/ba.dir/main_ba.cpp.o: CMakeFiles/ba.dir/flags.make
CMakeFiles/ba.dir/main_ba.cpp.o: ../main_ba.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/q/CLionProjects/ba/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ba.dir/main_ba.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ba.dir/main_ba.cpp.o -c /home/q/CLionProjects/ba/main_ba.cpp

CMakeFiles/ba.dir/main_ba.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ba.dir/main_ba.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/q/CLionProjects/ba/main_ba.cpp > CMakeFiles/ba.dir/main_ba.cpp.i

CMakeFiles/ba.dir/main_ba.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ba.dir/main_ba.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/q/CLionProjects/ba/main_ba.cpp -o CMakeFiles/ba.dir/main_ba.cpp.s

# Object files for target ba
ba_OBJECTS = \
"CMakeFiles/ba.dir/main_ba.cpp.o"

# External object files for target ba
ba_EXTERNAL_OBJECTS =

ba: CMakeFiles/ba.dir/main_ba.cpp.o
ba: CMakeFiles/ba.dir/build.make
ba: /usr/local/lib/libopencv_dnn.so.3.4.3
ba: /usr/local/lib/libopencv_ml.so.3.4.3
ba: /usr/local/lib/libopencv_objdetect.so.3.4.3
ba: /usr/local/lib/libopencv_shape.so.3.4.3
ba: /usr/local/lib/libopencv_stitching.so.3.4.3
ba: /usr/local/lib/libopencv_superres.so.3.4.3
ba: /usr/local/lib/libopencv_videostab.so.3.4.3
ba: /usr/local/lib/libopencv_calib3d.so.3.4.3
ba: /usr/local/lib/libopencv_features2d.so.3.4.3
ba: /usr/local/lib/libopencv_flann.so.3.4.3
ba: /usr/local/lib/libopencv_highgui.so.3.4.3
ba: /usr/local/lib/libopencv_photo.so.3.4.3
ba: /usr/local/lib/libopencv_video.so.3.4.3
ba: /usr/local/lib/libopencv_videoio.so.3.4.3
ba: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
ba: /usr/local/lib/libopencv_imgproc.so.3.4.3
ba: /usr/local/lib/libopencv_core.so.3.4.3
ba: CMakeFiles/ba.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/q/CLionProjects/ba/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ba"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ba.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ba.dir/build: ba

.PHONY : CMakeFiles/ba.dir/build

CMakeFiles/ba.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ba.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ba.dir/clean

CMakeFiles/ba.dir/depend:
	cd /home/q/CLionProjects/ba/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/q/CLionProjects/ba /home/q/CLionProjects/ba /home/q/CLionProjects/ba/cmake-build-debug /home/q/CLionProjects/ba/cmake-build-debug /home/q/CLionProjects/ba/cmake-build-debug/CMakeFiles/ba.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ba.dir/depend

