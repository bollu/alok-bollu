# CMake 3.5.1+ is recommended
cmake_minimum_required(VERSION 3.5.1)
# C++ 11 is required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(word2vec word2vec.cpp)

find_package(ArrayFire REQUIRED)
find_package(flashlight REQUIRED)

target_link_libraries(
  word2vec
  PRIVATE
   # the correct ArrayFire backend is transitively included by flashlight
  flashlight::flashlight
)
