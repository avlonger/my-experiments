cmake_minimum_required(VERSION 2.8.4)
project(strlib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(kmp src/kmp.c src/algo/kmp.c)
add_executable(duval src/duval.c src/algo/duval.c)
add_executable(exp src/exp.c src/algo/duval.c)
add_executable(borderless src/borderless.c src/algo/borderless.c src/algo/kmp.c)
add_executable(border src/border.c src/algo/kmp.c)
add_executable(experiment src/experiment.c src/algo/kmp.c src/algo/duval.c src/algo/borderless.c)