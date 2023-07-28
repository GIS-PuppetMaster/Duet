cmake_minimum_required(VERSION 3.22)
project(src)

set(CMAKE_CXX_STANDARD 14)

include_directories(C:\\Users\\admin\\anaconda3\\Lib\\site-packages\\torch\\include)
include_directories(C:\\Users\\admin\\anaconda3\\Lib\\site-packages\\torch\\include\\torch\\csrc\\api\\include)
include_directories(C:\\Users\\admin\\anaconda3\\include)


add_executable(src
        MySampler.cpp
        MySampler.h
        MySampler_rawsample.cpp
        MySampler_rawsample.h)
