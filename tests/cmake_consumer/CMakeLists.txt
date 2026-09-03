cmake_minimum_required(VERSION 3.18)
project(FastSASAConsumer LANGUAGES C)

find_package(FastSASA CONFIG REQUIRED)

add_executable(fastsasa_consumer main.c)
target_link_libraries(fastsasa_consumer PRIVATE FastSASA::fastsasa_native)
