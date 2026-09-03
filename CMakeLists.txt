cmake_minimum_required(VERSION 3.18)

project(FastSASA LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FASTSASA_VERSION "0.1.0rc19" CACHE STRING "FastSASA release version")

include(GNUInstallDirs)
find_package(Threads REQUIRED)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FASTSASA_ENABLE_CUDA "Build the NVIDIA CUDA backend" ON)
option(FASTSASA_ENABLE_VULKAN "Build the Vulkan fallback backend when its build tools are available" ON)
option(FASTSASA_BUILD_STATIC "Build the static FastSASA library" ON)
option(FASTSASA_BUILD_CLI "Build the FastSASA command-line tool" ${FASTSASA_BUILD_STATIC})
option(FASTSASA_ENABLE_WARNINGS "Build FastSASA C/C++ sources with common compiler warnings" ON)
option(FASTSASA_ENABLE_SANITIZERS "Build FastSASA C/C++ sources with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(FASTSASA_NATIVE_CPU_OPTIMIZE "Build CPU code with -march=native for this machine" OFF)

function(fastsasa_apply_developer_options target_name)
    if(FASTSASA_ENABLE_WARNINGS)
        if(MSVC)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/W3>)
        else()
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra>
            )
        endif()
    endif()
    if(MSVC)
        # The reference arithmetic must not be contracted or reassociated;
        # /fp:strict is MSVC's equivalent of -ffp-contract=off.
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/fp:strict>)
        target_compile_definitions(${target_name} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    endif()
    # The CPU reference and the bit-exact FP64 GPU paths must round every
    # operation identically; contraction into FMA (the default on
    # FMA-capable targets such as -march=native builds) would change results.
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=off>
        )
    endif()
    if(FASTSASA_ENABLE_SANITIZERS)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address,undefined -fno-omit-frame-pointer>
            )
            target_link_options(${target_name}
                PRIVATE
                    -fsanitize=address,undefined
            )
        endif()
    endif()
endfunction()

function(fastsasa_apply_cpu_simd_options target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-fopenmp-simd>
        )
    endif()
    if(FASTSASA_NATIVE_CPU_OPTIMIZE)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:C,CXX>:-march=native>
            )
        endif()
    endif()
endfunction()

if(FASTSASA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA compiler not found; building without the CUDA backend")
        set(FASTSASA_ENABLE_CUDA OFF)
    endif()
endif()

if(FASTSASA_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(FASTSASA_USE_DEFAULT_CUDA_ARCHITECTURES ON)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING
            "CUDA architectures compiled into FastSASA; override for a narrower deployment target")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    if(FASTSASA_USE_DEFAULT_CUDA_ARCHITECTURES)
        execute_process(
            COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
            OUTPUT_VARIABLE FASTSASA_NVCC_GPU_ARCHES
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        foreach(FASTSASA_ARCH 100 110 120 121)
            if(FASTSASA_NVCC_GPU_ARCHES MATCHES "(^|[ \n\r\t])compute_${FASTSASA_ARCH}([ \n\r\t]|$)")
                list(APPEND CMAKE_CUDA_ARCHITECTURES ${FASTSASA_ARCH})
            endif()
        endforeach()
        list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING
            "CUDA architectures compiled into FastSASA; override for a narrower deployment target" FORCE)
    endif()
    set(FASTSASA_DEVICE_SOURCE src/sr_cuda.cu)
else()
    set(FASTSASA_DEVICE_SOURCE src/fastsasa_device_stub.c)
endif()

set(FASTSASA_VULKAN_SOURCE)
set(FASTSASA_HAVE_VULKAN OFF)
if(FASTSASA_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    find_program(FASTSASA_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(Vulkan_FOUND AND FASTSASA_GLSLC)
        set(FASTSASA_HAVE_VULKAN ON)
        set(FASTSASA_VULKAN_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan)

        function(fastsasa_compile_vulkan_shader source output symbol)
            add_custom_command(
                OUTPUT ${output}.spv ${output}.h
                COMMAND ${CMAKE_COMMAND} -E make_directory ${FASTSASA_VULKAN_GENERATED_DIR}
                COMMAND ${FASTSASA_GLSLC} -O ${ARGN} ${source} -o ${output}.spv
                COMMAND ${CMAKE_COMMAND} -DINPUT=${output}.spv -DOUTPUT=${output}.h -DSYMBOL=${symbol}
                        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake
                DEPENDS ${source} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake
                VERBATIM
            )
        endfunction()

        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/cell_count.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_count_spv
            fastsasa_vk_cell_count_spv)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/cell_scan.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_scan_spv
            fastsasa_vk_cell_scan_spv)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/cell_fill.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_fill_spv
            fastsasa_vk_cell_fill_spv)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/cell_gather.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_gather_spv
            fastsasa_vk_cell_gather_spv)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/sr_cell_list.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_32_spv
            fastsasa_vk_sr_cell_list_32_spv -DSR_LOCAL_SIZE=32)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/sr_cell_list.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_64_spv
            fastsasa_vk_sr_cell_list_64_spv -DSR_LOCAL_SIZE=64)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/sr_cell_list.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_sg_spv
            fastsasa_vk_sr_cell_list_sg_spv -DSR_SUBGROUP_ATOMS=1 --target-env=vulkan1.1)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/lee_richards_cell.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_cell_spv
            fastsasa_vk_lee_richards_cell_spv)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/lee_richards_reduce.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_reduce_spv
            fastsasa_vk_lee_richards_reduce_spv)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/cell_count.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_count_fp64_spv
            fastsasa_vk_cell_count_fp64_spv -DFASTSASA_FP64=1)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/cell_fill.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_fill_fp64_spv
            fastsasa_vk_cell_fill_fp64_spv -DFASTSASA_FP64=1)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/sr_cell_list.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_fp64_32_spv
            fastsasa_vk_sr_cell_list_fp64_32_spv -DFASTSASA_FP64=1 -DSR_LOCAL_SIZE=32)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/sr_cell_list.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_fp64_64_spv
            fastsasa_vk_sr_cell_list_fp64_64_spv -DFASTSASA_FP64=1 -DSR_LOCAL_SIZE=64)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/sr_cell_list.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_fp64_sg_spv
            fastsasa_vk_sr_cell_list_fp64_sg_spv -DFASTSASA_FP64=1 -DSR_SUBGROUP_ATOMS=1 --target-env=vulkan1.1)
        # Surface-point export (VMD -visualize track): FP64 only, since the
        # trajectory CLI's surface-export path always uses double geometry;
        # no subgroup variant, since there is no reduction step left to make
        # subgroup-cooperative.
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/sr_exposed_points_cell_list.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_exposed_points_fp64_64_spv
            fastsasa_vk_sr_exposed_points_fp64_64_spv -DFASTSASA_FP64=1 -DSR_LOCAL_SIZE=64)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/lee_richards_cell.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_cell_fp64_spv
            fastsasa_vk_lee_richards_cell_fp64_spv -DFASTSASA_FP64=1)
        fastsasa_compile_vulkan_shader(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/lee_richards_reduce.comp
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_reduce_fp64_spv
            fastsasa_vk_lee_richards_reduce_fp64_spv -DFASTSASA_FP64=1)

        set(FASTSASA_VULKAN_HEADERS
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_count_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_scan_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_fill_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_gather_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_32_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_64_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_sg_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_cell_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_reduce_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_count_fp64_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_cell_fill_fp64_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_fp64_32_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_fp64_64_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_cell_list_fp64_sg_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_cell_fp64_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_lee_richards_reduce_fp64_spv.h
            ${FASTSASA_VULKAN_GENERATED_DIR}/fastsasa_vk_sr_exposed_points_fp64_64_spv.h)
        add_custom_target(fastsasa_vulkan_shaders DEPENDS ${FASTSASA_VULKAN_HEADERS})
        set(FASTSASA_VULKAN_SOURCE
            src/fastsasa_vulkan.cc
            third-party/volk/volk.c)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            set_source_files_properties(src/fastsasa_vulkan.cc PROPERTIES
                COMPILE_OPTIONS -Wno-missing-field-initializers)
        endif()
        message(STATUS "Vulkan fallback backend enabled")
    else()
        if(NOT Vulkan_FOUND)
            message(STATUS "Vulkan fallback backend disabled: the Vulkan loader/headers were not found (install the Vulkan SDK or libvulkan-dev plus vulkan-headers)")
        elseif(NOT FASTSASA_GLSLC)
            message(STATUS "Vulkan fallback backend disabled: glslc was not found (install shaderc or the Vulkan SDK)")
        else()
            message(STATUS "Vulkan fallback backend disabled")
        endif()
    endif()
endif()

file(GLOB FASTSASA_MOLFILE_INCLUDE_HINTS "/usr/local/lib/*/plugins/include")
find_path(FASTSASA_MOLFILE_INCLUDE_DIR
    NAMES molfile_plugin.h
    HINTS ${FASTSASA_MOLFILE_INCLUDE_HINTS}
)

if(FASTSASA_BUILD_STATIC)
    add_library(fastsasa STATIC
        src/fastsasa_api.c
        src/fastsasa_cpu.cc
        src/fastsasa_radius.c
        src/fastsasa_trajectory.c
        src/dcd_reader.c
        src/gemmi_topology.cc
        ${FASTSASA_DEVICE_SOURCE}
        ${FASTSASA_VULKAN_SOURCE}
        src/xtc_reader.c
    )
endif()

add_library(fastsasa_native SHARED
    src/fastsasa_api.c
    src/fastsasa_cpu.cc
    src/fastsasa_radius.c
    src/fastsasa_trajectory.c
    src/gemmi_topology.cc
    ${FASTSASA_DEVICE_SOURCE}
    ${FASTSASA_VULKAN_SOURCE}
)

if(FASTSASA_BUILD_STATIC)
    target_include_directories(fastsasa
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/gemmi/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            $<$<BOOL:${FASTSASA_HAVE_VULKAN}>:${FASTSASA_VULKAN_GENERATED_DIR}>
    )
endif()

target_include_directories(fastsasa_native
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/gemmi/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        $<$<BOOL:${FASTSASA_HAVE_VULKAN}>:${FASTSASA_VULKAN_GENERATED_DIR}>
)

if(FASTSASA_BUILD_STATIC)
    if(FASTSASA_ENABLE_CUDA OR FASTSASA_HAVE_VULKAN)
        target_compile_features(fastsasa PUBLIC cxx_std_17)
    else()
        target_compile_features(fastsasa PUBLIC cxx_std_11)
    endif()
endif()
if(FASTSASA_ENABLE_CUDA OR FASTSASA_HAVE_VULKAN)
    target_compile_features(fastsasa_native PUBLIC cxx_std_17)
else()
    target_compile_features(fastsasa_native PUBLIC cxx_std_11)
endif()

if(FASTSASA_ENABLE_CUDA)
    if(FASTSASA_BUILD_STATIC)
        set_target_properties(fastsasa PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
    endif()
    set_target_properties(fastsasa_native PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(FASTSASA_BUILD_STATIC)
    target_compile_definitions(fastsasa
        PRIVATE
            PACKAGE_STRING="FastSASA"
            FASTSASA_INSTALL_DEFAULT_CONFIG="${CMAKE_INSTALL_FULL_DATADIR}/fastsasa/protor.config"
    )
endif()

target_compile_definitions(fastsasa_native
    PRIVATE
        FASTSASA_INSTALL_DEFAULT_CONFIG="${CMAKE_INSTALL_FULL_DATADIR}/fastsasa/protor.config"
)

if(FASTSASA_HAVE_VULKAN)
    # The Vulkan loader is dlopen'd at runtime through the vendored volk
    # loader, so binaries carry no libvulkan.so.1 link-time dependency and
    # still load on systems without a Vulkan installation.
    if(FASTSASA_BUILD_STATIC)
        add_dependencies(fastsasa fastsasa_vulkan_shaders)
        target_compile_definitions(fastsasa PRIVATE FASTSASA_HAVE_VULKAN=1
                                   VK_NO_PROTOTYPES=1)
        target_include_directories(fastsasa PRIVATE
            ${Vulkan_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/third-party/volk)
    endif()
    add_dependencies(fastsasa_native fastsasa_vulkan_shaders)
    target_compile_definitions(fastsasa_native PRIVATE FASTSASA_HAVE_VULKAN=1
                               VK_NO_PROTOTYPES=1)
    target_include_directories(fastsasa_native PRIVATE
        ${Vulkan_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/third-party/volk)
endif()

if(FASTSASA_BUILD_STATIC)
    target_link_libraries(fastsasa
        PUBLIC
            $<$<NOT:$<PLATFORM_ID:Windows>>:stdc++>
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
endif()

target_link_libraries(fastsasa_native
    PUBLIC
        $<$<NOT:$<PLATFORM_ID:Windows>>:stdc++>
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

if(FASTSASA_BUILD_STATIC)
    fastsasa_apply_cpu_simd_options(fastsasa)
endif()
fastsasa_apply_cpu_simd_options(fastsasa_native)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(fastsasa_native
        PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/fastsasa_native.map"
    )
elseif(WIN32)
    # No version script on Windows: export every public (non-static)
    # symbol so the ctypes bridge can resolve the fastsasa_* API from the DLL.
    set_target_properties(fastsasa_native PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(FASTSASA_BUILD_STATIC AND FASTSASA_MOLFILE_INCLUDE_DIR)
    target_include_directories(fastsasa
        PRIVATE
            ${FASTSASA_MOLFILE_INCLUDE_DIR}
    )
    target_compile_definitions(fastsasa
        PRIVATE
            FASTSASA_HAVE_MOLFILE=1
    )
endif()

if(FASTSASA_BUILD_STATIC AND FASTSASA_ENABLE_CUDA)
    set_target_properties(fastsasa PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(FASTSASA_BUILD_STATIC)
    set_target_properties(fastsasa PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME fastsasa
        OUTPUT_NAME fastsasa
    )
    if(WIN32)
        # The CUDA-enabled fastsasa.exe emits an import library named
        # fastsasa.lib, which would collide with the static archive.
        set_target_properties(fastsasa PROPERTIES ARCHIVE_OUTPUT_NAME fastsasa_static)
    endif()
endif()

if(FASTSASA_ENABLE_CUDA)
    set_target_properties(fastsasa_native PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

set_target_properties(fastsasa_native PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME fastsasa_native
    OUTPUT_NAME fastsasa_native
)

set(FASTSASA_INSTALL_TARGETS fastsasa_native)
if(FASTSASA_BUILD_STATIC)
    list(APPEND FASTSASA_INSTALL_TARGETS fastsasa)
endif()

install(TARGETS ${FASTSASA_INSTALL_TARGETS}
    EXPORT FastSASATargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES
        include/fastsasa.h
        include/fastsasa_cpu.h
        include/fastsasa_radius.h
        include/fastsasa_trajectory.h
        include/fastsasa_topology.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT FastSASATargets
    FILE FastSASATargets.cmake
    NAMESPACE FastSASA::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FastSASA
)
configure_file(
    cmake/FastSASAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FastSASAConfig.cmake
    @ONLY
)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/FastSASAConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FastSASA
)
install(FILES
        python/fastsasa.py
        python/fastsasa_adapters.py
        python/fastsasa_features.py
        python/fastsasa_native.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fastsasa/python
)
install(FILES
        share/protor.config
        share/protor_glycans.config
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fastsasa
)
install(FILES
        LICENSE
        NOTICE
        licenses/Gemmi-MPL-2.0.txt
        licenses/Gemmi-third-party-notices.txt
        licenses/TAO-PEGTL-MIT.txt
        licenses/FreeSASA-radii-table-MIT.txt
        licenses/volk-MIT.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fastsasa/licenses
)
install(DIRECTORY examples/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fastsasa/examples
    PATTERN "__pycache__" EXCLUDE
)

option(FASTSASA_BUILD_NATIVE_TESTS "Build native FastSASA validation tests" OFF)

if(FASTSASA_BUILD_NATIVE_TESTS)
    enable_testing()
endif()

if(FASTSASA_BUILD_CLI AND NOT FASTSASA_BUILD_STATIC)
    message(FATAL_ERROR "The CLI requires FASTSASA_BUILD_STATIC=ON")
endif()

if(FASTSASA_BUILD_NATIVE_TESTS AND NOT FASTSASA_BUILD_CLI)
    message(FATAL_ERROR "Native validation tests require FASTSASA_BUILD_CLI=ON")
endif()

if(FASTSASA_BUILD_CLI)
    add_executable(fastsasa_cli
        tools/fastsasa_cli.cc
        tools/fastsasa_trajectory_cli.c
    )
    target_include_directories(fastsasa_cli
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(fastsasa_cli
        PRIVATE
            fastsasa
            Threads::Threads
            $<$<NOT:$<PLATFORM_ID:Windows>>:m>
    )
    target_compile_definitions(fastsasa_cli
        PRIVATE
            FASTSASA_VERSION="${FASTSASA_VERSION}"
    )
    set_target_properties(fastsasa_cli PROPERTIES
        OUTPUT_NAME fastsasa
    )
endif()

if(FASTSASA_BUILD_NATIVE_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    if(FASTSASA_ENABLE_CUDA)
        add_executable(fastsasa_cuda_boundary_validate
            tests/cuda_boundary_validation.c
        )
        target_link_libraries(fastsasa_cuda_boundary_validate
            PRIVATE
                fastsasa
                $<$<NOT:$<PLATFORM_ID:Windows>>:m>
        )
        add_executable(fastsasa_selected_center_validate
            tests/selected_center_validation.c
        )
        target_link_libraries(fastsasa_selected_center_validate
            PRIVATE
                fastsasa
                $<$<NOT:$<PLATFORM_ID:Windows>>:m>
        )
        add_executable(fastsasa_context_isolation_validate
            tests/context_isolation_validation.c
        )
        target_link_libraries(fastsasa_context_isolation_validate
            PRIVATE
                fastsasa
                $<$<NOT:$<PLATFORM_ID:Windows>>:m>
        )
        add_executable(fastsasa_autotune_dispatch_validate
            tests/autotune_dispatch_validation.c
        )
        target_link_libraries(fastsasa_autotune_dispatch_validate
            PRIVATE
                fastsasa
                $<$<NOT:$<PLATFORM_ID:Windows>>:m>
        )
    endif()
    if(FASTSASA_HAVE_VULKAN)
        add_executable(fastsasa_vulkan_batch_validate
            tests/vulkan_batch_validation.c
        )
        target_link_libraries(fastsasa_vulkan_batch_validate
            PRIVATE
                fastsasa
                $<$<NOT:$<PLATFORM_ID:Windows>>:m>
        )
    endif()
    add_executable(fastsasa_mmcif_selection_validate
        tests/mmcif_selection_validation.c
    )
    target_link_libraries(fastsasa_mmcif_selection_validate
        PRIVATE
            fastsasa_native
            $<$<NOT:$<PLATFORM_ID:Windows>>:m>
    )
    add_executable(fastsasa_mmcif_corpus_validate
        tests/mmcif_corpus_validation.cc
    )
    target_link_libraries(fastsasa_mmcif_corpus_validate
        PRIVATE
            fastsasa_native
            $<$<NOT:$<PLATFORM_ID:Windows>>:m>
    )
    file(GLOB FASTSASA_EXTERNAL_MMCIF_FIXTURES
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/external_mmcif/*.cif"
    )
    if(FASTSASA_EXTERNAL_MMCIF_FIXTURES)
        add_custom_target(fastsasa_external_mmcif_validate
            COMMAND $<TARGET_FILE:fastsasa_mmcif_corpus_validate> ${FASTSASA_EXTERNAL_MMCIF_FIXTURES}
            DEPENDS fastsasa_mmcif_corpus_validate
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Validating external mmCIF fixtures"
        )
    endif()
    add_custom_target(fastsasa_cli_validate
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_validation.py
                --fastsasa $<TARGET_FILE:fastsasa_cli>
                --data-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
        DEPENDS fastsasa_cli
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Validating FastSASA CLI GPU, CPU, and threaded CPU paths"
    )
    if(FASTSASA_ENABLE_CUDA)
        add_custom_target(fastsasa_python_feature_validate
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_feature_validation.py
                    --library $<TARGET_FILE:fastsasa_native>
                    --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
            DEPENDS fastsasa_native
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Validating FastSASA Python feature extraction"
        )
    endif()
    add_custom_target(fastsasa_python_feature_unit
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_feature_unit.py
                --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
        COMMENT "Validating FastSASA Python feature statistics without CUDA"
    )
    add_custom_target(fastsasa_python_mdanalysis_adapter_unit
        COMMAND ${CMAKE_COMMAND} -E env FASTSASA_NATIVE_LIBRARY=$<TARGET_FILE:fastsasa_native>
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_mdanalysis_adapter_unit.py
                --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
        DEPENDS fastsasa_native
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
        COMMENT "Validating FastSASA MDAnalysis-compatible adapter without CUDA"
    )
    if(FASTSASA_HAVE_VULKAN)
        add_custom_target(fastsasa_vulkan_backend_validate
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan_backend_validation.py
                    --fastsasa $<TARGET_FILE:fastsasa_cli>
                    --repo-root ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS fastsasa_cli
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Validating the integrated Vulkan SR/LR fallback"
        )
        add_custom_target(fastsasa_vulkan_batch_validation
            COMMAND $<TARGET_FILE:fastsasa_vulkan_batch_validate>
            DEPENDS fastsasa_vulkan_batch_validate
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Validating reusable Vulkan trajectory batching"
        )
        add_custom_target(fastsasa_python_feature_vulkan_validate
            COMMAND ${CMAKE_COMMAND} -E env
                    FASTSASA_BACKEND=vulkan
                    FASTSASA_REQUIRE_GPU_TESTS=1
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_feature_validation.py
                    --library $<TARGET_FILE:fastsasa_native>
                    --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
            DEPENDS fastsasa_native
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Validating Python feature extraction on Vulkan"
        )
    endif()
endif()

if(FASTSASA_BUILD_NATIVE_TESTS)
    add_custom_target(fastsasa_trajectory_cli_dispatch_validate
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/trajectory_cli_dispatch_validation.py
                --fastsasa $<TARGET_FILE:fastsasa_cli>
        DEPENDS fastsasa_cli
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Validating unified FastSASA trajectory CLI dispatch"
    )
    add_custom_target(fastsasa_trajectory_selection_validate
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/trajectory_selection_validation.py
                --fastsasa $<TARGET_FILE:fastsasa_cli>
        DEPENDS fastsasa_cli
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Validating FastSASA trajectory frame/filter/select semantics"
    )
    add_custom_target(fastsasa_scientific_validate
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/scientific_validation.py
                --fastsasa $<TARGET_FILE:fastsasa_cli>
                --repo-root ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS fastsasa_cli
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running broader FastSASA scientific fixture validation"
    )
endif()

if(FASTSASA_BUILD_CLI)
    install(TARGETS fastsasa_cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(FASTSASA_BUILD_NATIVE_TESTS)
    add_test(NAME fastsasa_cli_validation
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_validation.py
                --fastsasa $<TARGET_FILE:fastsasa_cli>
                --data-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
    )
    add_test(NAME fastsasa_mmcif_selection
        COMMAND $<TARGET_FILE:fastsasa_mmcif_selection_validate>
    )
    add_test(NAME fastsasa_mmcif_corpus
        COMMAND $<TARGET_FILE:fastsasa_mmcif_corpus_validate>
    )
    if(FASTSASA_EXTERNAL_MMCIF_FIXTURES)
        add_test(NAME fastsasa_external_mmcif_corpus
            COMMAND $<TARGET_FILE:fastsasa_mmcif_corpus_validate> ${FASTSASA_EXTERNAL_MMCIF_FIXTURES}
        )
    endif()
    add_test(NAME fastsasa_python_feature_unit
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_feature_unit.py
                --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
    add_test(NAME fastsasa_python_mdanalysis_adapter_unit
        COMMAND ${CMAKE_COMMAND} -E env FASTSASA_NATIVE_LIBRARY=$<TARGET_FILE:fastsasa_native>
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_mdanalysis_adapter_unit.py
                --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
    add_test(NAME fastsasa_trajectory_cli_dispatch
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/trajectory_cli_dispatch_validation.py
                --fastsasa $<TARGET_FILE:fastsasa_cli>
    )
    # The same feature/RSA validation the GPU builds run, pinned to the
    # threaded CPU tier so CPU-only builds exercise it too (including the
    # --format rsa cross-check, the CPU FP32 path, and the FP64 fallbacks).
    add_test(NAME fastsasa_python_feature_cpu
        COMMAND ${CMAKE_COMMAND} -E env
                FASTSASA_BACKEND=cpu
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_feature_validation.py
                --library $<TARGET_FILE:fastsasa_native>
                --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
                --fastsasa $<TARGET_FILE:fastsasa_cli>
    )
    set_tests_properties(fastsasa_python_feature_cpu
        PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} LABELS "cpu;release"
    )
    if(FASTSASA_HAVE_VULKAN)
        add_test(NAME fastsasa_vulkan_backend
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan_backend_validation.py
                    --fastsasa $<TARGET_FILE:fastsasa_cli>
                    --repo-root ${CMAKE_CURRENT_SOURCE_DIR}
                    --allow-missing-device
        )
        set_tests_properties(fastsasa_vulkan_backend
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       SKIP_RETURN_CODE 77
                       LABELS "vulkan"
        )
        add_test(NAME fastsasa_vulkan_batch
            COMMAND $<TARGET_FILE:fastsasa_vulkan_batch_validate>
        )
        set_tests_properties(fastsasa_vulkan_batch
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       SKIP_RETURN_CODE 77
                       LABELS "vulkan"
        )
        add_test(NAME fastsasa_vulkan_surface_points
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan_surface_points_validation.py
                    --fastsasa $<TARGET_FILE:fastsasa_cli>
                    --repo-root ${CMAKE_CURRENT_SOURCE_DIR}
                    --allow-missing-device
        )
        set_tests_properties(fastsasa_vulkan_surface_points
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       SKIP_RETURN_CODE 77
                       LABELS "vulkan"
        )
        add_test(NAME fastsasa_python_feature_vulkan
            COMMAND ${CMAKE_COMMAND} -E env
                    FASTSASA_BACKEND=vulkan
                    FASTSASA_REQUIRE_GPU_TESTS=1
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_feature_validation.py
                    --library $<TARGET_FILE:fastsasa_native>
                    --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
                    --fastsasa $<TARGET_FILE:fastsasa_cli>
        )
        set_tests_properties(fastsasa_python_feature_vulkan
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       LABELS "vulkan"
        )
    endif()
    add_test(NAME fastsasa_scientific_fixtures
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/scientific_validation.py
                --fastsasa $<TARGET_FILE:fastsasa_cli>
                --repo-root ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(
        fastsasa_cli_validation
        fastsasa_mmcif_selection
        fastsasa_mmcif_corpus
        fastsasa_python_feature_unit
        fastsasa_python_mdanalysis_adapter_unit
        fastsasa_trajectory_cli_dispatch
        fastsasa_scientific_fixtures
        PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} LABELS "cpu;release"
    )
    if(TEST fastsasa_external_mmcif_corpus)
        set_tests_properties(fastsasa_external_mmcif_corpus
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} LABELS "cpu;release"
        )
    endif()

    if(FASTSASA_ENABLE_CUDA)
        add_test(NAME fastsasa_cuda_boundary
            COMMAND $<TARGET_FILE:fastsasa_cuda_boundary_validate>
        )
        add_test(NAME fastsasa_selected_center
            COMMAND $<TARGET_FILE:fastsasa_selected_center_validate>
        )
        add_test(NAME fastsasa_context_isolation
            COMMAND $<TARGET_FILE:fastsasa_context_isolation_validate>
        )
        add_test(NAME fastsasa_autotune_dispatch
            COMMAND $<TARGET_FILE:fastsasa_autotune_dispatch_validate>
        )
        add_test(NAME fastsasa_python_feature_gpu
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_feature_validation.py
                    --library $<TARGET_FILE:fastsasa_native>
                    --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
                    --fastsasa $<TARGET_FILE:fastsasa_cli>
        )
        add_test(NAME fastsasa_backend_bit_identity
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/backend_bit_identity_validation.py
                    --library $<TARGET_FILE:fastsasa_native>
                    --python-dir ${CMAKE_CURRENT_SOURCE_DIR}/python
                    --fastsasa $<TARGET_FILE:fastsasa_cli>
        )
        add_test(NAME fastsasa_trajectory_selection_optional
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/trajectory_selection_validation.py
                    --fastsasa $<TARGET_FILE:fastsasa_cli>
        )
        set_tests_properties(
            fastsasa_cuda_boundary
            fastsasa_selected_center
            fastsasa_context_isolation
            fastsasa_autotune_dispatch
            fastsasa_python_feature_gpu
            fastsasa_backend_bit_identity
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} LABELS "gpu;release"
        )
        set_tests_properties(fastsasa_trajectory_selection_optional
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} LABELS "gpu;optional;trajectory"
        )
    endif()

    add_custom_target(fastsasa_validate
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG>
        DEPENDS fastsasa_cli fastsasa_native fastsasa_mmcif_selection_validate fastsasa_mmcif_corpus_validate
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running the complete FastSASA validation suite"
    )
endif()

foreach(FASTSASA_DEVELOPER_TARGET
        fastsasa
        fastsasa_native
        fastsasa_cli
        fastsasa_cuda_boundary_validate
        fastsasa_selected_center_validate
        fastsasa_context_isolation_validate
        fastsasa_autotune_dispatch_validate
        fastsasa_mmcif_selection_validate
        fastsasa_mmcif_corpus_validate)
    if(TARGET ${FASTSASA_DEVELOPER_TARGET})
        fastsasa_apply_developer_options(${FASTSASA_DEVELOPER_TARGET})
    endif()
endforeach()
