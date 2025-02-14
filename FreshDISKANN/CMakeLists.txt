if(MSVC)
	cmake_minimum_required(VERSION 3.15)
else()
	cmake_minimum_required(VERSION 2.8)
endif()
set(CMAKE_STANDARD 14)


#set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/CompilerOptions.cmake")
# if(MSVC)
# 	set(CMAKE_CXX_COMPILER $ENV{VCToolsInstallDir}/bin/Hostx64/x64/cl.exe)
# 	set(CMAKE_CXX_LINK_EXECUTABLE $ENV{VCToolsInstallDir}/bin/Hostx64/x64/link.exe)
# else()
# 	set(CMAKE_CXX_COMPILER g++)
# endif()

project(diskann)
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/dll)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()


function(checkEnvAndSetLocalVar env_var msg local_var)
	if (NOT EXISTS "$ENV{${env_var}}" )
		message (FATAL_ERROR ${msg})
	else()
		if ($ENV{${env_var}} MATCHES "\\$"  OR $ENV{${env_var}} MATCHES "/$" )
			set(${local_var} $ENV{${env_var}} PARENT_SCOPE)
		else()
			message(STATUS "Appending trailing backslash to ${env_var}")
			set(${local_var} "$ENV{${env_var}}\\" PARENT_SCOPE)
		endif()
	endif()
endfunction()

	

#MKL Config
if (MSVC)
#	if (NOT EXISTS "$ENV{INTEL_ROOT}" )
#		message (FATAL_ERROR "Please install MKL libraries and set env var INTEL_ROOT to the intel software directory. Should be similar to: C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\. ")
#	else()
#		if ($ENV{INTEL_ROOT} MATCHES "\\$"  OR $ENV{INTEL_ROOT} MATCHES "/$" )
#			set(INTEL_ROOT $ENV{INTEL_ROOT})
#		else()
#			message(STATUS "Appending trailing backslash to Intel tools path")
#			set(INTEL_ROOT "$ENV{INTEL_ROOT}\\")
#		endif()
#	endif()
	checkEnvAndSetLocalVar("INTEL_ROOT" "Please install Intel MKL libraries and set the env variable INTEL_ROOT to the intel software directory. Should be similar to: C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\. " "INTEL_ROOT")
	set(MKL_ROOT ${INTEL_ROOT}/mkl)
	add_compile_options(/arch:AVX2 /Qpar)
	link_libraries("${INTEL_ROOT}/mkl/lib/intel64/mkl_core_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_rt.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_thread_dll.lib" "${INTEL_ROOT}/compiler/lib/intel64/libiomp5md.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_ilp64_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_sequential_dll.lib")

	checkEnvAndSetLocalVar("BOOST_ROOT"  "Please download boost sources from boost.org and set env var BOOST_ROOT to the download folder." "BOOST_ROOT")
#	if (NOT EXISTS "$ENV{BOOST_ROOT}")
#		message (FATAL_ERROR "Please download boost sources from boost.org and set env var BOOST_ROOT to the appropriate directory.")	
#	else()
#		if ($ENV{BOOST_ROOT} MATCHES "\\$"  OR $ENV{BOOST_ROOT} MATCHES "/$" )
#			set(BOOST_ROOT $ENV{BOOST_ROOT})
#		else()
#			message(STATUS "Appending trailing backslash to Boost sources path")
#			set(BOOST_ROOT "$ENV{BOOST_ROOT}\\")
#		endif()
#	endif()


else()
    # expected path for manual intel mkl installs
    set(POSSIBLE_OMP_PATHS "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so;/usr/lib/x86_64-linux-gnu/libiomp5.so;/opt/intel/lib/intel64_lin/libiomp5.so")
    foreach(POSSIBLE_OMP_PATH ${POSSIBLE_OMP_PATHS})
        if (EXISTS ${POSSIBLE_OMP_PATH})
            get_filename_component(OMP_PATH ${POSSIBLE_OMP_PATH} DIRECTORY)
        endif()
    endforeach()

    if(NOT OMP_PATH)
        message(FATAL_ERROR "Could not find Intel OMP in standard locations; use -DOMP_PATH to specify the install location for your environment")
    endif()
    link_directories(${OMP_PATH})

    set(POSSIBLE_MKL_LIB_PATHS "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;/opt/intel/mkl/lib/intel64/libmkl_core.so")
    foreach(POSSIBLE_MKL_LIB_PATH ${POSSIBLE_MKL_LIB_PATHS})
        if (EXISTS ${POSSIBLE_MKL_LIB_PATH})
            get_filename_component(MKL_PATH ${POSSIBLE_MKL_LIB_PATH} DIRECTORY)
        endif()
    endforeach()

    set(POSSIBLE_MKL_INCLUDE_PATHS "/opt/intel/oneapi/mkl/latest/include;/usr/include/mkl;/opt/intel/mkl/include/;")
    foreach(POSSIBLE_MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATHS})
        if (EXISTS ${POSSIBLE_MKL_INCLUDE_PATH})
            set(MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATH})
        endif()
    endforeach()
    if(NOT MKL_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_PATH to specify the install location for your environment")
    elseif(NOT MKL_INCLUDE_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_INCLUDE_PATH to specify the install location for headers for your environment")
    endif()
    if (EXISTS ${MKL_PATH}/libmkl_def.so.2)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so.2)
    elseif(EXISTS ${MKL_PATH}/libmkl_def.so)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so)
    else()
        message(FATAL_ERROR "Despite finding MKL, libmkl_def.so was not found in expected locations.")
    endif()
    link_directories(${MKL_PATH})
    include_directories(${MKL_INCLUDE_PATH})

    # compile flags and link libraries
    add_compile_options(-m64 -Wl,--no-as-needed)
    if (NOT PYBIND)
        link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
    else()
        # static linking for python so as to minimize customer dependency issues
        link_libraries(
                ${MKL_PATH}/libmkl_intel_ilp64.a
                ${MKL_PATH}/libmkl_intel_thread.a
                ${MKL_PATH}/libmkl_core.a
                ${MKL_DEF_SO}
                iomp5
                pthread
                m
                dl
        )
    endif()
endif()

add_definitions(-DMKL_ILP64)
include_directories(include ${INTEL_ROOT}/include ${MKL_ROOT}/include)
# find_package(MKL CONFIG REQUIRED)
# if(MKL_FOUND)
#   message(STATUS "MKL found")
#   add_compile_definitions(USE_MKL)
# #   target_link_libraries(alaya PUBLIC MKL::MKL)
# else()
#   find_package(BLAS)
#   add_compile_definitions(USE_BLAS)
# #   target_link_libraries(alaya PUBLIC ${BLAS_LIBRARIES})
# endif()

#Main compiler/linker settings 
if(MSVC)
	#language options
	add_compile_options(/permissive- /openmp:experimental /Zc:wchar_t
        /Zc:twoPhase- /Zc:forScope /Zc:inline /WX- /std:c++14 /Gd /W3 /MP /Zi /FC /nologo /diagnostics:classic) 
	#code generation options
	add_compile_options(/Qpar /fp:fast /Zp8 /fp:except- /EHsc /GS- /Gm- /Gy )
	#optimization options
	add_compile_options(/Ot /Oy /Oi)
	#path options
	#add_compile_options(/Fdx64/Release/vc141.pdb /Fox64/Release/)
	add_definitions(-DUSE_AVX2 -DUSE_BING_INFRA -DUSE_BING_IO_LIB -DUSE_ACCELERATED_PQ -D_WINDOWS -DNOMINMAX -DUNICODE)
	
	set(CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS "/MANIFEST /MACHINE:X64 /DEBUG:FULL /LTCG:incremental /NXCOMPAT /DYNAMICBASE /OPT:REF /SUBSYSTEM:CONSOLE /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\"")
	set(CMAKE_EXECUTABLE_CXX_LINK_FLAGS "/MANIFEST /MACHINE:X64 /DEBUG:FULL /LTCG:incremental /NXCOMPAT /DYNAMICBASE /OPT:REF /SUBSYSTEM:CONSOLE /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\"")

	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG")
	set(CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS_DEBUG "${CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS_DEBUG} /DEBUG")

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
else()
	# set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -Ofast -DNDEBUG -mavx2 -march=native -mtune=native -ftree-vectorize")
	add_compile_options(-march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_ACCELERATED_PQ -DUSE_AVX2)
	# add_compile_options(-march=native -Wall -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_ACCELERATED_PQ -DUSE_AVX2)
endif()

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
