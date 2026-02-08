cmake_minimum_required(VERSION 3.22)
cmake_policy(SET CMP0135 NEW)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(PhaseCorrector VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch and build FFTW3 as static library with SIMD support
include(FetchContent)
FetchContent_Declare(
    fftw3
    URL https://www.fftw.org/fftw-3.3.10.tar.gz
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_THREADS OFF CACHE BOOL "" FORCE)
set(ENABLE_FLOAT OFF CACHE BOOL "" FORCE)
set(ENABLE_LONG_DOUBLE OFF CACHE BOOL "" FORCE)
set(ENABLE_QUAD_PRECISION OFF CACHE BOOL "" FORCE)
set(ENABLE_SSE ON CACHE BOOL "" FORCE)
set(ENABLE_SSE2 ON CACHE BOOL "" FORCE)
set(ENABLE_AVX ON CACHE BOOL "" FORCE)
set(ENABLE_AVX2 ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fftw3)

# Add JUCE as a subdirectory
add_subdirectory("C:/JUCE" "${CMAKE_BINARY_DIR}/JUCE")

juce_add_plugin(PhaseCorrector
    COMPANY_NAME "PhaseAudio"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE Phse
    PLUGIN_CODE PhCr
    FORMATS VST3 Standalone
    PRODUCT_NAME "PhaseCorrector"
)

target_sources(PhaseCorrector
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
)

target_compile_definitions(PhaseCorrector
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_include_directories(PhaseCorrector PRIVATE ${fftw3_SOURCE_DIR}/api)

target_link_libraries(PhaseCorrector
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        fftw3
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Enable AVX2 SIMD for optimized frequency-domain processing
if(MSVC)
    target_compile_options(PhaseCorrector PRIVATE /arch:AVX2)
else()
    target_compile_options(PhaseCorrector PRIVATE -mavx2 -mfma)
endif()
