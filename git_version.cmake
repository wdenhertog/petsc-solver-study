# Run git commands
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        OUTPUT_VARIABLE GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${SOURCE_DIR})

execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain
        OUTPUT_VARIABLE GIT_DIRTY_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${SOURCE_DIR})

if (GIT_DIRTY_OUT STREQUAL "")
    set(GIT_DIRTY "0")
else ()
    set(GIT_DIRTY "1")
endif ()

# Construct the C++ header content
set(HEADER_CONTENT "#pragma once\n#define GIT_SHA \"${GIT_SHA}\"\n#define GIT_DIRTY ${GIT_DIRTY}\n")

# Check existing file content to avoid unnecessary recompilations
if (EXISTS "${GEN_HEADER}")
    file(READ "${GEN_HEADER}" OLD_CONTENT)
    if (OLD_CONTENT STREQUAL HEADER_CONTENT)
        return() # Content is the same, do not overwrite
    endif ()
endif ()

file(WRITE "${GEN_HEADER}" "${HEADER_CONTENT}")
