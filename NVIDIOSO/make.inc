# ---------------------------------------------------------------------
# Compiler options and definition for the GNU libraries
# ---------------------------------------------------------------------
#CC = gcc
#CC = g++
CC = clang++

DEBUGFLAG = -W -Wall
# -m64: generate code for 64 bit env. Use -m32 for a 32 bit env.
# -DIL_STD -> #define IL_STD 1. Used to enable some STL features
CCOPT = -m64 -DIL_STD

## optimize
CCOPT += -O3 -std=c++0x 

# ---------------------------------------------------------------------
# Other options 
# ---------------------------------------------------------------------

OK_COLOR=\x1b[32;01m
ERROR_COLOR=\x1b[31;01m
WARN_COLOR=\x1b[33;01m

ESC=`echo "\033"`
OK_COLOR   ="${ESC}[0;32m"
WARN_COLOR ="${ESC}[0;33m"
NO_COLOR   ="${ESC}[m\017"

OK_STRING=${OK_COLOR}[OK]${NO_COLOR}
OK_DONE =${OK_COLOR}NVIDIOSO compilation succeeded${NO_COLOR}
CLEAN_DONE =${WARN_COLOR}NVIDIOSO Cleaning succeeded${NO_COLOR}

PRINT_COMPILE = @echo ${OK_STRING}
PRINT_COMPILE = @echo compiling...${OK_STRING}
PRINT_DONE    = @echo ${OK_DONE}
PRINT_CLEAN   = @echo ${CLEAN_DONE}

# ---------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------

PRG_PATH=.
NVIDIOSO_INC = include
NVIDIOSO_SRC = src
NVIDIOSO_LIB = lib

# Libraries
LIBNVIDIOSO = $(NVIDIOSO_LIB)/libnvidioso.a

# ---------------------------------------------------------------------
# SRC Folders’ name
# ---------------------------------------------------------------------
BASE=base
CORE=core
SEARCH=search
PARSER=FZ_parser
CONSTRAINTS=constraints

