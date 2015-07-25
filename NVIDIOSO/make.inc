#------------------------------------------------------------
# Compiler options and definition for GNU libraries
#------------------------------------------------------------

CC = g++
TARGET = invidioso
AR = ar
ARFLAGS = rcs
COMPILE = -c
DEBUGFLAG = -W -Wall
CCOPT = -m64 -DIL_STD
CCOPT += -O2 -std=c++11
CFLAGS = -DCUDAON=false

#------------------------------------------------------------
# Variable definitions
#------------------------------------------------------------

OK_STRING=[OK]
OK_DONE =iNVIDIOSO compilation succeeded
CLEAN_DONE =iNVIDIOSO Cleaning succeeded
PRINT_COMPILE = @echo compiling...${OK_STRING}
PRINT_CLEAN   = @echo ${CLEAN_DONE}

#------------------------------------------------------------
# Paths
#------------------------------------------------------------

PRG_PATH=.
NVIDIOSO_INS = bin
NVIDIOSO_INC = include
NVIDIOSO_SRC = src
NVIDIOSO_LIB = lib
LIBNVIDIOSO = $(NVIDIOSO_LIB)/libnvidioso.a

#------------------------------------------------------------
# SRC Folders name
#------------------------------------------------------------

BASE=base
CORE=core
SEARCH=search
PARSER=parser
FZN_PARSER=FZ_parser
VARIABLE=variable
CONSTRAINTS=constraints
CONSTRAINT_STORE=constraint_store
CUDA_VARIABLE=cuda_variable
CUDA_CONSTRAINTS=cuda_constraints
CUDA_CONSTRAINT_STORE=cuda_constraint_store
CUDA_UTILITIES=cuda_utilities
EXCEPTION=exception
