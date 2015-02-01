#------------------------------------------------------------
# Compiler options and definition for the GNU libraries      
#------------------------------------------------------------

CC = nvcc
COMPILE = -c
DEBUGFLAG = -W -Wall
CCOPT = -m64 -DIL_STD
CCOPT += -O3 -std=c++11
CFLAGS = -DCUDAON=true

#------------------------------------------------------------
# CUDA options                                               
#------------------------------------------------------------

CCOPT += -arch=sm_30
CUDAOPT += -ccbin /usr/local/Cellar/gcc47/4.7.4/bin/g++-4.7
COMPILE = -x cu -dc

#------------------------------------------------------------
# Other options                                              
#------------------------------------------------------------

OK_STRING=[OK]
OK_DONE =iNVIDIOSO compilation succeeded
CLEAN_DONE =NVIDIOSO Cleaning succeeded
PRINT_COMPILE = @echo compiling...${OK_STRING}
PRINT_CLEAN   = @echo ${CLEAN_DONE}

#------------------------------------------------------------
# Paths                                                      
#------------------------------------------------------------

PRG_PATH=.
NVIDIOSO_INC = include
NVIDIOSO_SRC = src
NVIDIOSO_LIB = lib
LIBNVIDIOSO = $(NVIDIOSO_LIB)/libnvidioso.a

#------------------------------------------------------------
# SRC Folders’ name                                          
#------------------------------------------------------------

BASE=base
CORE=core
SEARCH=search
PARSER=FZ_parser
CONSTRAINTS=constraints
EXCEPTION=exception
