#!/bin/csh
#########################################################################################
# Copyright (c) 2015, Campeotto Federico.                                               #
# All rights reserved.                                                                  #
#                                                                                       #
# Redistribution and use in source and binary forms, with or without                    #
# modification, are permitted provided that the following conditions are met:           #
#                                                                                       #
# 1. Redistributions of source code must retain the above copyright notice, this        #
#    list of conditions and the following disclaimer.                                   #
# 2. Redistributions in binary form must reproduce the above copyright notice,          #
#    this list of conditions and the following disclaimer in the documentation          #
#    and/or other materials provided with the distribution.                             #
#                                                                                       #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND       #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED         #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR       #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES        #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;          #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND           #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT            #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                          #
#                                                                                       #
# The views and conclusions contained in the software and documentation are those       #
# of the authors and should not be interpreted as representing official policies,       #
# either expressed or implied, of the FreeBSD Project.                                  #
#########################################################################################

source config/iNVIDIOSO.envs

clear
cat <<EOF
###########################################
#               iNVIDIOSO                 #
# NVIDIa-based cOnstraint SOlver v. 1.0   #
# Copyright (c) 2015, Federico Campeotto. #
# All rights reserved.                    #
###########################################
EOF

set PROG_COUNT = `ls | grep -c -x $iNVIDIOSO`
if ( $PROG_COUNT != 0 ) then
    echo "A version of the solver is already present in this folder."
    echo "Press any key to continue or ctrl-c to exit."
    set EXT = $<
endif

echo "Install CPU or GPU version? (C, G)"
set VERSION = $<
while ( $VERSION != "C" && $VERSION != "G" )   
    echo "Please, use C for CPU and G for GPU" 
    set VERSION = $<
end

# Default parameters
set OS = `uname`
switch ($OS)
    case [Ss]olaris*:
        set OS = "SOLARIS"
        breaksw
    case [Dd]arwin*:
        set OS = "OSX"
        breaksw
    case [Ll]inux*:
        set OS = "LINUX"
        breaksw
    case [Bb]sd*:
        set OS = "BSD"
        breaksw
    default:
        echo "Unknown OS: $OS"
        exit 1
endsw


# Set compiler and machine architecture
set CC     = "g++"
set CUDAON = "false"
set ARCHITECTURE = `uname -m`
set MACHINE_VAL  = "-m64"
if ( $ARCHITECTURE != "x86_64" ) then
    set MACHINE_VAL = "-m32"
endif

# Check for cuda compiler if GPU option is selected
if ( $VERSION == "G" ) then
    echo ""
    if ( { $NVCC --version } == 0 ) then
        cat <<EOF
        nvcc compiler not found.
        1 - Check if CUDA Toolkit is intalled in your platform.
            You can download it from http://developer.nvidia.com/cuda-downloads.
            They also provide useful NVIDIA "Getting Started" guides.
        2 - If CUDA is already installed in your platform,
            try to export environment variables or check iNVIDIOSO.envs for environment variables
            	export LD_LIBRARY_PATH=/usr/local/cuda/lib
            	export PATH=\$PATH:/usr/local/cuda/bin
        3 - Install again.
EOF
    else
        echo ""
        echo "Note: CUDA Compute Capability currently set: $CUDA_CAPABILITY."
        cat <<EOF
        If different, please change architecture in iNVIDIOSO.envs 
        before proceeding with the installation.
        Note: This version of the solver is written in C++11 which is supported
        only by CUDA 6.5 and higher.
        To download the lastest versions of CUDA, please visit:
        http://developer.nvidia.com/cuda-downloads.
        Press any key to continue or ctrl-c to exit.
EOF
        set EXT = $< 
        set CC     = $NVCC 
        set CUDAON = "true"
    endif
endif

if ( $OS == "OSX" && $CC == "nvcc" ) then
    goto OSXNote
endif

ProceedWithInstall:
echo ""
echo "Installing iNVIDIOSO (`date`)"

set OBJ = obj
set LIB = lib
set SRC = src
set OBJ_FOLDER = (base constraints cuda_constraints core FZ_parser search exception)

# Create obj (main) folder
if ( -d $OBJ ) then
    rm -rf ${OBJ}
endif
mkdir ${OBJ}
mkdir ${OBJ}/${SRC}

# Create lib folder for library
if ( ! -d $LIB ) then
    mkdir ${LIB}
endif

# Create obj folders
foreach dir ($OBJ_FOLDER)
    mkdir ${OBJ}/${SRC}/${dir}
end

# Create make.inc file according to the current architecture
set MAKE_INC      = "make.inc"
set MAKE_INC_CUDA = "make_cuda.inc"
if ( -f $MAKE_INC ) then
	rm -f ${MAKE_INC}
endif
if ( -f $MAKE_INC_CUDA ) then
	rm -f ${MAKE_INC_CUDA}
endif

goto CreateMakeInc

# Continue installation using make.inc
ProceedWithMake:
set FILELOG = "install_`date +%m-%d-%y`.log"
make clean >& $FILELOG
make >>& $FILELOG &

set MAKEBG = `echo $!`

set CNT     = 1
set VARMAKE = `ps -e -f -o pid -u ${USER} | grep -v grep | grep -c $MAKEBG`
echo -n "Installing..."
echo ""
while ( $VARMAKE > 0 )
	set VARMAKE = `ps -e -f -o pid -u ${USER} | grep -v grep | grep -c $MAKEBG`
	echo -n "."
	if ( $CNT % 31 == 0 ) then
		echo ""
	endif
	@ CNT++
	
	sleep 2
end
echo ""

# Check installation errors
set VARERROR = `grep -c Error $FILELOG`
if ( $VARERROR > 0 ) then
	echo "Something went wrong during installation."
	echo "Check log file: $FILELOG"
else
	echo "Installation completed (`date`)."
endif

exit 0

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
CreateMakeInc:
touch $MAKE_INC
cat  <<EOF > $MAKE_INC
#------------------------------------------------------------
# Compiler options and definition for GNU libraries
#------------------------------------------------------------

CC = $CC
TARGET = $iNVIDIOSO
AR = ar
ARFLAGS = rcs
COMPILE = -c
DEBUGFLAG = -W -Wall
CCOPT = $MACHINE_VAL -DIL_STD
CCOPT += -O2 -std=$CPP
CFLAGS = -DCUDAON=$CUDAON

#------------------------------------------------------------
# Variable definitions  
#------------------------------------------------------------

OK_STRING=[OK]
OK_DONE =iNVIDIOSO compilation succeeded
CLEAN_DONE =NVIDIOSO Cleaning succeeded
PRINT_COMPILE = @echo compiling...\${OK_STRING}
PRINT_CLEAN   = @echo \${CLEAN_DONE}

#------------------------------------------------------------
# Paths 
#------------------------------------------------------------

PRG_PATH=.
NVIDIOSO_INC = include
NVIDIOSO_SRC = src
NVIDIOSO_LIB = lib
LIBNVIDIOSO = \$(NVIDIOSO_LIB)/libnvidioso.a

#------------------------------------------------------------
# SRC Folders’ name
#------------------------------------------------------------

BASE=base
CORE=core
SEARCH=search
PARSER=FZ_parser
CONSTRAINTS=constraints
CUDA_CONSTRAINTS=cuda_constraints
EXCEPTION=exception
EOF

touch $MAKE_INC_CUDA
set CUDA_ON_OSX = "-ccbin $OSX_COMPILER"
set NVCC_VERSION = `$CC --version | grep -c 7`
if ( $NVCC_VERSION == 1 ) then
	set CUDA_ON_OSX = ""
endif

if ( $CUDAON == "true" ) then 
cat <<EOF > $MAKE_INC_CUDA
CUDAOPT += $CUDA_ON_OSX
CCOPT += -arch=sm_$CUDA_CAPABILITY
COMPILE = -x cu -dc
EOF
endif

goto ProceedWithMake

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
OSXNote:
cat <<EOF
On OSX system the nvcc compiler might cause some problems.
Here are some useful tips:
1 - Check if clang is installed in the system by writing clang++ in the terminal.
2 - Xcode Command Line Tools should be installed.
	To install Xcode Command Line Tools enter "xcode-select --install" in the terminal
3 - g++ should be installed in order to work with nvcc.
	If g++4.7 or higher is already installed in your system you can go directly
	to step 4. Otherwise, install g++ as follows.
	To install g++ we suggest to use Homebrew.
	Type:
	ruby -e "\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	to install homebrew if not present in your system.
	To install gcc and g++ type the following commands:
		brew update
		brew tap homebrew/versions
		brew install [flags] gcc48
	You can view available install flags by using
		brew options gcc48
	If you don't want to use Homebrew,
	you may want to have a look at the gcc website
		gcc.gnu.org
	and, in particular, at the Installation section.
	If you used Homebrew to install the compiler,
	you should be able to locate it in the following directory:
		/usr/local/Cellar/gcc47/
	In any case, you need to change the global variable storing the path
	where the compiler is located to be used together with nvcc.
4 - THIS STEP IS NO LONGER NEEDED IN CUDA7
	If <path> is the path where the compiler is located, open "iNVIDIOSO.envs"
	and set <path> as the env variable OSX_COMPILER.
	
Press any key to continue or ctrl-c to exit.
EOF
set EXT = $< 
goto ProceedWithInstall



