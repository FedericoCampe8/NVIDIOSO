#! /bin/bash
#########################################################################################
# Copyright (c) 2014-2015, Campeotto Federico.                                          #
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
. config/install_functions

clear
cat <<EOF
###########################################
#               iNVIDIOSO                 #
# NVIDIa-based cOnstraint SOlver v. 1.0   #
# Copyright (c) 2015, Federico Campeotto. #
# All rights reserved.                    #
###########################################
EOF

PROG_COUNT=`ls | grep -c -x $iNVIDIOSO`
if [ $PROG_COUNT -ne 0 ]; then
    echo "A version of the solver is already present in this folder."
    echo "Press any key to continue or ctrl-c to exit."
    read EXT
fi

echo "Install CPU or GPU version? [C|G]"
read VERSION
while [ $VERSION != "C" -a $VERSION != "G" ]
do
  echo "Please, use C for CPU and G for GPU"
  read VERSION
done

# Default parameters
OS=$(uname)
case "$OS" in
    [Ss]olaris* )
    OS="SOLARIS" 
    ;;
    [Dd]arwin* )
    OS="OSX"
    ;;
    [Ll]inux* )
    OS="LINUX"
    ;;
    [Bb]sd* )
    OS="BSD"
    ;;
    CYGWIN* )
    OS="CYGWIN"
    ;;
    * )
        echo "Unknown OS: $OS"
        exit 1;;
esac

# Set compiler and machine architecture
CC="g++"
CUDAON="false"
ARCHITECTURE=$(uname -m)
ARCHITECTURE_ALL=$(uname -a)
MACHINE_VAL="-m64"
if [ $ARCHITECTURE != "x86_64" ]; then
    MACHINE_VAL="-m32"
fi

# Check for cuda compiler if GPU option is selected
if [ $VERSION == "G" ]; then
    echo ""
    if [ { $NVCC --version } -eq 0 ]; then
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

		read EXT 
        CC=$NVCC 
        CUDAON="true"
    fi
fi

if [ $OS == "OSX" -a $CC == "nvcc" ]; then
    OSXNote
fi

echo ""
echo "Installing iNVIDIOSO (`date`)"

OBJ=obj
LIB=lib
SRC=src
OBJ_FOLDER=("base constraints" "cuda_constraints" \
"global_constraints" "cuda_global_constraints" "constraint_store" \
"cuda_constraint_store" "parser" "FZ_parser" "core" "search exception" \
"local_search" "cuda_local_search" \
"cuda_utilities" "variable cuda_variable" )

# Create install folder if default is used
if [ ${INSTALL_PATH} == "bin" ]; then
    if [ ! -d ${INSTALL_PATH} ]; then
        mkdir ${INSTALL_PATH}
    fi
else
    if [ ! -d $INSTALL_PATH ]; then
        echo "Installation folder: $INSTALL_PATH not found"
        exit 1
    fi
fi

# Create lib folder for library
if [ ! -d $LIB ]; then
    mkdir ${LIB}
fi

# Create obj (main) folder
if [ -d $OBJ ]; then
    rm -rf ${OBJ}
fi
mkdir ${OBJ}
mkdir ${OBJ}/${SRC}

# Create obj folders
for dir in ${OBJ_FOLDER[@]} ; do
    mkdir ${OBJ}/${SRC}/${dir}
done

# Create make.inc file according to the current architecture
MAKE_INC="make.inc"
MAKE_INC_CUDA="make_cuda.inc"
if [ -f ${MAKE_INC} ]; then
    rm -f ${MAKE_INC}
fi
if [ -f ${MAKE_INC_CUDA} ]; then
    rm -f ${MAKE_INC_CUDA}
fi

CreateMakeInc

# Continue installation using make.inc
FILELOG="install_`date +%Y%m%d`.log"
echo "====== iNVIDIOSO-1.0 Installation ======" > $FILELOG
echo "Architecture: $ARCHITECTURE_ALL" >> $FILELOG

make clean 1>>$FILELOG
make 1>>$FILELOG &

MAKEBG=`echo $!`

CNT=1
VARMAKE=`ps -eu ${USER} | grep -v grep | grep -c $MAKEBG`
echo -n "Installing..."
echo ""
while [ $VARMAKE -gt 0 ]
do
  VARMAKE=`ps -eu ${USER} | grep -v grep | grep -c $MAKEBG`
  echo -n "."
  NUM_LINE=`expr $CNT % 31`
  if [ $NUM_LINE -eq 0 ]; then
      echo ""
  fi
  CNT=`expr $CNT + 1`
  sleep 2
done
echo ""

# Check install errors
VARERROR=`grep -c -i Error $FILELOG`
VARSTOP=`grep -c -i Stop $FILELOG`
PROG_COUNT=`ls | grep -c -x $iNVIDIOSO`
if [ $VARERROR -gt 0 ] || [ $VARSTOP -gt 0 ] || [ $PROG_COUNT -eq 0 ]; then
    echo "Something went wrong during installation."
    echo "Check log file: $FILELOG"
else
    echo "Installation completed (`date`)."
fi

exit 0
