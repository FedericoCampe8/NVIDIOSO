#!/bin/csh
#########################################################################################
# Copyright (c) 2015, CLP Lab Members.                                                  #								
# All rights reserved.                                                                  #
#                                                                                       #
# Redistribution and use in source and binary forms, with or without                    #
# modification, are permitted provided that the following conditions are met:			#
#                                                                                       #
# 1. Redistributions of source code must retain the above copyright notice, this 		#
#    list of conditions and the following disclaimer.                                   #
# 2. Redistributions in binary form must reproduce the above copyright notice,			#
#    this list of conditions and the following disclaimer in the documentation			#
#    and/or other materials provided with the distribution.                             #
#                                                                                       #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND		#
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED			#
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR		#
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES		#
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;			#
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND			#
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT			#
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS			#
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                          #
#                                                                                       #
# The views and conclusions contained in the software and documentation are those		#
# of the authors and should not be interpreted as representing official policies, 		#
# either expressed or implied, of the FreeBSD Project.                                  #
#########################################################################################

#########################################################################################
# COMMENT/UNCOMMENT THE FOLLOWING IF DIFFERENT FROM THE ACTUAL ARCHITECTURE             #
#set CUDAOPT = "-arch=sm_20"
#set CUDAOPT = "-arch=sm_35"
set CUDAOPT = "-arch=sm_30"
#                                                                                       #
# THE FOLLOWING OPTION SETS THE PATH FOR THE HOST COMPILER TO USE IN AN OSX ENVIRONMENT #
# CHANGE THE PATH IF USING OSX AND THE COMPILER OR THE PATH ARE DIFFERENT               #
set OSXENV = '-ccbin /usr/local/Cellar/gcc47/4.7.4/bin/g++-4.7'
# THE FOLLOWING OPTION SETS THE PATH FOR THE CUDA COMPILER                              #
# SET HERE GLOBAL PATH TO NVCC, e.g., set NVCC =/machine_name/cuda-6.5/bin/nvcc         #
set NVCC = nvcc
# SET HERE c++0x for gcc versions less than 4.7                                         #
set CCPP = c++11
#########################################################################################

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

clear
echo "#########################################"
echo "#               iNVIDIOSO               #"
echo "# NVIDIa-based cOnstraint SOlver v. 1.0 #"
echo "# Copyright (c) 2015, CLP Lab Members.  #"
echo "# All rights reserved.                  #"
echo "#########################################"

echo "Install iNVIDIOSO? (y, n)"
set INST = $<
if ( $INST != "y" && $INST != "Y" ) then
	exit 0
endif

set PROG_COUNT = `ls | grep -c -x nvidioso`
if ( $PROG_COUNT != 0 ) then
	echo "A version of the solver is already present in this folder."
	echo "Press any key to continue or ctrl-c to exit."
	set EXT = $< 
endif

echo "Install CPU or GPU version? (c, g)"
set VERSION = $<
if ( $VERSION != "c" && $VERSION != "g" ) then
	set VERSION = "c"
endif

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
        echo "Unknown: $OS"
		exit 1
endsw

set CC = "g++"
set CUDAON = "false"
set architecture = `uname -m`
set machine_val  = "-m64"
if ( $architecture != "x86_64" ) then
	set machine_val = "-m32"
endif

if ( $VERSION == "g" ) then 
	echo ""
	#set NVIDIA = `nvcc --version | grep -c NVIDIA`
	if ( { nvcc --version } == 0 ) then
		echo "nvcc compiler not found."
		echo "1 - Check if CUDA Toolkit is intalled in your platform."
		echo '    You can download it from http://developer.nvidia.com/cuda-downloads.'
		echo "    They also provide useful NVIDIA '"'Getting Started'"' guides." 
		echo "2 - If CUDA is already installed in your platform,"
		echo "    try to export environment variables:"
		echo '    export PATH=/Developer/NVIDIA/CUDA-6.5/bin:$PATH'
		echo '    export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-6.5/lib:$DYLD_LIBRARY_PATH'
		echo '    export PATH=/usr/local/cuda/:$PATH'
		echo "3 - Relaunch this install."
		exit 0
	else
		echo ""
		echo "Note: CUDA architecture setting: $CUDAOPT."
		echo "If different, please change architecture in '"'install.csh'"' before proceeding."
		echo "Note: This version of the solver is written in C++11 which is supported"
		echo "      only by CUDA 6.5 and higher."
		echo " 		To download the lastest versions of CUDA, please visit:"
		echo '      http://developer.nvidia.com/cuda-downloads.'     
		echo "Press any key to continue or ctrl-c to exit."
		set EXT = $< 
		#if ( $EXT == "y" || $EXT == "Y" ) then
		#	exit 0
		#endif
		set CC     = "nvcc"
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
set SRC = src
set OBJ_FOLDER = (base constraints core FZ_parser search exception)

# Create obj (main) folder
if ( -d $OBJ ) then
	rm -rf ${OBJ}
endif
mkdir ${OBJ}
mkdir ${OBJ}/${SRC}
 
# Create obj folders
foreach dir ($OBJ_FOLDER)
mkdir ${OBJ}/${SRC}/${dir}
end

# Create make.inc file
set MAKEINC = "make.inc"
if ( -f $MAKEINC ) then
	rm -f ${MAKEINC}
endif
goto CreateMakeInc

ProceedWithMake:
set FILELOG = "install_`date +%m-%d-%y`.log"
make clean >& $FILELOG
make >>& $FILELOG &
  
set MAKEBG = `echo $!`

set CNT     = 1
set VARMAKE = `ps -ef -o pid -u ${USER} | grep -v grep | grep -c $MAKEBG`
echo -n "Installing..."
while ( $VARMAKE > 0 )
	set VARMAKE = `ps -ef -o pid -u ${USER} | grep -v grep | grep -c $MAKEBG`
	
	echo -n "."
	if ( $CNT % 31 == 0 ) then
		echo ""
	endif
	@ CNT++
	
	sleep 2
end
echo ""

# Check possible errors
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
touch $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo "# Compiler options and definition for the GNU libraries      " 	>> $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo ""                                                                 >> $MAKEINC
echo "CC = $CC"                                                         >> $MAKEINC
echo "COMPILE = -c"                                                     >> $MAKEINC
echo "DEBUGFLAG = -W -Wall"                                             >> $MAKEINC
echo "CCOPT = $machine_val -DIL_STD"                                    >> $MAKEINC
echo "CCOPT += -O3 $CCPP -fPIC"                                         >> $MAKEINC
echo "CFLAGS = -DCUDAON=$CUDAON"                                        >> $MAKEINC
echo ""                                                                 >> $MAKEINC
if ( $CC == "nvcc" ) then 
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo "# CUDA options                                               " 	>> $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo ""                                                                 >> $MAKEINC
echo "CCOPT += $CUDAOPT"                                                >> $MAKEINC
if ( $OS == "OSX" ) then
echo "CUDAOPT += $OSXENV"                                               >> $MAKEINC
endif
echo "COMPILE = -x cu -dc"                                              >> $MAKEINC
echo ""                                                                 >> $MAKEINC
endif 
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo "# Other options                                              " 	>> $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo ""                                                                 >> $MAKEINC
echo "OK_STRING=[OK]"                                                   >> $MAKEINC
echo "OK_DONE =iNVIDIOSO compilation succeeded"                         >> $MAKEINC
echo "CLEAN_DONE =NVIDIOSO Cleaning succeeded"                          >> $MAKEINC
echo 'PRINT_COMPILE = @echo compiling...${OK_STRING}'                   >> $MAKEINC
echo 'PRINT_CLEAN   = @echo ${CLEAN_DONE}'                              >> $MAKEINC
echo ""                                                                 >> $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo "# Paths                                                      " 	>> $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo ""                                                                 >> $MAKEINC
echo "PRG_PATH=."                                                       >> $MAKEINC
echo "NVIDIOSO_INC = include"                                           >> $MAKEINC
echo "NVIDIOSO_SRC = src"                                               >> $MAKEINC
echo "NVIDIOSO_LIB = lib"                                               >> $MAKEINC
echo 'LIBNVIDIOSO = $(NVIDIOSO_LIB)/libnvidioso.a'                      >> $MAKEINC
echo ""                                                                 >> $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo "# SRC Folders’ name                                          " 	>> $MAKEINC
echo "#------------------------------------------------------------" 	>> $MAKEINC
echo ""                                                                 >> $MAKEINC
echo "BASE=base"                                                        >> $MAKEINC
echo "CORE=core"                                                        >> $MAKEINC
echo "SEARCH=search"                                                    >> $MAKEINC
echo "PARSER=FZ_parser"                                                 >> $MAKEINC
echo "CONSTRAINTS=constraints"                                          >> $MAKEINC
echo "EXCEPTION=exception"                                              >> $MAKEINC

goto ProceedWithMake

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

OSXNote:
echo ""
echo "On OSX system the nvcc compiler might cause some problems."
echo "Here are some useful tips:"
echo "1 - Check if clang is installed in the system by writing clang++ in the terminal." 
echo "2 - Xcode Command Line Tools should be installed."
echo '    To install Xcode Command Line Tools enter "xcode-select --install" in the terminal' 
echo "3 - g++ should be installed in order to work with nvcc."
echo "    If g++4.7 or higher is already installed in your system you can go directly 
echo "    to step 4. Otherwise, install g++ as follows.
echo "    To install g++ we suggest to use Homebrew."
echo "    Type:"
echo '        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"'
echo "    to install homebrew if not present in your system."
echo "    To install gcc and g++ type the following commands:"
echo "        brew update"
echo '        brew tap homebrew/versions'
echo "        brew install [flags] gcc48"
echo "    You can view available install flags by using"
echo "        brew options gcc48 
echo "    If you don't want to use Homebrew," 
echo "    you may want to have a look at the gcc website
echo '        gcc.gnu.org'
echo "    and, in particular, at the Installation section."
echo "    If you used Homebrew to install the compiler,"
echo "    you should be able to locate it in the following directory:"
echo '        /usr/local/Cellar/gcc47/' 
echo "    In any case, you need to change the global variable storing the path"
echo "    where the compiler is located to be used together with nvcc."
echo '4 - If <path> is the path where the compiler is located, open "install.csh" '
echo '    and change the global variable "OSXENV" as follows:'
echo "    set OSXENV = '-ccbin <path>'"
echo '    Note that -ccbin <dir> specify the directory <dir> containing'
echo "    the compiler to use as host compiler."
echo "    Note the gcc48 is also available but at this time there is "
echo "    a bug in some of its libraries which makes it incompatible with nvcc."
echo "    Please, use gcc47."
echo "Press any key to continue or ctrl-c to exit."
set EXT = $< 
goto ProceedWithInstall

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

