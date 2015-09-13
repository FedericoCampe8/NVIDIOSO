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

ARCHITECTURE_ALL=$(uname -a)

FILELOG="unit_test_report_`date +%Y%m%d`.log"
echo "====================== iNVIDIOSO-1.0 Unit Test ======================"
echo "====================================================================="
echo "Architecture: $ARCHITECTURE_ALL"
echo "====================================================================="
echo "====================== iNVIDIOSO-1.0 Unit Test ======================" > $FILELOG
echo "=====================================================================" >> $FILELOG
echo "Architecture: $ARCHITECTURE_ALL" >> $FILELOG
echo "=====================================================================" >> $FILELOG

valgrind --leak-check=yes -v ./invidioso -v 1>>$FILELOG  2>&1 &
MAKEBG=`echo $!`

CNT=1
VARMAKE=`ps -eu ${USER} | grep -v grep | grep -c $MAKEBG`
echo -n "Running unit tests..."
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

echo "=====================================================================" >> $FILELOG
echo "=====================================================================" >> $FILELOG

# Remove the log (duplicate)
rm unit_test_[0-9]*.log

# Remove
#rm -rf invidioso.dSYM

# Analyze results
./invidioso -a $FILELOG

echo "====================================================================="
echo "====================================================================="
