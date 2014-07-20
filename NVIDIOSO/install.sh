#!/bin/bash
echo "Installing NVIDIOSO"
echo "NVIDIa-based cOnstraint SOlver v. 1.0"
echo "(C) Copyright 2014"
echo "NVIDIOSO is free software."

OBJ=obj
SRC=src
obj_folder=("base" "constraints" "core" "FZ_parser" "search" "exception")


#Create obj (main) folder
if [ ! -d ${OBJ} ]; then
  mkdir ${OBJ}
fi

if [ ! -d ${OBJ}/${SRC} ]; then
mkdir ${OBJ}/${SRC}
fi

#Create obj folders
#@note ${#ArrayName[@]}: length of ArrayName
for ((k=0; k<${#obj_folder[@]}; k++))
do
  if [ -d ${OBJ}/${SRC}/${obj_folder[$k]} ]; then
    echo ${OBJ}/${SRC}/${obj_folder[$k]} "already exists"
  else
    mkdir ${OBJ}/${SRC}/${obj_folder[$k]}
  fi
done

make clean
make

exit 0
#@todo create make.inc here
echo "#CC = gcc" >> data.txt
echo "#CC = g++" >> data.txt
echo "CC = clang++" >> data.txt
echo "DEBUGFLAG = -W -Wall" >> data.txt
echo "CCOPT = -m64 -DIL_STD" >> data.txt
echo "BASE="${obj_folder[0]} >> data.txt
echo "CONSTRAINTS="${obj_folder[1]} >> data.txt
echo "CORE="${obj_folder[2]} >> data.txt
echo "PARSER="${obj_folder[3]} >> data.txt
echo "CONSTRAINTS="${obj_folder[4]} >> data.txt