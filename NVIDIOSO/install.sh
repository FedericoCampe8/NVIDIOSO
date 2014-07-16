#!/bin/bash
echo "Installing NVIDIOSO"
echo "NVIDIa-based cOnstraint SOlver v. 1.0"
echo "(C) Copyright 2014"
echo "NVIDIOSO is free software."

OBJ=obj
obj_folder=("src" "base" "constraints" "core" "FZ_parser" "search" "test")

#Create obj (main) folder
if [ ! -d ${OBJ} ];
  mkdir ${OBJ}
fi

#Create obj folders
#@note ${#ArrayName[@]}: length of ArrayName
for ((k=0; k<${#obj_folder[@]}; k++))
do
  if [ -d ${OBJ}/${obj_folder[$k]} ]; then
    echo ${OBJ}/${obj_folder[$k]} "already exists"
  else
    mkdir ${OBJ}/${obj_folder[$k]}
  fi
done

make clean
make
