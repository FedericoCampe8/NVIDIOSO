//
//  nvd_exception.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 18/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "nvd_exception.h"

using namespace std;

NvdException::NvdException ( const char* msg ) :
_expt_file ( "" ),
_expt_line ( -1 ) {
  _expt_message = msg;
}//NvdException

NvdException::NvdException ( const char* msg, const char* file ) :
NvdException ( msg ) {
  _expt_file = file;
}//NvdException

NvdException::NvdException ( const char* msg, const char* file, int line ) :
NvdException ( msg, file ) {
  _expt_line = line;
}//NvdException

const char*
NvdException::what () const noexcept {
  if ( (_expt_file.compare ( "" ) != 0)  &&
       (_expt_line >= 0) ) {
    stringstream convert;
    convert << _expt_line;
    return
    (_expt_message + " - At " + _expt_file + ", line " + convert.str() + ".").c_str();
    }
    else if ( _expt_file.compare ( "" ) != 0 ) {
      return
      (_expt_message + " - At " + _expt_file + ".").c_str();
    }
    else {
      return _expt_message.c_str();
    }
}//what



