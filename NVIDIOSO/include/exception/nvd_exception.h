//
//  nvd_exception.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 18/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class extends the excepion C++ class to handle exceptions.
//


#ifndef NVIDIOSO_nvd_exception_h
#define NVIDIOSO_nvd_exception_h

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#if GCC4
#define nullptr NULL
#define noexcept throw()
#endif

class NvdException : public std::exception {
protected:
  
  //! Code line where the exception was thrown
  int _expt_line;
  
  //! Name of the file where the exception was thrown
  std::string _expt_file;
  
  //! Exception message
  std::string _expt_message;
  
public:
  /**
   * Constructor.
   * @param msg the message related to the exception.
   */
  NvdException ( const char* msg = "" );
  
  /**
   * Constructor.
   * @param msg the message related to the exception.
   * @param file where the excpetion has been raised.
   */
  NvdException ( const char* msg, const char* file );
  
  /**
   * Constructor.
   * @param msg the message related to the exception.
   * @param file where the excpetion has been raised.
   * @param line of code where the excpetion has been raised.
   */
  NvdException ( const char* msg, const char* file, int line );

#if GCC4
    ~NvdException () throw();
#else
    ~NvdException();
#endif
    
  /**
   * Overwrite the what method to print other
   * information about the exception.
   */
  virtual const char* what () const noexcept;
  
};


#endif
