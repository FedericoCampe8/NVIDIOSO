//
//  logger.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Logger class used for log messages, prints and errors.
//

#ifndef NVIDIOSO_logger_h
#define NVIDIOSO_logger_h

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#if CUDAON
#include <cuda_runtime_api.h>
#endif 

class Logger;
//! logger instance: global to all classes if included
extern Logger* logger;

class Logger {
  
private:
  //! Static (singleton) instance of logger
  static Logger * _l_instance;
  // Variables for loggin
  bool _verbose;
  std::string _out_log_file;
  time_t _rawtime;
  struct tm * _timeinfo;
  
  //! Output log stream (to file)
  std::ofstream  * _of_stream;
protected:
  Logger ( std::string="" );
  
public:
  ~Logger();
  
  //! Constructor get (static) instance
  static Logger* get_instance ( std::string log_file="" ) {
    if ( _l_instance == nullptr ) {
      _l_instance = new Logger ( log_file );
    }
    return _l_instance;
  }//get_instance
  
  // Set method
  void set_out_file ( std:: string );
  void set_verbose  ( bool );
  
  //! Print message on stdout or file (print_message force printing)
  void message       ( std::string );
  void print_message ( std::string );
  
  //! Print log on stdout or file
  void log       ( std::string );
  void oflog     ( std::string );
  
  //! Print error message on cerr (optional: __FILE__ and __LINE__)
  void error     ( std::string );
  void error     ( std::string, const char* );
  void error     ( std::string, const char*, const int );
  
#if CUDAON
  /** Error handling for CUDA
   * @return true if there is an error on the device
   */
  bool cuda_handle_error ( cudaError_t err );
  bool cuda_handle_error ( cudaError_t err, const char *file );
  bool cuda_handle_error ( cudaError_t err, const char *file, const int line );
#endif
  
};

#endif
