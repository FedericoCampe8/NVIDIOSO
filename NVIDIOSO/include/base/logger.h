//
//  logger.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
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
#include <typeinfo>

#if CUDAON
#include <cuda_runtime_api.h>
#endif

#if GCC4
#define nullptr NULL
#endif

#define LOG_PREFIX std::string ("iNVIDIOSO")

#define LogMsg logger

//! logger instance: global to all classes if included
class Logger;
extern Logger& logger;

class Logger {
private:

  // Variables for log
  bool _verbose;
  std::string _out_log_file;
  std::stringstream _ss_log;

  time_t _raw_time;
  struct tm * _time_info;
  
  //! Output log stream (to stdout)
  std::ostream&    _out;
  
  //! Output log stream (to file)
  std::ofstream _of_stream;
  
protected:
  	Logger ( std::ostream& out, std::string = "" );
  
  	//! Time stamp
	virtual std::string get_time_stamp ();
	
  //! Print log on stdout or file
  template<typename T>
  void log  ( const T& v, bool flush = false )
  {
  	_ss_log << v;
  	std::string str = _ss_log.str();
  	if ( (find ( str.begin(), str.end(), '\n' ) != str.end()) || flush )
  	{
  		_ss_log.str("");
  		std::string to_file = "[" +  get_time_stamp() + "]: " + str;
  			
  		if ( _verbose )
  		{
  			_out << str;
  		}
  		
  		if ( flush )
  			to_file += "\n";
  		oflog<std::string> ( to_file );
  	}
  }//log
  
  template<typename T>
  void oflog ( const T& v )
  { 
    // Sanity check
    if ( _out_log_file.compare( "" ) == 0 )
    {
      	std::cerr << "No log file found\n";
      	return;
    }
    
    if ( !_of_stream.is_open() )
    {
      _of_stream.open( _out_log_file );
    }
    if ( _of_stream.is_open() )
    {
      _of_stream << v;
    }
    else
    {
      std::cerr << "Can't open file " << _out_log_file << " - At "
      << __FILE__ << __LINE__ << std::endl;
    }
  }//oflog
  
  	
public:
  virtual ~Logger();
  
  Logger ( const Logger& other )            = delete;
  Logger& operator= ( const Logger& other ) = delete;
  
  //! Constructor get (static) instance
  static Logger& get_instance ( std::ostream& out, std::string log_file="" ) 
  {
    static Logger log_instance ( out, log_file );
    return log_instance;
  }//get_instance
  
  // Input operator
  template<typename T>
  Logger& operator<< ( const T& v )
  {
	// Store log
    log<T> ( v );
    return *this;
  }//<<
  
  // Input operator
  Logger const& operator<< ( std::ostream& (*F)(std::ostream&) )
  {
  	// Flush the stream
  	log<std::string> ( "", true );
  	
  	if ( _verbose )
  	{
  		F ( _out );
  	}
    
    return *this;
  }//<<
  
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
