//
//  logger.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Modified by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Logger class used for log messages, prints and errors.
//  @note Logger is a singleton class. To handle different output files 
//        (e.g., iNVIDOSO standard logs and unit test logs), the output streams
//        are duplicated if the system is build in unit test mode.
//  @todo Use Multiton pattern to create two (or more) different singleton instaces of logger:
//        one for iNVIDOSO logs and the other for Unit Test.
//

#ifndef NVIDIOSO_logger_h
#define NVIDIOSO_logger_h

#if WINDOWS_REL
  #include <windows.h>
#else
  #include <unistd.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <algorithm>
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

// Logging message for unit test
#define LogMsgUT logger.set_log_ut(); logger

// Error and Warning MACROs
#define LogMsgE logger << "[ERROR] "
#define LogMsgW logger << "[WARNING] "

//! logger instance: global to all classes if included
class Logger;
extern Logger& logger;

class Logger {
private:

  // Variables for logging 
  bool _verbose;
  std::string _out_log_file;
  std::stringstream _ss_log;

  time_t _raw_time;
  
 #if WINDOWS_REL
    struct tm _time_info;
 #else
    struct tm * _time_info;
 #endif
 
  //! Output log stream (to stdout)
  std::ostream& _out;
  
  //! Output log stream (to file)
  std::ofstream _of_stream;

#if UNIT_TEST 
  std::string   _std_log_file;
  std::ofstream _std_of_stream;
  
  //! Flag logging for unit test
  bool _log_ut;
#endif
 
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
  	if ( (std::find ( str.begin(), str.end(), '\n' ) != str.end()) || flush )
  	{
  		_ss_log.str("");
      
#if UNIT_TEST 
      std::string to_file;
      if ( _log_ut ) 
      {
        to_file = str;
      }
      else
      {
        to_file = "[" +  get_time_stamp() + "]: " + str;
      }
#else
      std::string to_file = "[" +  get_time_stamp() + "]: " + str;
#endif
  			
  		if ( _verbose )
  		{// If unit test, print on std out only Unit Test logs
      
#if UNIT_TEST 
        if ( _log_ut ) 
        {
          _out << str;
        }  
#else
        _out << str;
#endif  
  		}
  		
  		if ( flush )
      {
  			to_file += "\n";
      }
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
#if UNIT_TEST 
      if ( _log_ut ) 
      {
          _of_stream << v;
          set_log_ut ( false );
      }
      else
      {// Standard output on non log file
          if ( !_std_of_stream.is_open() )
          {
            _std_of_stream.open( _std_log_file );
          }
          _std_of_stream << v;
      }
#else
      _of_stream << v;
#endif
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
 
#if UNIT_TEST 
  inline void set_log_ut ( bool log_ut = true )
  {
    _log_ut = log_ut;
  }
#endif

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
    
#if UNIT_TEST 
  bool to_ut = _log_ut;
#endif 
  	// Flush the stream
  	log<std::string> ( "", true );
  	
  	if ( _verbose )
  	{
#if UNIT_TEST 
      if ( to_ut )
      {
        F ( _out );
      }	
#else
      F ( _out );
#endif 
  	}
    
    return *this;
  }//<<
  
  // Set method
  void set_out_file ( std:: string );
  void set_verbose  ( bool v );
  
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
