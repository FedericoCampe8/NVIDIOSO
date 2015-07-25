//
//  logger.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Modified by Luca Da Rin Fioretto on 25/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "logger.h"

using namespace std;

// Global instance
Logger& logger = Logger::get_instance ( std::cout );

Logger::Logger ( ostream& out, std::string log_file ) : 
	_out ( out ) {
  	if ( log_file.compare( "" ) == 0 )
  	{
    	_out_log_file = "invidioso_" + get_time_stamp () + ".log";
  	}
  	else
  	{
    	_out_log_file = log_file;
  	}
  	_of_stream.open( _out_log_file );
  	_verbose   = false;
}//Logger

Logger::~Logger () {
  if ( _of_stream.is_open() )
  {
    _of_stream.close();
  }
}//~Logger

void
Logger::set_out_file( string out_file ) {
  _out_log_file = out_file;
  if ( _of_stream.is_open() )
  {
    _of_stream.close();
  }
  _of_stream.open( _out_log_file );
}//set_out_file

void
Logger::set_verbose ( bool verbose ) {
  _verbose = verbose;
}//set_verbose

void
Logger::message ( std::string msg ) 
{
  if ( !_verbose ) return;
  cout << msg << endl;
  if ( _out_log_file.compare( "" ) != 0 ) {
    oflog<std::string> ( msg + "\n" );
  }
}//message

void
Logger::print_message ( std::string msg ) {
  cout << msg << endl;
}//print_message

string
Logger::get_time_stamp ()
{
  	_raw_time  = std::time ( nullptr );
  	_time_info = std::localtime ( &_raw_time );
  	string yy  = to_string ( _time_info->tm_year + 1900 );
  	string mm  = to_string ( _time_info->tm_mon  + 1    );
  	
  	string
  	yymmdd = __DATE__;
  	yymmdd += "-";
  	yymmdd += __TIME__;
  	
  	return yymmdd;
}//get_time_stamp

// Print error message on cerr
void
Logger::error ( std::string err ) {
  cerr << err << "\n";
  if ( _out_log_file.compare( "" ) != 0 ) {
    oflog<std::string> ( err + ".\n" );
  }
}//error

void
Logger::error ( std::string err, const char* file ) {
  cerr << err << " - At " << file << ".\n";
  if ( _out_log_file.compare( "" ) != 0 ) {
    oflog<std::string> ( err + " - At " + file + ".\n" );
  }
}//error

void
Logger::error ( std::string err, const char* file, const int line ) {
  cerr << err << " - At " << file << ", line " << line << ".\n";
  if ( _out_log_file.compare( "" ) != 0 ) {
    ostringstream convert;
    convert << line;
    oflog<std::string> ( err + " - At " + file + ", line " + convert.str() + ".\n" );
  }
}//error


#if CUDAON
bool
Logger::cuda_handle_error ( cudaError_t err  ) {
  if ( _verbose && (err != cudaSuccess) )
    cout << cudaGetErrorString( err ) << ".\n";
  if (err != cudaSuccess) return true;
  return false;
}//error

bool
Logger::cuda_handle_error ( cudaError_t err, const char *file ) {
  if ( _verbose && (err != cudaSuccess) )
    cerr << cudaGetErrorString( err ) << " - At " << file << ".\n";
  
  if (err != cudaSuccess) return true;
  return false;
}//error

bool
Logger::cuda_handle_error ( cudaError_t err, const char *file, const int line ) {
  if ( _verbose && (err != cudaSuccess) )
    cerr << cudaGetErrorString( err ) << " - At " << file << ", line " << line << ".\n";
  
  if (err != cudaSuccess) return true;
  return false;
}//error
#endif



