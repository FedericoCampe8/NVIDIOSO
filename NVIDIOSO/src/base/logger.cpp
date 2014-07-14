//
//  logger.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "logger.h"

using namespace std;

// Init static variable
Logger* Logger::_l_instance = nullptr;

//Global instance
Logger* logger = Logger::get_instance();

Logger::Logger ( std::string log_file ) {
  if ( log_file.compare( "" ) == 0 ) {
    _out_log_file = "nvidioso.log";
  }
  else {
    _out_log_file = log_file;
  }
  _verbose   = false;
  _of_stream = nullptr;
}//Logger

Logger::~Logger () {
  if ( (_of_stream != nullptr) &&
       (_of_stream->is_open()) ) {
    _of_stream->close();
    delete _of_stream;
  }
}//~Logger

void
Logger::set_out_file( string out_file ) {
  _out_log_file = out_file;
}//set_out_file

void
Logger::set_verbose ( bool verbose ) {
  _verbose = verbose;
}//set_verbose

void
Logger::message ( std::string msg ) {
  if ( !_verbose ) return;
  cout << msg << endl;
  if ( _out_log_file.compare( "" ) != 0 ) {
    oflog ( msg + "\n" );
  }
}//message

void
Logger::print_message ( std::string msg ) {
  cout << msg << endl;
}//print_message

void
Logger::log ( std::string log ) {
  time ( &_rawtime );
  _timeinfo = localtime( &_rawtime );
  cout << "#log (" << asctime ( _timeinfo ) << "):" << log << endl;
}//log

void
Logger::oflog ( std::string log ) {
  // Check output file (path)
  if ( _out_log_file.compare( "" ) == 0 ) {
    cerr << "No path for log file!\n";
    return;
  }
  // Check whether the stream is instantiated
  if ( _of_stream == nullptr ) {
    _of_stream = new ofstream( _out_log_file );
  }
  else if ( !(_of_stream->is_open()) ) {
    _of_stream->open( _out_log_file );
  }
  if ( _of_stream->is_open() ) {
    time ( &_rawtime );
    _timeinfo = localtime( &_rawtime );
    string log_string = "#log (" + ((string) asctime ( _timeinfo )) + "):" + log + "\n";
    _of_stream->operator<<( log_string.c_str() );
    _of_stream->close ();
  }
  else {
    cerr << "Can't open file " << _out_log_file << " - At "
    << __FILE__ << __LINE__ << endl;
  }
}//oflog

// Print error message on cerr
void
Logger::error ( std::string err ) {
  cerr << err << "\n";
  if ( _out_log_file.compare( "" ) != 0 ) {
    oflog ( err + ".\n" );
  }
}//error

void
Logger::error ( std::string err, const char* file ) {
  cerr << err << " - At " << file << ".\n";
  if ( _out_log_file.compare( "" ) != 0 ) {
    oflog ( err + " - At " + file + ".\n" );
  }
}//error

void
Logger::error ( std::string err, const char* file, const int line ) {
  cerr << err << " - At " << file << ", line " << line << ".\n";
  if ( _out_log_file.compare( "" ) != 0 ) {
    ostringstream convert;
    convert << line;
    oflog ( err + " - At " + file + ", line " + convert.str() + ".\n" );
  }
}//error






