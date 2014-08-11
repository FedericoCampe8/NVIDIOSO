//
//  statistics.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 17/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "statistics.h"

using namespace std;

// Init static variable
Statistics* Statistics::_s_instance = nullptr;

//Global instance
Statistics* statistics = Statistics::get_instance ();

Statistics::Statistics () :
_dbg        ( "Statistics - " ) {
  for ( int i = 0; i < MAX_T_TYPE; i++ ) {
    _time       [ i ] = 0;
    _stop_watch [ i ] = false;
  }
}//Statistics

Statistics::~Statistics () {
}//~Statistics

void
Statistics::set_timer () {
  for ( int i = 0; i < MAX_T_TYPE; i++ ) _stop_watch [ i ] = false;
  gettimeofday( &_time_stats, nullptr );
  _time_start = _time_stats.tv_sec+( _time_stats.tv_usec / USEC );
}//set_timer

void
Statistics::set_timer ( int tt ) {
  if ( tt < 0 || tt >= MAX_T_TYPE ) return;
  
  _stop_watch [ tt ] = false;
  gettimeofday( &_time_stats, nullptr );
  _time[ tt ] = _time_stats.tv_sec+( _time_stats.tv_usec / USEC );
}//set_timer

void
Statistics::stopwatch ( int tt ) {
  if ( tt < 0 || tt >= MAX_T_TYPE ) return;
  
  // Get current time
  gettimeofday( &_time_stats, nullptr );
  
  // Stop watch set
  _stop_watch [ tt ] = true;
  
  // Set time in the store
  if ( _time[ tt ] ) {
    _time[ tt ] =
    _time_stats.tv_sec+(_time_stats.tv_usec/1000000.0) - _time[ tt ];
  }
  else {
    _time[ tt ] =
    _time_stats.tv_sec+(_time_stats.tv_usec/1000000.0) - _time_start;
  }
}//stopwatch

void
Statistics::stopwatch_and_add ( int tt ) {
  if ( tt < 0 || tt >= MAX_T_TYPE ) return;
  
  // Get current time
  gettimeofday( &_time_stats, nullptr );
  
  // Stop watch set
  _stop_watch [ tt ] = true;
  
  // Set time in the store
  if ( _time[ tt ] ) {
    _time[ tt ] +=
    _time_stats.tv_sec+(_time_stats.tv_usec/1000000.0) - _time[ tt ];
  }
  else {
    _time[ tt ] +=
    _time_stats.tv_sec+(_time_stats.tv_usec/1000000.0) - _time_start;
  }
}//stopwatch

double
Statistics::get_timer ( int tt ) {
  if ( tt < 0 || tt >= MAX_T_TYPE ) return 0;
  
  // Check if the watch was already set
  if ( !_stop_watch [ tt ] ) {
    stopwatch ( tt );
    _stop_watch [ tt ] = true;
  }
  
  return  _time[ tt ];
}//get_timer

void
Statistics::print () const {
  cout << "\t============ NVIDIOSO Statistics ============\n";
  cout << "\t\tInitialization time: " << _time[ T_PREPROCESS ] << " sec.\n";
  cout << "\t\tSearch time:         " << _time[ T_SEARCH ]     << " sec.\n";
  cout << "\t\tTotal time:          " << _time[ T_ALL ]        << " sec.\n";
  cout << "\t---------------------------------------------\n";
  
}//print

