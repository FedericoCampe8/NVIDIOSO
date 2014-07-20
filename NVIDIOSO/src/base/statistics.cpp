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
_dbg        ( "Statistics - " ),
_stop_watch ( false ) {
  for ( int i = 0; i < MAX_T_TYPE; i++ ) _time[ i ] = 0;
}//Statistics

Statistics::~Statistics () {
}//~Statistics

void
Statistics::set_timer () {
  _stop_watch = false;
  gettimeofday( &_time_stats, nullptr );
  _time_start = _time_stats.tv_sec+( _time_stats.tv_usec / USEC );
}//set_timer

void
Statistics::stopwatch ( int tt ) {
  // Get current time
  gettimeofday( &_time_stats, nullptr );
  
  // Stop watch set
  _stop_watch = true;
  
  // Set time in the store
  _time[ tt ] =
  _time_stats.tv_sec+(_time_stats.tv_usec/1000000.0) - _time_start;
}//stopwatch

void
Statistics::stopwatch_and_add ( int tt ) {
  // Get current time
  gettimeofday( &_time_stats, nullptr );
  
  // Stop watch set
  _stop_watch = true;
  
  // Set time in the store
  _time[ tt ] +=
  _time_stats.tv_sec+(_time_stats.tv_usec/1000000.0) - _time_start;
}//stopwatch

double
Statistics::get_timer ( int tt ) {
  // Check if the watch was already set
  if ( !_stop_watch ) {
    stopwatch ( tt );
    _stop_watch = true;
  }
  
  return  _time[ tt ];
}//get_timer

void
Statistics::print () const {
  cout << "\t======= NVIDIOSO Statistics =======\n";
  cout << "\t\tGlobal time:       \t" << _time[ T_GENERAL ]    << " sec.\n";
  cout << "\t\tPreprocessing time:\t" << _time[ T_PREPROCESS ] << " sec.\n";
  cout << "\t\tFiltering time:    \t" << _time[ T_FILTERING ]  << " sec.\n";
  cout << "\t\tSearch time:       \t" << _time[ T_SEARCH ]     << " sec.\n";
  cout << "\t\tSolution time:     \t" << _time[ T_FIRST_SOL ]  << " sec.\n";
  cout << "\t-----------------------------------\n";
  
}//print

