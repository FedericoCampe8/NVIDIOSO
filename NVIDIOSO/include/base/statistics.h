//
//  statistics.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 17/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Statistic class use to measure and produce statistics and
//  other information about the (run-time) system.
//

#ifndef NVIDIOSO_statistics_h
#define NVIDIOSO_statistics_h

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

class Statistics;

//! logger instance: global to all classes if included
extern Statistics* statistics;

class Statistics {
private:
  //! Static (singleton) instance of Statistics.
  static Statistics * _s_instance;
  
protected:
  
  //! USEC unit
  static constexpr double USEC = 1000000.0;
  
  //! Max size of the array of times.
  static constexpr int MAX_T_TYPE = 10;
  
  //! Debug string info.
  std::string _dbg;
  
  // Times parameters used for statistics
  timeval _time_stats;
  double  _time_start;
  
  //! Computational times are recorded here.
  double _time [ MAX_T_TYPE ];
  
  //! Partial times (i.e., from set timer to stop watch) are recorded here.
  double _partial_time [ MAX_T_TYPE ];
  
  //! States if a watching has been stopped for a given computation.
  bool _stop_watch [ MAX_T_TYPE ];
  
  /*
   * Protected constructor to avoid instantiate more 
   * than one instante of Statistics.
   */
  Statistics  ();
  
public:
  static constexpr int T_GENERAL    = 0;
  static constexpr int T_SEARCH     = 1;
  static constexpr int T_FIRST_SOL  = 2;
  static constexpr int T_PREPROCESS = 3;
  static constexpr int T_FILTERING  = 4;
  static constexpr int T_BACKTRACK  = 5;
  static constexpr int T_ALL        = 6;
  
  ~Statistics ();
  
  //! Get (static) instance (singleton) of Statistics
  static Statistics* get_instance () {
    if ( _s_instance == nullptr ) {
      _s_instance = new Statistics ();
    }
    return _s_instance;
  }//get_instance
  
  //! Set timer (starts "watching" the running time)
  void set_timer   ();
  
  /**
   * Set timer for a given computation which will be observed.
   * @param tt describes which kind of computation will be observed.
   */
  void set_timer ( int tt );
  
  /**
   * Stop watching the running time.
   * @param tt describes which kind of computation has been observed.
   */
  void stopwatch ( int tt = T_GENERAL );
  
  /**
   * Stop watching the running time and add the time to the previous
   * times watched for tt.
   * @param tt describes which kind of computation has been observed.
   */
  void stopwatch_and_add ( int tt = T_GENERAL );
  
  /**
   * Get the value of the running time in seconds.
   * @param tt describes which kind of computation time
   *        must be returned,
   * @return the computational time related to tt in seconds.
   */
  double get_timer ( int tt = T_GENERAL );
  
  //!Print info about statistics on the program.
  virtual void print () const;
};


#endif
