//
//  statistics.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 17/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
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
#include <chrono>
#include <ctime>

#define constexpr const

//! Statistics's class instance (declaration): global to all classes including this .h
class Statistics;
extern Statistics& statistics;

class Statistics {    
public:
    enum class TIMING
    {
        GENERAL,
            SEARCH,
            FIRST_SOL,
            PREPROCESS,
            FILTERING,
            BACKTRACK,
            ALL,
            Count
    };
    
    virtual ~Statistics ();
    Statistics ( const Statistics& other )            = delete;
    Statistics& operator= ( const Statistics& other ) = delete;

    //! Get (static) instance (singleton) of Statistics
    static Statistics& get_instance () {
        static Statistics statistic;
        return statistic;
    }//get_instance
  
    //! Set timer (starts "watching" the running time)
    void set_timer   ();
  
    /**
     * Set timer for a given computation which will be observed.
     * @param tt describes which kind of computation will be observed.
     */
    void set_timer ( TIMING t );
  
    /**
     * Stop watching the running time.
     * @param tt describes which kind of computation has been observed.
     */
    void stopwatch ( TIMING t = TIMING::GENERAL );
  
    /**
     * Stop watching the running time and add the time to the previous
     * times watched for tt.
     * @param tt describes which kind of computation has been observed.
     */
    void stopwatch_and_add ( TIMING t = TIMING::GENERAL );
  
    /**
     * Get the value of the running time in seconds.
     * @param tt describes which kind of computation time
     *        must be returned,
     * @return the computational time related to tt in seconds.
     */
    double get_timer ( TIMING t = TIMING::GENERAL );
  
    //!Print info about statistics on the program.
    virtual void print () const;

protected:

    //! Max size of the array of times.
    static constexpr int MAX_T_TYPE = 100;

    //! Debug string info.
    std::string _dbg;

    // Times parameters used for statistics
    std::chrono::time_point<std::chrono::system_clock> _time_start;

    //! Computational times are recorded here.
    double _time [ MAX_T_TYPE ];

    //! Partial times (i.e., from set timer to stop watch) are recorded here.
    std::chrono::time_point<std::chrono::system_clock>  _partial_time [ MAX_T_TYPE ];

    //! States if a watching has been stopped for a given computation.
    bool _stop_watch [ MAX_T_TYPE ];

    /**
     * Converter from TIMING enum values to
     * int values.
     * @param t TIMING value
     * @return integer value mapping t, or -1 if no
     *         mapping exists for the given t.
     * @note This is done to decouple TIMING with integer indeces.
     */
    virtual int timing_to_int ( Statistics::TIMING t ) const;

    /**
     * Converter from integer to TIMING values.
     * This is mostly used for ease of implementation
     * and decoupling from TIMING class and integer indeces for arrays.
     * @param i integer value
     * @return TIMING value corresponding to i or count if
     *         if no mapping exist for the given i.
     */
    virtual Statistics::TIMING int_to_timing ( int i ) const;

    /*
     * Protected constructor to avoid instantiate more
     * than one instante of Statistics.
     */
    Statistics  ();
    
};


#endif
