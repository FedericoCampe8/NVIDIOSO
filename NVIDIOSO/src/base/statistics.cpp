//
//  statistics.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 17/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "statistics.h"

using namespace std;

//Global instance
Statistics& statistics = Statistics::get_instance ();

Statistics::Statistics () :
    _dbg ( "Statistics - " ) {
    for ( int i = 0; i < MAX_T_TYPE; i++ )
    {
        _stop_watch   [ i ] = false;
    }
}//Statistics

Statistics::~Statistics () {
}//~Statistics

int
Statistics::timing_to_int ( TIMING t ) const
{
    switch ( t )
    {
        case TIMING::GENERAL:
            return 0;
        case TIMING::SEARCH:
            return 1;
        case TIMING::FIRST_SOL:
            return 2;
        case TIMING::PREPROCESS:
            return 3;
        case TIMING::FILTERING:
            return 4;
        case TIMING::BACKTRACK:
            return 5;
        case TIMING::ALL:
            return 6;
        case TIMING::Count:
            return 7;
        default:
            return -1;
    }
}//timing_to_int

Statistics::TIMING
Statistics::int_to_timing ( int i ) const
{
    if ( i < 0 || i > timing_to_int ( TIMING::Count ) )
        return TIMING::Count;
    switch ( i )
    {
        case 0:
            return TIMING::GENERAL;
        case 1:
            return TIMING::SEARCH;
        case 2:
            return TIMING::FIRST_SOL;
        case 3:
            return TIMING::PREPROCESS;
        case 4:
            return TIMING::FILTERING;
        case 5:
            return TIMING::BACKTRACK;
        case 6:
            return TIMING::ALL;
        default:
            return TIMING::Count;
    }
}//int_to_timing

void
Statistics::set_timer ()
{
    _time_start = std::chrono::system_clock::now();

    for ( int i = 0; i < MAX_T_TYPE; i++ )
    {
        _stop_watch [ i ]   = false;
        _partial_time [ i ] = _time_start;
    }
}//set_timer

void
Statistics::set_timer ( TIMING t )
{
    int idx = timing_to_int ( t );

    // Sanity check
    if ( idx < 0 || idx >= MAX_T_TYPE ) return;
  
    _stop_watch [ idx ] = false;
    std::chrono::time_point<std::chrono::system_clock> p_time;
    p_time = std::chrono::system_clock::now();
  
    _partial_time[ idx ] = p_time;
}//set_timer

void
Statistics::stopwatch ( TIMING t )
{
    int idx = timing_to_int ( t );
    
    // Get current time
    std::chrono::time_point<std::chrono::system_clock> end;
    end = std::chrono::system_clock::now();
    
    // Set time in the store
    if ( !_stop_watch [ idx ] )
    {
        std::chrono::duration<double> elapsed_seconds = end - _partial_time[ idx ];
        _time[ idx ] = elapsed_seconds.count();

        // Stop watch set
        _stop_watch [ idx ] = true;
    }
    else
    {
        cerr << _dbg + "Set timer before stop watch." << endl;
    }
}//stopwatch

void
Statistics::stopwatch_and_add ( TIMING t )
{
    // Get current time
    std::chrono::time_point<std::chrono::system_clock> curr_time;
    curr_time = std::chrono::system_clock::now();

    int idx = timing_to_int ( t );

    // Set time in the store
    if ( !_stop_watch [ idx ] )
    {
        std::chrono::duration<double> elapsed_seconds = curr_time - _partial_time[ idx ];
        _time[ idx ] += elapsed_seconds.count();
    
        // Stop watch set
        _stop_watch [ idx ] = true;
    }
    else
    {
        cerr << _dbg + "Set timer before stop watch." << endl;
    }
}//stopwatch

double
Statistics::get_timer ( TIMING t  )
{
    int idx = timing_to_int ( t );

    // Sanity check
    if ( idx < 0 || idx >= MAX_T_TYPE ) return -1;
    
    // Check if the watch was already set
    if ( !_stop_watch [ idx ] )
    {
        stopwatch ( t );
    }
    
    return  _time[ idx ];
}//get_timer

void
Statistics::print () const
{
    cout << "\t============ iNVIDIOSO Statistics ============\n";
    if ( _time[ timing_to_int ( TIMING::PREPROCESS ) ] )
        cout << "\t\tInitialization time: " << _time[ timing_to_int ( TIMING::PREPROCESS ) ] << " sec.\n";
    if ( _time[ timing_to_int ( TIMING::SEARCH ) ] )
        cout << "\t\tSearch time:         " << _time[ timing_to_int ( TIMING::SEARCH ) ]     << " sec.\n";
    if ( _time[ timing_to_int ( TIMING::FILTERING ) ] )
        cout << "\t\tFiltering time:      " << _time[ timing_to_int ( TIMING::FILTERING ) ]  << " sec.\n";
    if ( _time[ timing_to_int ( TIMING::BACKTRACK ) ] )
        cout << "\t\tBacktrack time:      " << _time[ timing_to_int ( TIMING::BACKTRACK ) ]  << " sec.\n";
    cout << "\t\tTotal time:          " << _time[ timing_to_int ( TIMING::ALL ) ]        << " sec.\n";
    cout << "\t---------------------------------------------\n";
}//print

