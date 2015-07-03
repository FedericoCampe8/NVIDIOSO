//
//  main.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 26/06/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "globals.h"
#include "input_data.h"
#include "cp_store.h"
#include "cp_solver.h"

int main( int argc, char * argv[] )
{
  	std::string dbg = "Main - ";
  
  	/***************************************
   	 *         INIT DATA/READ INPUT        *
   	 ***************************************/
  	InputData& i_data = InputData::get_instance ( argc, argv );

  	/***************************************
   	 *  LOAD MODEL & INIT DATA STRUCURES   *
   	 ***************************************/
	LogMsg << dbg << "Load Store" << std::endl;
  
  	statistics.set_timer ( Statistics::TIMING::ALL );
  	statistics.set_timer ( Statistics::TIMING::PREPROCESS );

  	// Load model
  	DataStore& d_store = CPStore::get_store ( i_data.get_in_file() );
  	if ( !d_store.load_model() )
  	{
  		LogMsg << dbg << "Failed to load the Model" << std::endl; 
  		exit ( 2 );
  	}
  	
  	// Init store (variables, domains, and constraints)
  	try 
  	{
  		LogMsg << dbg + "Model Initialization" << std::endl;
    	d_store.init_model();
  	} 
  	catch ( std::exception& e ) 
  	{
  		LogMsg << dbg << "Failed to initialize the Model" << std::endl;
  		exit ( 3 );
  	}
  
  	/***************************************
   	 *      	   CREATE MODEL            *
   	 ***************************************/
	LogMsg << dbg + "Instantiate CP solver" << std::endl;
  
  	CPSolver * cp_solver = new CPSolver( d_store.get_model() );
  	if ( cp_solver == nullptr ) 
  	{
  		LogMsg << dbg << "Failed to create the constraint model" << std::endl;
  		exit( 4 );
  	}
  
  	// Set some other user options on the solver
  	cp_solver->customize  ( i_data );
  	statistics.stopwatch ( Statistics::TIMING::PREPROCESS );
  
  	LogMsg << dbg + "CP model created" << std::endl;
  
  	/***************************************
   	 *              RUN SOLVER             *
   	 ***************************************/
	LogMsg << dbg + "Run solver" << std::endl;
  
  	statistics.set_timer ( Statistics::TIMING::SEARCH );
  	cp_solver->run();
  	statistics.stopwatch ( Statistics::TIMING::SEARCH );
  	
  	LogMsg << dbg + "Solver end computation" << std::endl;
  
  	// Print statistics
  	statistics.stopwatch ( Statistics::TIMING::ALL );
  	if ( i_data.verbose() ) statistics.print ();
  
  	/***************************************
   	 *            CLEAN AND EXIT           *
   	***************************************/
   	LogMsg << dbg << "Exit" << std::endl;

  	return 0;
}

