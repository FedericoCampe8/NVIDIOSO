//
//  main.cpp
//  iNVIDIOSO1.0
//
//  Created by Federico Campeotto on 26/06/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "globals.h"
#include "input_data.h"
#include "cp_model_engine.h"
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
	
		// Instantiate a model engine
		CPModelEngine data_engine;
		
  	// Load model
  	if ( !data_engine.load_model ( i_data.get_in_file() ) )
  	{
  		LogMsg << dbg << "Failed to load the model " << i_data.get_in_file() << std::endl; 
  		exit ( 2 );
  	}
  	
  	// Init store (variables, domains, and constraints)
  	try 
  	{
  		LogMsg << dbg << "Model Initialization" << std::endl;
  		data_engine.initialize_model ();
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
  
		std::unique_ptr<CPSolver> cp_solver ( new CPSolver ( std::move ( data_engine.get_model() ) ) );
  	if ( cp_solver == nullptr ) 
  	{
  		LogMsg << dbg << "Failed to create the constraint model" << std::endl;
  		exit( 4 );
  	}
  
  	statistics.stopwatch ( Statistics::TIMING::PREPROCESS );
  
  	LogMsg << dbg + "CP model created" << std::endl;
  
  	/***************************************
   	 *              RUN SOLVER             *
   	 ***************************************/
		LogMsg << dbg + "Run solver" << std::endl;
  
  	statistics.set_timer ( Statistics::TIMING::SEARCH );
  	cp_solver->run();
  	statistics.stopwatch ( Statistics::TIMING::SEARCH );
  	
  	LogMsg << dbg + "Solver done" << std::endl;
  
  	// Print statistics
  	statistics.stopwatch ( Statistics::TIMING::ALL );
  	if ( i_data.verbose() ) statistics.print ();
  
  	/***************************************
   	 *            CLEAN AND EXIT           *
   	***************************************/
   	LogMsg << dbg << "Exit" << std::endl;

  	return 0;
}

