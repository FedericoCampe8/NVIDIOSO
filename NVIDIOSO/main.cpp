//
//  main.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 26/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
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
  InputData * i_data = InputData::get_instance ( argc, argv );
  
  /***************************************
   *  LOAD MODEL & INIT DATA STRUCURES   *
   ***************************************/
  logger->message ( dbg + "Load Store." );
  
  DataStore * d_store = CPStore::get_store ( i_data->get_in_file() );
  
  // Load model
  if ( (d_store == nullptr) || (!d_store->load_model()) ) {
    logger->error ( dbg + "Failed to load the Model." );
    logger->error ( dbg + "Clean and exit." );
    // Clean & exit
    delete d_store;
    delete i_data;
    delete logger;
    exit ( 1 );
  }
  
  // Init store (variables, domains, and constraints)
  try {
    d_store->init_model();
  } catch (...) {
    logger->error ( dbg + "Failed to initialize the msodel." );
    logger->error ( dbg + "Clean and exit." );
    // Clean & exit
    delete d_store;
    delete i_data;
    delete logger;
    exit ( 1 );
  }
  
  logger->message ( dbg + "Init model." );
  
  /***************************************
   *      CREATE CONSTRAINT PROGRAM      *
   ***************************************/
  logger->message ( dbg + "Instantiate CP solver." );
  
  CPSolver * cp_solver = new CPSolver( /*d_store->get_model()*/ );
  if ( cp_solver == nullptr ) {
    logger->error ( dbg + "Failed to create the constraint program." );
    logger->error ( dbg + "Clean and exit." );
    delete d_store;
    delete i_data;
    delete logger;
    exit( 2 );
  }
  
  logger->message ( dbg + "CP model created." );
  
  /***************************************
   *              RUN SOLVER             *
   ***************************************/

  logger->message ( dbg + "Run solver." );
  
  cp_solver->run();
  
  logger->message ( dbg + "End solver computation." );
  
  /***************************************
   *            CLEAN AND EXIT           *
   ***************************************/
  logger->message ( dbg + "Clean objects & Exit." );
  
  delete cp_solver;
  delete d_store;
  delete i_data;
  delete logger;
  return 0;
}

