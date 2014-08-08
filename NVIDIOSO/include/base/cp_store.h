//
//  csp_store.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class is in charge of defining a data store for a CP model.
//  It uses a parser to parse CP models and to allocate data structures
//  for searching for solutions.
//  With the information parsed from file it creates a CPModel.
//

#ifndef NVIDIOSO_cp_store_h
#define NVIDIOSO_cp_store_h

#include "data_store.h"
#include "parser.h"

class CPStore : public DataStore {
private:
  
  //! Static (Singleton) instance
  static CPStore * _cp_ds_instance;
  
  //! Parser instance
  Parser * _parser;
  
protected:
  
  //! Protected constructor for singleton pattern
  CPStore ( std::string );
  
public:
  ~CPStore();
  
  //! Constructor get (static) instance
  static CPStore* get_store ( std::string in_file ) {
    if ( _cp_ds_instance == nullptr ) {
      if ( in_file.compare ( "" ) == 0 ) {
        logger->error( "DataStore: No input file.", __FILE__, __LINE__ );
        return nullptr;
      }
      _cp_ds_instance = new CPStore ( in_file );
    }
    return _cp_ds_instance;
  }//get_instance
  
  //! Load model from input file (FlatZinc model)
  virtual bool load_model ( std::string= "" );
  
  /**
   * Init store with the loaded model.
   * This method works on the internal state of the store.
   * It uses a generator to generate the right instances of the objects
   * (e.g. CUDA-FD variabes) and add them to the model.
   * A generator takes tokens as input and returns the corresponding
   * pointer to the instantiated objects.
   */
  virtual void init_model ();
  
  // Print info about the model
  virtual void print_model_info ();
  virtual void print_model_variable_info   ();
  virtual void print_model_domain_info     ();
  virtual void print_model_constraint_info ();
};


#endif
