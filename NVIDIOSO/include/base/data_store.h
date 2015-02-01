//
//  data_store.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  DataStore class:
//  interface for loading the CP model and creates all the
//  data structures needed for computations.
//  It istantiates the objects (variables, constraints, etc).
//

#ifndef NVIDIOSO_data_store_h
#define NVIDIOSO_data_store_h

#include "globals.h"
#include "cp_model.h"

class DataStore {
  
protected:
  // Options for Data_store.
  bool _timer;
  bool _verbose;
  std::string _dbg;
  std::string _in_file = "";
  
  //! CP Model
  CPModel * _cp_model;
  
  /**
   * Constructor.
   * @param in_file file path of the model to parse.
   */
  DataStore ( std::string in_file );
  
public:
  virtual ~DataStore();
  
  /**
   * Load model from input file (FlatZinc model).
   * @note: the model described as a set of tokens is
   * stored in the Tokenization class used by the parser.
   * The parser has access to the set of tokens and it
   * manages them in order to retrieve the correct set
   * of tokens to initialize variables, and constraints.
   * See Parser interface.
   */
  virtual bool load_model ( std::string= "" ) = 0;
  
  //! Init model using the information read from files
  virtual void init_model () = 0;
  
  //! Print info about the model
  virtual void print_model_info () = 0;
  
  //! Get the instantiated model
  virtual CPModel * get_model ();
  
  // Other info methods
  virtual void print_model_variable_info   ();
  virtual void print_model_domain_info     ();
  virtual void print_model_constraint_info ();
};


#endif
