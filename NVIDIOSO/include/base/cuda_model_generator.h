//
//  cuda_model_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Generates the objects for the model w.r.t. to a CUDA implementation.
//  It specializes the right data structures (e.g. domains) for CUDA.
//

#ifndef NVIDIOSO_cuda_model_generator_h
#define NVIDIOSO_cuda_model_generator_h

#include "model_generator.h"

class CudaGenerator : public ModelGenerator {
private:
  //! This map is used to link strings ids with variables.
  std::map<std::string, VariablePtr>  _var_lookup_table;
  
protected:
  std::string _dbg;
  
public:
  CudaGenerator  ();
  ~CudaGenerator ();
  
  //! See "model_generator.h"
  VariablePtr     get_variable      ( TokenPtr );
  
  //! See "model_generator.h"
  ConstraintPtr   get_constraint    ( TokenPtr );
  
  //! See "model_generator.h"
  SearchEnginePtr get_search_engine ( TokenPtr );
  
};

#endif

