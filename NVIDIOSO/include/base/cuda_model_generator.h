//
//  cuda_model_generator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/09/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Generates the objects for the model w.r.t. to a CUDA implementation.
//  It specializes the right data structures (e.g. domains) for CUDA.
//

#ifndef NVIDIOSO_cuda_model_generator_h
#define NVIDIOSO_cuda_model_generator_h

#include "model_generator.h"

class BaseConstraintRegister;
class GlobalConstraintRegister;
extern BaseConstraintRegister&   bse_constraint_register;
extern GlobalConstraintRegister& glb_constraint_register;

class CudaGenerator : public ModelGenerator {
private:

	//! Objective variable if any
	VariablePtr _obj_var;
	
	//! This hash table is used to link (string) ids to aux arrays.
    std::unordered_set<std::string>  _arr_lookup_table;
    
    //! This hash table is used to link (string) ids to variables.
    std::unordered_map<std::string, VariablePtr>  _var_lookup_table;
  
protected:
    std::string _dbg;
  
public:
    CudaGenerator  ();
    ~CudaGenerator ();
  	
  	//! See "model_generator.h"
    std::pair < std::string, std::vector< int > > get_auxiliary_parameters ( UTokenPtr );
    
    //! See "model_generator.h"
    VariablePtr get_variable ( UTokenPtr );
  
    //! See "model_generator.h"
    ConstraintPtr get_constraint ( UTokenPtr );
  
    //! See "model_generator.h"
    SearchEnginePtr get_search_engine ( UTokenPtr );
  
    //! See "model_generator.h"
    ConstraintStorePtr get_store ( UTokenPtr );
};

#endif

