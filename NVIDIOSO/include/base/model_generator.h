//
//  model_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class generates the objects needed by the (CP) model.
//  It acts as a facade (i.e., see Facade design pattern) for
//  the client.
//  Given the appropriate token it generates the appropriate
//  instance of the object.
//


#ifndef NVIDIOSO_model_generator_h
#define NVIDIOSO_model_generator_h

#include "globals.h"
#include "token.h"
#include "variable.h"
#include "constraint.h"
#include "search_engine.h"
#include "simple_constraint_store.h"

enum class
GeneratorType
{
  CUDA,
  CPU,
  THREAD_CPU,
  OTHER
};

class ModelGenerator {
  
public:
  virtual ~ModelGenerator () {};
  
  /**
   * It returns a pair of <id, array> identifying auxiliary 
   * elements/value/info needed by the solver (e.g., tables for "table constraint"s).
   * Id is the "unique" identifier of the array as declared in the model.
   * @param TokenPtr pointer to the token describing an auxiliary array.
   */
  virtual std::pair < std::string, std::vector< int > > get_auxiliary_parameters ( UTokenPtr ) = 0;
  
  /**
   * It creates the instance of a variable 
   * object given a (pointer to a) token.
   * It returns the correspondent (shared) pointer.
   * @param TokenPtr pointer to the token describing a variable.
   *        If the token does not correspond to the object to
   *        instantiate, it returns nullptr.
   */
  virtual VariablePtr get_variable ( UTokenPtr ) = 0;
  
  /**
   * It creates the instance of a constraint 
   * object given a (pointer to a) token.
   * It returns the correspondent (shared) pointer.
   * @param TokenPtr pointer to the token describing a constraint.
   *        If the token does not correspond to the object to
   *        instantiate, it returns nullptr.
   */
  virtual ConstraintPtr get_constraint ( UTokenPtr ) = 0;
  
  /**
   * It creates the instance of a search engine 
   * object given a (pointer to a) token.
   * It returns the correspondent (shared) pointer.
   * @param TokenPtr pointer to the token describing a search engine.
   *        If the token does not correspond to the object to
   *        instantiate, it returns nullptr.
   */
  virtual SearchEnginePtr get_search_engine ( UTokenPtr ) = 0;
   
  /**
   * It creates the instance of a constraint store
   * object given a (pointer to a) token.
   * It returns the correspondent (shared) pointer.
   * @param TokenPtr pointer to the token describing a constraint store.
   * @note Usually a constraint store is not specified by any token/string in the 
   *       input model. Therefore, usually UTokenPtr is NULL.
   */
  virtual ConstraintStorePtr get_store ( UTokenPtr ) = 0;
  
};



#endif
