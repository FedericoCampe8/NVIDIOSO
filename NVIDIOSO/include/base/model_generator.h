//
//  model_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
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

enum class GeneratorType {
  CUDA,
  CPU,
  THREAD_CPU,
  OTHER
};

class ModelGenerator {
  
public:
  virtual ~ModelGenerator () {};
  
  /**
   * These methods create the instances of the
   * objects and return the correspondent 
   * (shared) pointers to them.
   * @param TokenPtr pointer to the token describing a variable.
   *        If the token does not correspond to the object to
   *        instantiate, it returns nullptr.
   */
  virtual VariablePtr     get_variable      ( TokenPtr ) = 0;
  
  /**
   * These methods create the instances of the
   * objects and return the correspondent
   * (shared) pointers to them.
   * @param TokenPtr pointer to the token describing a constraint.
   *        If the token does not correspond to the object to
   *        instantiate, it returns nullptr.
   */
  virtual ConstraintPtr   get_constraint    ( TokenPtr ) = 0;
  
  /**
   * These methods create the instances of the
   * objects and return the correspondent
   * (shared) pointers to them.
   * @param TokenPtr pointer to the token describing a search engine.
   *        If the token does not correspond to the object to
   *        instantiate, it returns nullptr.
   */
  virtual SearchEnginePtr get_search_engine ( TokenPtr ) = 0;
  
  /**
   * These methods create the instances of the
   * objects and return the correspondent
   * (shared) pointers to them.
   * @param TokenPtr pointer to the token describing a search engine.
   *        If the token does not correspond to the object to
   *        instantiate, it returns nullptr.
   */
  virtual ConstraintStorePtr get_store () = 0;
  
};



#endif
