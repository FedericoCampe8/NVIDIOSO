//
//  model_engine.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Modified by Federico Campeotto on 09/14/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  ModelEngine class:
//  interface of an engine which main task is to load a (constraint) model from
//  input (file) and generate the correspondent internal model, i.e., an instance of
//  a model which as variables, constraints, search engine, etc.
//  The data flow is the following:
//    input -> load_model() - ( -> parse input -> ) -> init_model() - ( -> instantiate objects -> ) -> get_model()
//

#ifndef NVIDIOSO_model_engine_h
#define NVIDIOSO_model_engine_h

#include "globals.h"
#include "cp_model.h"

class ModelEngine {
public:
  virtual ~ModelEngine() {};
  
  /**
   * Load model from input: this method loads the model from 
   * the given input, i.e., it parses the input producing a set
   * of tokens which will be used later to initialize the model 
   * by creating the corresponding object instances.
   * @param input_path path to the file to load.
   * @param input_name string representing a "name-friendly" identifier
   *        for the model to load. 
   * @return true if the model has been loaded correctly, false other wise.
   * @note see Parser.h for interface of the parser used to create the
   *       set of tokens corresponding to the elements read from input.
   */
  virtual bool load_model ( std::string input_path, std::string input_name ) = 0;
  
  /**
   * Initilize the model using the information read from input.
   * @param input_name name of the model to initialize.
   * @return true if the model has been initialized, false otherwise.
   */
  virtual bool initialize_model ( std::string input_name ) = 0;
  
  /**
   * Get the instance of a previously initialized model.
   * @param model_name name of the model to return.
   * @return a (unique) pointer to the initialized model.
   *         returns NULL if model_name is not present.
   */
  virtual CPModelUPtr get_model ( std::string model_name ) = 0;
  
  //! Print info about the engine
  virtual void print () const = 0;
};


#endif
