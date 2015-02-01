//
//  factory_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Factory method for model generators.
//  This class allows the client to select the kinds of
//  data structure to instantiate according to the model
//  to create (e.g., CUDA model, CPU model, Thread-CPU model, etc.)
//

#ifndef NVIDIOSO_factory_generator_h
#define NVIDIOSO_factory_generator_h

#include "globals.h"
#include "model_generator.h"
#include "cuda_model_generator.h"

class FactoryModelGenerator {
  
public:
  //! Get the right instance of a generator based on the input
  static ModelGenerator* get_generator ( GeneratorType gt ) {
    switch ( gt ) {
      case GeneratorType::CUDA:
        return new CudaGenerator ();
      default:
  
        return nullptr;
    }
  }//get_parser
};

#endif



