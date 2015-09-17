//
//  cp_model_engine.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Modified by Federico Campeotto on 09/14/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a model_engine for Constraint Programming (CP) models.
//  It uses an internal parser which parse CP models and it allocates data structures
//  for running the solver on the given model(s).
//  This engine has two main subcomponents:
//  1 - Parser: parses input files producing tokens;
//  2 - ModelGenerator: "converts" tokens into internal objects.
//

#ifndef NVIDIOSO_cp_model_engine_h
#define NVIDIOSO_cp_model_engine_h

#include "model_engine.h"
#include "parser.h"

class CPModelEngine : public ModelEngine {
protected:
  //! Debug info
  std::string _dbg;
  
  /**
   * Lookup table of pairs <key, value>, where:
   * - key: string name of the input file to parse
   * - value: (unique) pointer to the parser which parses the file "key"
   * @note this map is used to avoid re-creating and re-parsing the same file
   *       when generating a new model from the same input.
   */
  std::unordered_map < std::string, ParserUPtr > _parser_map;
  
  //! Map of CPModel: key = model name, value = (unique) pointer to a CPModel.
  std::unordered_map < std::string, CPModelUPtr  > _model_map;
  
public:
  
  //! Constructor
  CPModelEngine ();
  
  /**
   * Move constructor is deleted. 
   * This implicitly declares copy constructor and assignment operator as deleted.
   */
  CPModelEngine ( CPModelEngine&& ) = delete;
  
  //! Destructor
  ~CPModelEngine();
  
  /**
   * load_model - see "model_engine.h".
   * @note default name of the model is "model".
   *       Invoking this method without using unique input name will result in  
   *       not loading the model.
   * @note It DOES NOT load the same input if already loaded.
   */
  bool load_model ( std::string input_path, std::string input_name = "model" ) override;
  
  /**
   * initialize_model - see "model_engine.h".
   * @note default name of the model is "model".
   *       Invoking this method without using unique input name will result in  
   *       not initializing the model.
   */
  bool initialize_model ( std::string input_name = "model" ) override;
  
  //! get_model - see "model_engine.h".
  CPModelUPtr get_model ( std::string model_name = "model" ) override;
  
  //! Print info about the engine
  void print () const override;
};


#endif
