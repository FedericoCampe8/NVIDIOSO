//
//  csp_store.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cp_store.h"
#include "factory_parser.h"
#include "model_generator.h"
#include "factory_generator.h"

using namespace std;

/// Init static variable
CPStore* CPStore::_cp_ds_instance = nullptr;

CPStore::CPStore ( std::string in_file ) :
DataStore ( in_file ),
_parser   ( nullptr ) {
  DataStore::_dbg = "CPStore - ";
}//CSPStore

CPStore::~CPStore () {
  delete _parser;
}//~CSPStore

bool
CPStore::load_model ( string ifile ) {
  
  // Delete previous instances of parsers (i.e., internal state)
  delete _parser;
  
  // Create a new parser
  if ( ifile.compare( "" ) != 0 ) _in_file = ifile;
  
  /*
   * Instantiate a parser for FlatZinc files.
   * @note: FlatZinc files are the only
   * supported files at this time.
   */
  _parser = FactoryParser::get_parser ( ParserType::FLATZINC );
  _parser->set_input( _in_file );
  
  /*
   * Get next content.
   * @note: it initializes the set of
   * tokens that will be used to create
   * variables, constraints, and the 
   * search engine.
   */
  while ( _parser->more_tokens() ) {
    _parser->get_next_content();
  }
  
  // Check whether the parser has failed
  if ( _parser->is_failed() ) {
    logger->error( _dbg + "Error while loading the model" );
    return false;
  }
  
  return true;
}//load_model

void
CPStore::init_model () {
  
  // New Model
  _cp_model = new CPModel ();
  
  /*
   * Use a model generator to instantiate variables,
   * constraints, and the search engine.
   */
  ModelGenerator * generator =
  FactoryModelGenerator::get_generator( GeneratorType::CUDA );
  
  /*
   * This works as follows:
   * While the parser has some more tokens (of a specific token type),
   * the client asks the parser to get the next token.
   * The token is given in input to the generator that handles it and
   * instantiate the right object (e.g., a FD variable).
   * The object is then added to the CP model.
   */
  
  // Variables
  while ( _parser->more_variables () ) {
    try {
      _cp_model->add_variable ( generator->get_variable ( _parser->get_variable() ) );
    } catch ( exception& e ) {
      // Log exception
      logger->error( e.what() );
      // Throw again to exit the program in a clean fashion
      throw;
    }
  }
  
  // Constraints
  while ( _parser->more_constraints () ) {
    _cp_model->add_constraint ( generator->get_constraint ( _parser->get_constraint() ) );
  }
  
  // Constraint store
  _cp_model->add_constraint_store( generator->get_store () );
  
  /*
   * Search engine.
   * @note search engine needs a constraint store.
   */
  while ( _parser->more_search_engines () ) {
  _cp_model->add_search_engine ( generator->get_search_engine ( _parser->get_search_engine() ) );
  }
  
  delete generator;
}//init_store


// Print info about the model
void
CPStore::print_model_info () {
  
}//print_model_info

void
CPStore::print_model_variable_info () {
  
}//print_model_variable_info

void
CPStore::print_model_domain_info () {
  
}//print_model_domain_info

void
CPStore::print_model_constraint_info () {
  
}//print_model_constraint_info

