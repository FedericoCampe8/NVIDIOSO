//
//  cp_model_engine.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 06/27/14.
//  Modified by Federico Campeotto on 09/14/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cp_model_engine.h"
#include "factory_parser.h"
#include "model_generator.h"
#include "factory_generator.h"
#include "factory_cp_model.h"

using std::cout;
using std::endl;

CPModelEngine::CPModelEngine () :
  _dbg ( "CPModelEngine - " ) {
}//CPModelEngine

CPModelEngine::~CPModelEngine () {
#if CUDAON
    cudaDeviceReset();
#endif
}//~CPModelEngine

bool
CPModelEngine::load_model ( std::string input_path, std::string input_name )
{ 
  //Sanity check
  auto it = _parser_map.find ( input_name );
  if ( it != _parser_map.end () )
  {
    LogMsgW << _dbg << "Model " << input_name << " already loaded" << endl;
    return true;
  }
  
  //Sanity check
  if ( input_path.substr ( input_path.size() - 3 ) != "fzn" )
  {
    LogMsgW << _dbg << "model extension not supported" << endl;
    return false;
  }
  
  // Generate a new parser to parse the given input
  ParserUPtr parser ( FactoryParser::get_parser ( ParserType::FLATZINC ) );
  
  // Se input and parse the model
  parser->set_input ( input_path );
  bool success = parser->parse ();
  
  // Check whether the parser has failed
  if ( !success || parser->is_failed() )
  {
    LogMsgE << _dbg + "Error while loading the model" << std::endl;
    return false;
  }

  // Store the (loaded) parser
  _parser_map [ input_name ] = std::move ( parser );
  
  return true;
}//load_model

bool 
CPModelEngine::initialize_model ( std::string input_name )
{
    auto it = _parser_map.find ( input_name );
    if ( it == _parser_map.end () )
    {
      LogMsgW << _dbg << "Model " << input_name << " not loaded" << endl;
      return false;
    }
    
    auto it_mdl = _model_map.find ( input_name );
    if ( it_mdl != _model_map.end () )
    {
      LogMsgW << _dbg << "Model " << input_name << " already loaded" << endl;
      return true;
    }
    
    // Instantiate a new CPModel
    std::string cp_mdl { "CP_Model" };
    LogMsg << _dbg + "Instantiating a CP model" << std::endl;
    
    CPModel * cp_model {};
    
#if CUDAON
    cp_model = FactoryCPModel::get_cp_model ( CPModelType::CUDA_CP_MODEL_SIMPLE );
    cp_mdl += " -- CUDA Version --";
#else
    cp_model = FactoryCPModel::get_cp_model ( CPModelType::CP_MODEL );
#endif

    if ( cp_model != nullptr )
    {
      LogMsg << _dbg + cp_mdl + " created" << std::endl;
    }
    else
    {
      LogMsgE << _dbg + cp_mdl + " failed" << std::endl;
      return false;
    }
  
    /*
     * Use a model generator to instantiate variables,
     * constraints, and the search engine.
     */
    ModelGenerator * generator {};
    try
    {
      generator = FactoryModelGenerator::get_generator( GeneratorType::CUDA );
    }
    catch (...)
    {
      LogMsgE << _dbg + "Cannot create generator" << std::endl;
      return false;
    }
    
    // Sanity check
    if ( generator == nullptr )
    {
      LogMsgE << _dbg + "Cannot create generator" << std::endl;
      return false;
    }
        
    /*
     * This works as follows:
     * while the parser has some more tokens (of a specific token type),
     * the client asks the parser to get the next token.
     * The token is given in input to the generator that handles it and
     * instantiate the right object (e.g., a FD variable).
     * The object is then added to the CP model.
     */
     
  	LogMsg << _dbg + "Add parameters to the model" << std::endl;
  	while ( _parser_map[ input_name ]->more_aux_arrays () )
  	{
  		try
        {
        	std::pair < std::string, std::vector< int > > aux_pair = 
        	generator->get_auxiliary_parameters ( _parser_map[ input_name ]->get_aux_array() );
        	
            cp_model->add_aux_array ( aux_pair.first, aux_pair.second );
        }
        catch ( std::exception& e )
        {
            // Log exception
            LogMsgE << e.what() << endl;

            // Free and return
            delete cp_model;
            return false;
        }
  	}
  	
    // Variables
    LogMsg << _dbg + "Add variables to the model" << std::endl;
    while ( _parser_map[ input_name ]->more_variables () ) 
    {
    	try
        {
            cp_model->add_variable ( generator->get_variable ( _parser_map[ input_name ]->get_variable() ) );
        }
        catch ( std::exception& e )
        {
            // Log exception
            LogMsgE << e.what() << endl;

            // Free and return
            delete cp_model;
            return false;
        }
    }
  
    // Constraints
    LogMsg << _dbg + "Add constraints to the model" << std::endl;
    while ( _parser_map[ input_name ]->more_constraints () )
    {
        try
        {
            cp_model->add_constraint ( generator->get_constraint ( _parser_map[ input_name ]->get_constraint() ) );
        }
        catch ( std::exception& e )
        {
            // Log exception
            LogMsgE << e.what() << endl;
            
            // Free and return
            delete cp_model;
            return false;
        }
    }
  
    // Constraint store
    LogMsg << _dbg + "Add constraint store to the model" << std::endl;
    try
    {
        cp_model->add_constraint_store( generator->get_store ( _parser_map[ input_name ]->get_constraint_store () ) );
    }
    catch ( std::exception& e )
    {
        // Log exception
        LogMsgE << e.what() << endl;
        
        // Free and return
        delete cp_model;
        return false;
    }
  
    /*
     * Search engine.
     * @note search engine needs a constraint store.
     */
    LogMsg << _dbg + "Add a search engine to the model" << std::endl;
    while ( _parser_map[ input_name ]->more_search_engines () )
    {
        try
        {
            cp_model->add_search_engine ( generator->get_search_engine ( _parser_map[ input_name ]->get_search_engine() ) );
        }
        catch ( std::exception& e )
        {
            // Log exception
            LogMsgE << e.what() << endl;
            
           // Free and return
           delete cp_model;
           return false;
        }
    }
  
    // Finalize the model according to different architectures
    LogMsg << _dbg + "finalize the model" << std::endl;
    try
    {
        cp_model->finalize ();
    }
    catch ( NvdException& e )
    {
        // Log exception
        LogMsgE << e.what() << endl;
        
       // Free and return
       delete cp_model;
       return false;
    }
  
    // Store the new CP Model into the map 
    _model_map [ input_name ] = CPModelUPtr ( cp_model );
    
    // Free generator
    try
    {
      delete generator;
    }
    catch (...)
    {
      LogMsgE << _dbg + "Cannot delete generator" << std::endl;
    }
    
    return true;
}//initialize_model

CPModelUPtr
CPModelEngine::get_model ( std::string model_name )
{  
  // Sanity check
  auto it = _model_map.find ( model_name );
  if ( it == _model_map.end () )
  {
    LogMsgW << _dbg <<  "Model " << model_name << " not found" << std::endl;
    return nullptr;
  }
  
  return std::move ( _model_map [ model_name ] );
}//get_model

void
CPModelEngine::print () const
{
    cout << "======== MODEL ENGINE ========\n";
    cout << "Number of parsed models:\t" << _parser_map.size () << endl;
    cout << "Number of loaded models:\t" << _model_map.size () << endl;
    cout << "==============================\n";
}//print_model_info
