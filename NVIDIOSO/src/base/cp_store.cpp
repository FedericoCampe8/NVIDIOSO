//
//  csp_store.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 06/27/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cp_store.h"
#include "factory_parser.h"
#include "model_generator.h"
#include "factory_generator.h"
#include "factory_cp_model.h"

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
#if CUDAON
    cudaDeviceReset();
#endif
}//~CSPStore

bool
CPStore::load_model ( string ifile )
{ 
    // Delete possible previous instances of parser (i.e., internal state)
    delete _parser;
  
    // Create a new parser
    if ( ifile.compare( "" ) != 0 ) _in_file = ifile;
    
    // Get parser for FlatZinc models
    _parser = FactoryParser::get_parser ( ParserType::FLATZINC );
    _parser->set_input( _in_file );

    // Parse model
    bool success = _parser->parse ();
  
    // Check whether the parser has failed
    if ( _parser->is_failed() || !success )
    {
    	LogMsg << _dbg + "Error while loading the model" << std::endl;
        return false;
    }

    return true;
}//load_model

void
CPStore::init_model ()
{
    string cp_mdl;
  
    // Create a new CPModel
    LogMsg << _dbg + "Instantiating a (CUDA) CP model" << std::endl;
    cp_mdl = "CP_Model";
    
#if CUDAON
    _cp_model = FactoryCPModel::get_cp_model ( CPModelType::CUDA_CP_MODEL_SIMPLE );
    cp_mdl += " (CUDA Version)";
#else
    _cp_model = FactoryCPModel::get_cp_model ( CPModelType::CP_MODEL );
#endif

    if ( _cp_model )
    {
    	LogMsg << _dbg + cp_mdl + " created" << std::endl;
    }
    else
    {
    	LogMsg << _dbg + cp_mdl + " failed" << std::endl;
        throw;
    }
  
    /*
     * Use a model generator to instantiate variables,
     * constraints, and the search engine.
     */
    ModelGenerator * generator =
        FactoryModelGenerator::get_generator( GeneratorType::CUDA );
        
    /*
     * This works as follows:
     * while the parser has some more tokens (of a specific token type),
     * the client asks the parser to get the next token.
     * The token is given in input to the generator that handles it and
     * instantiate the right object (e.g., a FD variable).
     * The object is then added to the CP model.
     */
  	// Aux info
  	LogMsg << _dbg + "add auxiliary info/arrays to the model" << std::endl;
  	while ( _parser->more_aux_arrays () )
  	{
  		try
        {
        	std::pair < std::string, std::vector< int > > aux_pair = 
        	generator->get_auxiliary_parameters ( _parser->get_aux_array() );
        	
            _cp_model->add_aux_array ( aux_pair.first, aux_pair.second );
        }
        catch ( exception& e )
        {
            // Log exception
            LogMsg << e.what() << endl;

            // Throw again to exit the program in a clean fashion
            throw;
        }
  	}
  	
    // Variables
    LogMsg << _dbg + "add variables to the model" << std::endl;
    while ( _parser->more_variables () ) 
    {
    	try
        {
            _cp_model->add_variable ( generator->get_variable ( _parser->get_variable() ) );
        }
        catch ( exception& e )
        {
            // Log exception
            LogMsg << e.what() << endl;

            // Throw again to exit the program in a clean fashion
            throw;
        }
    }
  
    // Constraints
    LogMsg << _dbg + "add constraints to the model" << std::endl;
    while ( _parser->more_constraints () )
    {
        try
        {
            _cp_model->add_constraint ( generator->get_constraint ( _parser->get_constraint() ) );
        }
        catch ( exception& e )
        {
            // Log exception
            LogMsg << e.what() << endl;
            
            // Throw again to exit the program in a clean fashion
            throw;
        }
    }
  
    // Constraint store
    LogMsg << _dbg + "add constraint store to the model" << std::endl;
    try
    {
        _cp_model->add_constraint_store( generator->get_store ( nullptr ) );
    }
    catch ( exception& e )
    {
        // Log exception
        LogMsg << e.what() << endl;
        
        // Throw again to exit the program in a clean fashion
        throw;
    }
  
  
    /*
     * Search engine.
     * @note search engine needs a constraint store.
     */
    LogMsg << _dbg + "add a search engine to the model" << std::endl;
    while ( _parser->more_search_engines () )
    {
        try
        {
            _cp_model->add_search_engine ( generator->get_search_engine ( _parser->get_search_engine() ) );
        }
        catch ( exception& e )
        {
            // Log exception
            LogMsg << e.what() << endl;
            
            // Throw again to exit the program in a clean fashion
            throw;
        }
    }
  
    // Finalize the model according to different architectures
    LogMsg << _dbg + "finalize the model" << std::endl;
    try
    {
        _cp_model->finalize ();
    }
    catch ( NvdException& e )
    {
        cout << e.what() << endl;
        throw;
    }
  
    delete generator;
}//init_model

// Print info about the model
void
CPStore::print_model_info ()
{
    cout << "======== MODEL ========\n";
    print_model_variable_info   ();
    print_model_domain_info     ();
    print_model_constraint_info ();
    cout << "=======================\n";
    _cp_model->print ();
    cout << "=======================\n";
}//print_model_info

void
CPStore::print_model_variable_info ()
{
    cout << "Variables: " << _cp_model->num_variables () << endl;
}//print_model_variable_info

void
CPStore::print_model_domain_info ()
{
}//print_model_domain_info

void
CPStore::print_model_constraint_info ()
{
    cout << "Constraints: " << _cp_model->num_constraints () << endl;
}//print_model_constraint_info

