//
//  main.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "globals.h"
#include "input_data.h"
#include "parser.h"
#include "factory_parser.h"

int main( int argc, char * argv[] )
{
  	std::string dbg = "Main - ";
  
  	// Input
  	InputData& i_data = InputData::get_instance ( argc, argv );

	// Get parser for FlatZinc models
    Parser * parser = FactoryParser::get_parser ( ParserType::FLATZINC );
    parser->set_input( i_data.get_in_file() );

    // Parse model
    bool success = parser->parse ();
    
    // Check whether the parser has failed
    if ( parser->is_failed() || !success )
    {
    	std::cout << "Error while parsing the model" << std::endl;
    	delete parser;
        exit ( 1 );
    }
    
    while ( parser->more_tokens () )
  	{
  		parser->get_next_content()->print();
  	}
	
	delete parser;
  	return 0;
}

