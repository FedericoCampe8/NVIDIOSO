//
//  main_test.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "tester.h"

int main( int argc, char * argv[] ) 
{	
  	std::string dbg = "Main (Test) - ";
	  
	/**************************************
   	 *             CHECK INPUT            *
   	 **************************************/
	if ( argc > 2 )
	{
		std::cerr << "Unit Test: invalid input\n";
		std::cerr << "Usage:\n";
		std::cerr << "./invidioso [-v | --verbose]\n";
		return 0;					
	}
	
	bool verbose = false;
	if ( argc == 2 )
	{
		std::string opt ( argv[ 1 ] );
		if ( opt != "-v" && opt != "--verbose" )
		{
			std::cerr << "Unit Test: invalid input\n";
			std::cerr << "Usage:\n";
			std::cerr << "./invidioso [-v | --verbose]\n";
			return 0;	
		}
		else
		{
			LogMsg.set_verbose ( true );
		}
	}
	
  	/**************************************
   	 *             INIT TESTER            *
   	 **************************************/
	std::unique_ptr<Tester> tester ( new Tester() );
  	
  	/***************************************
   	 *           Run  Unit Tests           *
   	 ***************************************/
	LogMsg << dbg << "Run Tests" << std::endl;

	bool success = true;
	try
	{
		tester->run();
	}
	catch (std::exception& e)
	{
		success = false;
		LogMsg << dbg << "Testing failed: ";
		LogMsg << std::string ( e.what() ) << std::endl;
	}

	if (success)
	{
		LogMsg << dbg << "Testing succeed" << std::endl;
	}
	 
  	/***************************************
   	 *            CLEAN AND EXIT           *
   	***************************************/
   	LogMsg << dbg << "Exit" << std::endl;

  	return 0;
}

