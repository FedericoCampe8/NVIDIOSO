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
  	std::string dbg = "Main[invidioso_utest] - ";
	  
	/**************************************
   	 *             CHECK INPUT            *
   	 **************************************/
	if ( argc > 2 )
	{
		std::cerr << "Unit Test: invalid input.\n";
		std::cerr << "Usage:\n";
		std::cerr << "./invidioso [-h | --help | -v | --verbose]\n";
		return 0;					
	}
	
	bool verbose = false;
	if ( argc == 2 )
	{
		std::string opt ( argv[ 1 ] );
		if ( opt == "-h" || opt == "--help" )
		{
			std::cout << "Unit Test usage:\n";
			std::cout << "./invidioso [-h | --help | -v | --verbose]\n";
			std::cout << "Where:\n";
			std::cout << "\t-h | --help: print this help message and exit.\n";
			std::cout << "\t-v | --verbose: print verbose information during unit test.\n";
			std::cout << "@note:\n";
			std::cout << "This is a Unit Test framework for iNVIDIOSO1.0.\n";
			std::cout << "The purpouse of this program is to perform unit testing on iNVIDIOSO1.0.\n";
			std::cout << "Unit test should be performed every time a new component is added to iNVIDIOSO1.0.\n";
			std::cout << "For any question feel free to write at fede.campe@gmail.com.\n";
			return 0;	
		}
		if ( opt != "-v" && opt != "--verbose" )
		{
			std::cerr << "Unit Test: invalid input.\n";
			std::cerr << "Usage:\n";
			std::cerr << "./invidioso [-h | --help | -v | --verbose]\n";
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
	LogMsgUT << dbg << "Run Tests" << std::endl;

	bool success = true;
	try
	{
		tester->run();
	}
	catch (std::exception& e)
	{
		success = false;
		LogMsgUT << dbg << "Testing failed: ";
		LogMsgUT << std::string ( e.what() ) << std::endl;
	}

	if (success)
	{
		LogMsgUT << dbg << "Testing succeed" << std::endl;
	}
	 
  	/***************************************
   	 *            CLEAN AND EXIT           *
   	***************************************/
   	LogMsgUT << dbg << "Exit" << std::endl;

  	return 0;
}

