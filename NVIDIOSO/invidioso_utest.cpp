//
//  main_test.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "tester.h"

// Function to analyse results given by (valgrind) unit tests (this main)
void run_analysis ( const char * const in_file_str );

int main ( int argc, char * argv[] ) 
{	
  	std::string dbg = "Main[invidioso_utest] - ";
	  
	/**************************************
   *             READ INPUT             *
   **************************************/
	if ( argc > 4 )
	{
		std::cerr << "Unit Test: invalid input.\n";
		std::cerr << "Usage:\n";
		std::cerr << "./invidioso [-h | --help] [-v | --verbose] [-a <file_name>] [-t <test>] \n";
		return 0;					
	}
	
	std::string on_test {};
	bool verbose = false;
	if ( argc == 2 )
	{
		std::string opt ( argv[ 1 ] );
		if ( opt == "-h" || opt == "--help" )
		{
			std::cout << "Unit Test usage:\n";
			std::cout << "./invidioso [-h | --help] [-v | --verbose] [-a <file_name>] [-t <test>] \n";
			std::cout << "Where:\n";
			std::cout << "\t-h | --help: print this help message and exit.\n";
			std::cout << "\t-v | --verbose: print verbose information during unit test.\n";
			std::cout << "\t-t <test>: unit test to run (must be registered in the register of tests).\n";
			std::cout << "\t-a <file_name>: analyze file_name to get a summary of errors given by this tool.\n";
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
			std::cerr << "./invidioso [-h | --help] [-v | --verbose] [-a <file_name>] [-t <test>]\n";
			return 0;	
		}
		else
		{
			LogMsg.set_verbose ( true );
		}
	}
	else
	{
		std::string opt;
		for ( int i = 1; i < argc; i++ )
		{
			opt = std::string ( argv[ i ] );
			if ( opt == "-a" )
			{
				if ( i == argc - 1 )
				{
					std::cerr << "Unit Test: invalid input.\n";
					std::cerr << "Usage:\n";
					std::cerr << "./invidioso [-h | --help] [-v | --verbose] [-a <file_name>] [-t <test>]\n";
				}
				else
				{
					// Run analysis of error on (log) given file
					run_analysis ( argv[ i + 1 ] );
					return 0;
				}	
			}
			else if ( opt == "-t" )
			{
				if ( i == argc - 1 )
				{
					std::cerr << "Unit Test: invalid input.\n";
					std::cerr << "Usage:\n";
					std::cerr << "./invidioso [-h | --help] [-v | --verbose] [-a <file_name>] [-t <test>]\n";
				}
				else
				{
					on_test = std::string ( argv[ i + 1 ] ); 
					++i;
				}	
			}
			else if ( opt ==  "-v" || opt == "--verbose")
			{
				LogMsg.set_verbose ( true );
			}
			else
			{
				std::cerr << "Unit Test: invalid input.\n";
				std::cerr << "Usage:\n";
				std::cerr << "./invidioso [-h | --help] [-v | --verbose] [-a <file_name>] [-t <test>]\n";
			}
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
	
	// If specified, perform only one single unit test
	if ( on_test != "" )
	{
		tester->set_running_test ( on_test );
	}
	
	bool success = true;
	try
	{
		tester->run();
	}
	catch (std::exception& e)
	{
		success = false;
		LogMsgUT << dbg << "Unit test failed: ";
		LogMsgUT << std::string ( e.what() ) << std::endl;
	}

	if (success)
	{
		LogMsgUT << dbg << "Unit test passed" << std::endl;
	}
	 
 	/***************************************
   *            CLEAN AND EXIT           *
   ***************************************/
   	LogMsgUT << dbg << "Exit" << std::endl;

  	return 0;
}

void run_analysis ( const char * const in_file_str )
{
	//Sanity check
    std::ifstream infile ( in_file_str, std::ifstream::in );
    if ( !infile.is_open() ) 
    {
    	std::cerr << "Can't open file for analysis of errors: " << in_file_str << std::endl;
		return;
    }
	
	bool utest_pass = true;
	std::size_t found {};
	std::string line {};
	std::string memory_error_str {};
	while ( getline( infile, line ) ) 
    {
    	found = line.find ( "Unit test failed: " );
		if ( found != std::string::npos )
		{
			 utest_pass = false;
			 continue;
		}
		found = line.find ( "ERROR SUMMARY: " );
		if ( found != std::string::npos )
		{
			std::size_t E_char_pos = line.find_first_of ("E");
			memory_error_str = line.substr ( E_char_pos );
			break;
		}
    }
	
	std::cout << "ANALYSIS RESULT:" << std::endl;
	if ( utest_pass )
	{
		std::cout << "- UNIT TEST COMPLETED SUCCESSFULLY" << std::endl;
	}
	else
	{
		std::cout << "\a\n";
		std::cout << "- UNIT TEST NOT PASSED (see log file " << in_file_str << ")" << std::endl;
	}
	std::cout << "- " << memory_error_str << std::endl;	 
			 
    infile.close ();
}//run_analysis

