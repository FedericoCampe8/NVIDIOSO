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
		tester->run ();
	}
	catch ( std::exception& e )
	{
		success = false;
		LogMsg << dbg << "Testing failed" << std::endl;
	}
  
  	if ( success )
	 {
		 std::cout << dbg << "Testing succeed" << std::endl;
	 }
	 
  	/***************************************
   	 *            CLEAN AND EXIT           *
   	***************************************/
   	LogMsg << dbg << "Exit" << std::endl;

  	return 0;
}

