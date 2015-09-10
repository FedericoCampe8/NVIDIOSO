//
//  tester.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Tester class to run unit tests.
//  This class runs all the unit tests registered in the unit test register.
//

#include "globals.h"
#include "input_data.h"
#include "cp_store.h"
#include "cp_solver.h"
#include "unit_test_register.h"

#ifndef __NVIDIOSO__tester__
#define __NVIDIOSO__tester__

extern UnitTestRegister& utest_register;

class Tester {
public:

	Tester ();
	
	virtual ~Tester ();
	
	virtual void run ();
};
	
#endif
