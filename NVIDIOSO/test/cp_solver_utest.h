//
//  cp_solver_utest.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/10/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Unit test for CPSolver class.
//

#ifndef NVIDIOSO_cp_solver_utest_h
#define NVIDIOSO_cp_solver_utest_h

#include "unit_test.h"
#include "cp_solver.h"

class CPSolverUTest : public UnitTest {
public:
	CPSolverUTest();

	virtual ~CPSolverUTest();

	bool test() override;
};

#endif