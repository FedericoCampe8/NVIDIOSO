//
//  input_data_utest.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/10/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Unit test for InputData class.
//

#ifndef NVIDIOSO_input_data_utest_h
#define NVIDIOSO_input_data_utest_h

#include "unit_test.h"

class InputDataUTest : public UnitTest {
private:
	//! String describing failures
	std::string _failure_string;	
public:
	InputDataUTest();

	virtual ~InputDataUTest();

	bool run_test() override;
	
	std::string get_failure () override;

};

#endif