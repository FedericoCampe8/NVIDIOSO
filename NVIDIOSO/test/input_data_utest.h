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
#include "input_data.h"

class InputDataUTest : public UnitTest {
public:
	InputDataUTest();

	virtual ~InputDataUTest();

	bool test() override;
};

#endif