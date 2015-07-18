//
//  cuda_simple_constraint_store_1b1c.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/18/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store on device 
//  using 1 block per constraint.
//  @note This class is mainly for testing and comparison purposes.
//

#ifndef __NVIDIOSO__cuda_simple_constraint_store_1b1c__
#define __NVIDIOSO__cuda_simple_constraint_store_1b1c__

#include "cuda_simple_constraint_store.h"

class CudaSimpleConstraintStore1b1c : public CudaSimpleConstraintStore {
protected:
    //! Invoke the kernel which performs consistency on device
	void dev_consistency ();

public:
    CudaSimpleConstraintStore1b1c ();

    ~CudaSimpleConstraintStore1b1c ();
};

#endif
