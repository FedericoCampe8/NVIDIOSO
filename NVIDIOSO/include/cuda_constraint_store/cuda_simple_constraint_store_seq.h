//
//  cuda_simple_constraint_store_seq.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/18/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store on device 
//  using sequential propagation.
//  @note This class is mainly for testing and comparison purposes.
//

#ifndef __NVIDIOSO__cuda_simple_constraint_store_seq__
#define __NVIDIOSO__cuda_simple_constraint_store_seq__

#include "cuda_simple_constraint_store.h"

class CudaSimpleConstraintStoreSeq : public CudaSimpleConstraintStore {
protected:
    //! Invoke the kernel which performs consistency on device
	void dev_consistency ();

public:
    CudaSimpleConstraintStoreSeq ();

    ~CudaSimpleConstraintStoreSeq ();
};

#endif
