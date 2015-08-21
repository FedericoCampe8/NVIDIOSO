//
//  cuda_simple_constraint_store_1bKv.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/21/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store on device 
//  using 1 block per K variables.
//

#ifndef __NVIDIOSO__cuda_simple_constraint_store_1bKv__
#define __NVIDIOSO__cuda_simple_constraint_store_1bKv__

#include "cuda_simple_constraint_store_1b1v.h"

class CudaSimpleConstraintStore1bKv : public CudaSimpleConstraintStore1b1v {
private:
	std::size_t _shared_memory;
	
protected:
    
    //! Copy the current queue of constraints on device
    void move_queue_to_device () override;
    
    //! Invoke the kernel which performs consistency on device
    void dev_consistency ();

public:
    CudaSimpleConstraintStore1bKv ();

    ~CudaSimpleConstraintStore1bKv ();
	
};

#endif
