//
//  cuda_simple_constraint_store_1bKc.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/02/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store on device 
//  using 1 block per K constraints.
//

#ifndef __NVIDIOSO__cuda_simple_constraint_store_1bKc__
#define __NVIDIOSO__cuda_simple_constraint_store_1bKc__

#include "cuda_simple_constraint_store_1b1c.h"

class CudaSimpleConstraintStore1bKc : public CudaSimpleConstraintStore1b1c {
private:
	std::size_t _grid_size;
	std::size_t _block_size;
	std::size_t _shared_mem_size;
	std::size_t _shared_mem_array_size;
	std::size_t _num_constraints_per_block;
protected:

	//! Move queue to device
    void move_queue_to_device () override; 
	
    //! Invoke the kernel which performs consistency on device
    void dev_consistency () override;
    
    //! Calculate number of constraints per block
    void set_propagation_parameters ();
    
public:
    CudaSimpleConstraintStore1bKc ();

    ~CudaSimpleConstraintStore1bKc ();
};

#endif
