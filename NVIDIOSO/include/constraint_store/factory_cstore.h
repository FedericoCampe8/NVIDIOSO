//
//  factory_cstore.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/18/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Factory method for constraint store classes.
//

#ifndef NVIDIOSO_factory_cstore_h
#define NVIDIOSO_factory_cstore_h

#include "globals.h"
#include "soft_constraint_store.h"
#include "cuda_simple_constraint_store.h"
#include "cuda_simple_constraint_store_seq.h"
#include "cuda_simple_constraint_store_1b1c.h"
#include "cuda_simple_constraint_store_1bKc.h"
#include "cuda_simple_constraint_store_1b1v.h"
#include "cuda_simple_constraint_store_1bKv.h"

class FactoryCStore { 
  
public:
  /**
   * Get the right instance of constraint store based on input options.
   * @param on_device, if True it generates a constraint store for device propagation,
   *        otherwise it generates the constraint store for host propagation.
   * @param type, type of constraint store to generate.
   */
  static ConstraintStorePtr get_cstore ( bool on_device=false, int type=0, bool local_search=false ) 
  {
  		if ( !on_device )
  		{
  			if ( local_search )
  			{
  				return std::make_shared<SoftConstraintStore> ();
  			}
  			else
  			{
  				return std::make_shared<SimpleConstraintStore> ();
  			}
  		}
  		CudaPropParam dev_cstore_type = solver_params->cstore_int_to_type ( type );
    	switch ( dev_cstore_type ) 
    	{
      		case CudaPropParam::SEQUENTIAL:
        		return std::make_shared<CudaSimpleConstraintStoreSeq> ();
        	case CudaPropParam::BLOCK_PER_CON:
        		return std::make_shared<CudaSimpleConstraintStore1b1c> ();
        	case CudaPropParam::BLOCK_PER_VAR:
        		return std::make_shared<CudaSimpleConstraintStore1b1v> ();
        	case CudaPropParam::BLOCK_K_CON:
        		return std::make_shared<CudaSimpleConstraintStore1bKc> ();
        	case CudaPropParam::BLOCK_K_VAR:
        		return std::make_shared<CudaSimpleConstraintStore1bKv> ();
      		default:
        		return std::make_shared<CudaSimpleConstraintStoreSeq> ();
    	}
  	}//get_cstore
};

#endif

    

