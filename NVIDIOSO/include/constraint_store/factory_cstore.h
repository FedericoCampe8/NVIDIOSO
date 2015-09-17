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
  		std::string dev_cstore_type = solver_configurator.get_configuration_string ( "CSTORE_CUDA_PROP" );
      if ( dev_cstore_type == "sequential" )
      {
        return std::make_shared<CudaSimpleConstraintStoreSeq> ();
      }
      else if ( dev_cstore_type == "block_per_constraint" )
      {
        return std::make_shared<CudaSimpleConstraintStore1b1c> ();
      }
      else if ( dev_cstore_type == "block_per_variable" )
      {
        return std::make_shared<CudaSimpleConstraintStore1b1v> ();
      }
      else if ( dev_cstore_type == "block_per_k_constraint"  )
      {
        return std::make_shared<CudaSimpleConstraintStore1bKc> ();
      }
      else if ( dev_cstore_type == "block_per_k_variable" )
      {
        return std::make_shared<CudaSimpleConstraintStore1bKv> ();
      }
      else
      {
        return std::make_shared<CudaSimpleConstraintStoreSeq> (); 
      }
  	}//get_cstore
};

#endif

    

