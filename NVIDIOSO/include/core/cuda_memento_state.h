//
//  cuda_memento_state.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents a Memento considering CUDA implementation,
//  i.e., cuda domains for variables' states.
//

#ifndef __NVIDIOSO__cuda_memento_state__
#define __NVIDIOSO__cuda_memento_state__

#include "memento_state.h"
#include "int_domain.h"

class CudaMementoState : public MementoState {
private:
  //! Size of this Cuda Memento in Bytes
  size_t _memento_size;
  
  //! State of a Cuda Memento
  int * _domain_state;

public:
  /**
   * Constructor for Cuda Memento.
   * @param int_domain a reference to a int domain
   *        from which get the internal domain's representation.
   */
  CudaMementoState ( IntDomainPtr int_domain );
  
  ~CudaMementoState ();
  
  /**
   * Sets domain's state as new state into
   * the given (int) domain
   * @param int_domain a reference to the domain to update.
   */
  void set_memento ( IntDomainPtr int_domain );
  
  void print () const override;
};

#endif /* defined(__NVIDIOSO__cuda_memento_state__) */
