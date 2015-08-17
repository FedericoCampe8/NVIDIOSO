//
//  cuda_memento_state.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "cuda_memento_state.h"
#include "cuda_domain.h"

CudaMementoState::CudaMementoState ( IntDomainPtr int_domain ) {
  _memento_size =
  (static_cast<CudaDomain *>(int_domain.get()))->allocated_bytes();
  _domain_state = (int *) malloc( _memento_size );
  
  memcpy( (void *) _domain_state,
          (void *) (static_cast<CudaDomain *> (int_domain.get()))->get_concrete_domain(),
          _memento_size );
}//CudaMementoState

CudaMementoState::~CudaMementoState () {
  free ( _domain_state );
}//CudaMementoState

void
CudaMementoState::set_memento ( IntDomainPtr int_domain ) {
  if ( int_domain == nullptr ) {
    throw NvdException ( "Failed to set (concrete) Memento: no given pointer." );
  }

  (static_cast<CudaDomain *>(int_domain.get()))->set_domain_status ( _domain_state );
}//set_memento

void
CudaMementoState::print () const {
  std::cout << "Cuda Memento State:\n";
  std::cout << "Memento's size: " << _memento_size << std::endl;
  std::cout << "Internal representation: " << std::endl;
  int idx = 0;
  for (int i = 0; i < _memento_size; i += sizeof(int) )  {
    std::cout << _domain_state[ idx++ ] << " ";
  }
  std::cout << std::endl;
}//print