//
//  cuda_concrete_domain.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 15/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_concrete_domain.h"

using namespace std;

CudaConcreteDomain::CudaConcreteDomain ( size_t size ) :
_dbg ( "CudaConcreteDomain -" ) {
  
  // Force alloc of a multiple of sizeof( int )
  if ( size % sizeof( int ) != 0 ) {
    size += sizeof( int ) - ( size % sizeof( int ) );
  }
  
  // Allocate the concrete domain representation
  _concrete_domain = (int*) malloc( size );
  
  if ( !_concrete_domain ) {
    throw  NvdException ( (_dbg + "Domain not singleton").c_str(),
                          __FILE__, __LINE__ );
  }
  
  _num_chunks = (( int ) size) / sizeof( int );
}//CudaConcreteDomain

CudaConcreteDomain::~CudaConcreteDomain () {
  free( _concrete_domain );
}//~CudaConcreteDomain


int
CudaConcreteDomain::lower_bound () const {
  return _lower_bound;
}//lower_bound

//! Returns upper bound
int
CudaConcreteDomain::upper_bound () const {
  return _upper_bound;
}//upper_bound

int
CudaConcreteDomain::get_num_chunks () const {
  return _num_chunks;
}//get_num_chunks

size_t
CudaConcreteDomain::get_alloc_bytes () const {
  return ( get_num_chunks () * sizeof( int ) );
}//get_alloc_bytes

void
CudaConcreteDomain::flush_domain () {
  memset( _concrete_domain, 0, get_alloc_bytes() );
  set_empty ();
}//empty_domain

void
CudaConcreteDomain::set_empty () {
  _lower_bound = 1;
  _upper_bound = 0;
}//empty_domain

bool
CudaConcreteDomain::is_empty () const {
  return _upper_bound < _lower_bound;
}//is_empty

