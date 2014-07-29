//
//  cuda_variable.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_variable.h"
#include "cuda_domain.h"

using namespace std;

CudaVariable::CudaVariable () :
IntVariable () {
  _dbg = "CudaVariable - ";
  
  //Instantiate CUDA-Specific domain
  _domain_ptr = make_shared<CudaDomain> ();
}//CudaVariable

CudaVariable::CudaVariable ( int id ) :
IntVariable ( id ) {
  _dbg = "CudaVariable - ";
  
  //Instantiate CUDA-Specific domain
  _domain_ptr = make_shared<CudaDomain> ();
}//CudaVariable

CudaVariable::~CudaVariable () {
}//~CudaVariable

void
CudaVariable::set_domain () {
  _domain_ptr->init_domain ( Domain::MIN_DOMAIN(), Domain::MAX_DOMAIN() );
}//set_domain

void
CudaVariable::set_domain ( int lower_bound, int upper_bound ) {
  _domain_ptr->init_domain( lower_bound, upper_bound);
}//set_domain

void
CudaVariable::set_domain ( vector < vector < int > > set_of_subsets ) {
  
  /*
   * Check for set of subsets.
   * @todo implement set of sets.
   */
  if ( set_of_subsets.size () > 1 ) {
    throw  NvdException ( ( _dbg + "Set of subsets not yet supported" ).c_str(),
                           __FILE__, __LINE__ );
  }
  
  int min_element = *std::min_element( set_of_subsets[ 0 ].begin(),
                                       set_of_subsets[ 0 ].end() );
  int max_element = *std::max_element( set_of_subsets[ 0 ].begin(),
                                       set_of_subsets[ 0 ].end() );
  set_domain ( min_element, max_element );
  
  // Clear values not belonging to the set of values
  for ( int val = min_element; val < max_element; val++ ) {
    auto it = std::find( set_of_subsets[ 0 ].begin(),
                         set_of_subsets[ 0 ].end(),
                         val );
    if ( it == set_of_subsets[ 0 ].end() ) {
      _domain_ptr->subtract ( val );
    }
  }
}//set_domain

void
CudaVariable::print () const {
  cout << "  - CUDA_Variable -  " << endl;
  IntVariable::print();
}//print
