//
//  cuda_domain.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_domain.h"
#include "cuda_utilities.h"

using namespace std;

CudaDomain::CudaDomain () :
_num_allocated_bytes ( 0 ),
_domain ( nullptr ) {
}//CudaDomain

CudaDomain::~CudaDomain () {
  delete [] _domain;
}//~CudaDomain

size_t
CudaDomain::get_allocated_bytes () const {
  return _num_allocated_bytes;
}//get_allocated_bytes

void
CudaDomain::init_domain ( int min, int max ) {
  
  // Return if domain has been already initialized
  if ( _domain != nullptr ) return;
  
  // Consistency check
  assert ( min <= max );
  
  // Get proper size of domain
  int size;
  if ( ((min <= Domain::MIN_DOMAIN() + 1) && max > 0) ||
       ((max == Domain::MAX_DOMAIN()    ) && min < 0) ||
       ((max == Domain::MAX_DOMAIN() / 2) && (max == Domain::MIN_DOMAIN() / 2))
      ) {
    size = Domain::MAX_DOMAIN();
  }
  else {
    size = abs( max - min + 1 );
  }
  
  // Allocate domain according to its size
  if ( min >= 0 &&  min <= VECTOR_MAX && size <= VECTOR_MAX ) {
    _num_int_chunks = num_chunks ( VECTOR_MAX );
    _num_allocated_bytes = MAX_STATUS_SIZE + (VECTOR_MAX / BITS_IN_BYTE);
    _domain = new int [ _num_allocated_bytes / sizeof( int ) ];
    
    // Set everything as available
    for ( int i = BIT_IDX(); i < BIT_IDX() + _num_int_chunks; i++ )  {
      _domain[ i ] = 0xffffffff;
    }
    
    // Set bit-chunks to 0 if smaller than min or bigger than max
    int lower_idx = IDX_CHUNK ( min ) + BIT_IDX();
    int upper_idx = IDX_CHUNK ( max ) + BIT_IDX();
    for ( int i = BIT_IDX(); i < lower_idx; i++ )
      _domain[ i ] = 0;
    for ( int i = upper_idx + 1; i < BIT_IDX() + _num_int_chunks; i++ )
      _domain[ i ] = 0;
    
    // Set representation
    set_bit_representation ();
  }
  else {
    _num_allocated_bytes = MAX_BYTES_SIZE;
    _num_int_chunks      = MAX_DOMAIN_VALUES;
    _domain = new int [ _num_allocated_bytes / sizeof( int ) ];
    
    // Set representation
    set_list_representation ();
  }
  
  // Set bounds
  _lower_bound = min;
  _upper_bound = max;
  _domain[ LB_IDX() ] = _lower_bound;
  _domain[ UB_IDX() ] = _upper_bound;
  
  if ( _lower_bound == _upper_bound ) {
    event_to_int ( EventType::SINGLETON_EVT );
  }
  else {
    event_to_int ( EventType::NO_EVT );
  }
  
  /*
   * Set size w.r.t. the new bounds.
   * Change internal domain representation according to 
   * the current domain's elements.
   */
  update_domain ();
}//init_domain

DomainPtr
CudaDomain::clone_impl () const {
  return ( shared_ptr<CudaDomain> ( new CudaDomain ( *this ) ) );
}//clone_impl

EventType
CudaDomain::int_to_event () const {
  switch ( _domain[ EVT_IDX() ] ) {
    case INT_NO_EVT:
      return EventType::NO_EVT;
    case INT_SINGLETON_EVT:
      return EventType::SINGLETON_EVT;
    case INT_BOUNDS_EVT:
      return EventType::BOUNDS_EVT;
    case INT_CHANGE_EVT:
      return EventType::CHANGE_EVT;
    case INT_FAIL_EVT:
      return EventType::FAIL_EVT;
    default:
      return EventType::OTHER_EVT;
  }
}//int_to_event

void
CudaDomain::event_to_int ( EventType evt ) const {
  switch ( evt ) {
    case EventType::NO_EVT:
      _domain[ EVT_IDX() ] = INT_NO_EVT;
      break;
    case EventType::SINGLETON_EVT:
      _domain[ EVT_IDX() ] = INT_SINGLETON_EVT;
      break;
    case EventType::BOUNDS_EVT:
      _domain[ EVT_IDX() ] = INT_BOUNDS_EVT;
      break;
    case EventType::CHANGE_EVT:
      _domain[ EVT_IDX() ] = INT_CHANGE_EVT;
      break;
    case EventType::FAIL_EVT:
      _domain[ EVT_IDX() ] = INT_FAIL_EVT;
      break;
    default:
      _domain[ EVT_IDX() ] = INT_OTHER_EVT;
      break;
  }
}//int_to_event

void
CudaDomain::set_bit_representation () {
  _domain[ REP_IDX() ] = INT_BITMAP;
}//set_bit_representation

void
CudaDomain::set_bitlist_representation ( int num_list ) {
  assert( num_list < 0 );
  _domain[ REP_IDX() ] = num_list;
}//set_bit_representation

void
CudaDomain::set_list_representation ( int num_list ) {
  assert( num_list > 0 );
  _domain[ REP_IDX() ] = num_list;
}//set_bit_representation

CudaDomainRepresenation
CudaDomain::get_representation () const {
  if ( _domain[ REP_IDX() ] < 0 ) {
    return CudaDomainRepresenation::BITMAP_LIST;
  }
  else if ( _domain[ REP_IDX() ] == 0 ) {
    return CudaDomainRepresenation::BITMAP;
  }
  else {
    return CudaDomainRepresenation::LIST;
  }
}//get_representation

DomainPtr
CudaDomain::clone () const {
  return clone_impl ();
}//clone 

EventType
CudaDomain::get_event () const {
  return int_to_event ();
}//get_event

size_t
CudaDomain::get_size () const {
  return ( (int) _domain[ DSZ_IDX() ] );
}//get_size

void
CudaDomain::set_bounds ( int min, int max ) {
  
  // Domain failure: non valid min/max
  if ( max < min ) {
    event_to_int ( EventType::FAIL_EVT );
    return;
  }
  
  // Domain failure: out or range
  if ( _upper_bound < min || _lower_bound > max ) {
    event_to_int ( EventType::FAIL_EVT );
    return;
  }
  
  // Trying to enlarge domain has no effect
  if ( (min <= _lower_bound) && (max >= _upper_bound) ) {
    return;
  }
  
  if ( _lower_bound < min ) {
    _lower_bound = min;
    _domain[ LB_IDX() ] = min;
  }
  if ( _upper_bound > max ) {
    _upper_bound = max;
    _domain[ UB_IDX() ] = max;
  }
  
  // Event handling
  if ( _lower_bound == _upper_bound ) {
    event_to_int ( EventType::SINGLETON_EVT );
  }
  else {
    event_to_int ( EventType::BOUNDS_EVT );
  }
  
  /*
   * Set size w.r.t. the new bounds.
   * Change internal domain representation according to
   * the current domain's elements.
   */
  update_domain ();
}//set_bounds

void
CudaDomain::update_domain () {
  if ( get_representation () == CudaDomainRepresenation::BITMAP ) {
    int num_bits = update_bitmap ( _domain[ LB_IDX() ], _domain[ UB_IDX() ] );
    
    // Set new domain size
    _domain [ DSZ_IDX() ] = num_bits;
    
    // Set events if singleton or failed
    if ( num_bits == 1 ) {
      _domain [ EVT_IDX() ] = INT_SINGLETON_EVT;
    }
    else if ( num_bits == 0 ) {
      _domain [ EVT_IDX() ] = INT_FAIL_EVT;
    }
    
  }
  else  if ( get_representation () == CudaDomainRepresenation::BITMAP_LIST ) {
    update_bitmap_list ();
  }
  else {
    update_list ();
  }
}//update_size

int
CudaDomain::update_bitmap ( int min, int max, int offset_bitmap ) {
  int lower_idx = IDX_CHUNK ( min ) + offset_bitmap;
  int upper_idx = IDX_CHUNK ( max ) + offset_bitmap;
  
  /*
   * Set to zero all the bits < lower_bound and > upper_bound 
   * in the chunks identified by lower_bound and upper_bound respectively.
   */

  _domain [ lower_idx ] = CudaBitUtils::clear_bits_i_through_0( _domain [ lower_idx ], IDX_BIT ( min ) - 1 );
  
  _domain [ upper_idx ] = CudaBitUtils::clear_bits_MSB_through_i( _domain [ upper_idx ], IDX_BIT ( max ) + 1 );
  
  int num_bits = 0;
  for ( ; lower_idx <= upper_idx; lower_idx++ ) {
    num_bits += CudaBitUtils::num_1bit( (uint) _domain[ lower_idx ] );
  }
  
  return num_bits;
}//update_bitmap

void
CudaDomain::update_bitmap_list () {
  
}//update_bitmap_list

void
CudaDomain::update_list () {
  
  // Check the length of the list of pairs
  int num_pairs = _domain[ REP_IDX() ];
  
  // There is a single pair of contiguous elements
  if ( num_pairs == 1 ) {
    // Calculate new domain size and set it
    int d_size = _domain[ UB_IDX() ] - _domain[ LB_IDX() ] + 1;

    // Return if singleton or failed
    if ( d_size == 1 ) {
      _domain [ EVT_IDX() ] = INT_SINGLETON_EVT;
      return;
    } else if ( d_size <= 0 ) {
      _domain [ EVT_IDX() ] = INT_FAIL_EVT;
      return;
    }
    
    // Set new domain's size
    _domain [ DSZ_IDX() ] = d_size;
  }
  else {
    // Base case: singleton
    if ( _domain[ LB_IDX() ] == _domain[ UB_IDX() ] ) {
      _domain[ DSZ_IDX() ] = 1;
      // Bit list representation
      set_bitlist_representation ( -1 );
      _domain[ BIT_IDX()     ] = _domain[ LB_IDX() ];
      _domain[ BIT_IDX() + 1 ] = _domain[ UB_IDX() ];
      _domain[ BIT_IDX() + 2 ] = ( unsigned int ) 1;
      return;
    }
    
    /*
     * There are num_pairs pairs of contiguous elements.
     * Four cases to consider for both the new lower/upper bounds:
     * 1) new bound is equal to a lower bound
     * 2) new bound is equal to an upper bound
     * 3) new bound is within a pair {lower bound, upper bound}
     * 4) new bound is betweent to pair of bounds
     * Therefore, 16 combinations should be considered.
     * Let us consider that some combinantions are symmetric.
     */
    // First check which case is for new bounds
    bool find_lb_case = false;
    bool find_ub_case = false;
    int  lb_case =  4, ub_case =  4; // cases
    int  lb_idx  = -1, ub_idx  = -1; // index pair with a match for a case
    int  num_lw  =  0;    // sum elements before match on the left
    int  num_up  =  0;    // sum elements before match on the right
    int  curr_pair_size;  // num elements in the current pair
    for ( int i = 0; i < num_pairs; i++ ) {
      curr_pair_size = _domain[ BIT_IDX() + 2 * i + 1 ] -
                       _domain[ BIT_IDX() + 2 * i     ] + 1;
      // Lower bound
      if ( (!find_lb_case) &&
           (_domain[ BIT_IDX() + 2 * i ]    == _domain[ LB_IDX() ]) ) {
        lb_case = 1;
        lb_idx  = BIT_IDX() + 2 * i;
        find_lb_case = true;
      }
      if ( (!find_lb_case) &&
          (_domain[ BIT_IDX() + 2 * i + 1 ] == _domain[ LB_IDX() ]) ) {
        lb_case = 2;
        lb_idx  = BIT_IDX() + 2 * i;
        find_lb_case = true;
      }
      if ( (!find_lb_case) &&
           (_domain[ LB_IDX() ] > _domain[ BIT_IDX() + 2 * i     ] ) &&
           (_domain[ LB_IDX() ] < _domain[ BIT_IDX() + 2 * i + 1 ] ) ) {
        lb_case = 3;
        lb_idx  = BIT_IDX() + 2 * i;
        find_lb_case = true;
      }
      if ( (!find_lb_case) && (i < (num_pairs - 1)) &&
          (_domain[ LB_IDX() ] > _domain[ BIT_IDX() + 2 * i + 1 ] ) &&
          (_domain[ LB_IDX() ] < _domain[ BIT_IDX() + 2 * (i + 1) ] ) ) {
        lb_case = 4;
        lb_idx  = BIT_IDX() + 2 * (i + 1); //Next one is the valid one
        find_lb_case = true;
      }
      
      if ( (i > 0) && (!find_lb_case) ) {
        num_lw += curr_pair_size;
      }
      
      if ( find_ub_case ) {
        //Start counting the elements on the right
        num_up += curr_pair_size;
      }
      
      // Upper bound
      if ( (!find_ub_case) &&
          (_domain[ BIT_IDX() + 2 * i ]    == _domain[ UB_IDX() ]) ) {
        ub_case = 1;
        ub_idx  = BIT_IDX() + 2 * i;
        find_ub_case = true;
      }
      if ( (!find_ub_case) &&
          (_domain[ BIT_IDX() + 2 * i + 1 ] == _domain[ UB_IDX() ]) ) {
        ub_case = 2;
        ub_idx  = BIT_IDX() + 2 * i;
        find_ub_case = true;
      }
      if ( (!find_ub_case) &&
          (_domain[ UB_IDX() ] > _domain[ BIT_IDX() + 2 * i     ] ) &&
          (_domain[ UB_IDX() ] < _domain[ BIT_IDX() + 2 * i + 1 ] ) ) {
        ub_case = 3;
        ub_idx  = BIT_IDX() + 2 * i;
        find_ub_case = true;
      }
      if ( (!find_ub_case) && (i < (num_pairs - 1)) &&
          (_domain[ UB_IDX() ] > _domain[ BIT_IDX() + 2 * i + 1 ] ) &&
          (_domain[ UB_IDX() ] < _domain[ BIT_IDX() + 2 * (i + 1) ] ) ) {
        ub_case = 4;
        ub_idx  = BIT_IDX() + 2 * i;
        find_ub_case = true;
      }
    }//i
    
    // Find the new number of pairs
    int num_valid_pairs =
    ((ub_idx - BIT_IDX()) / 2) -
    ((lb_idx - BIT_IDX()) / 2) + 1;
    
    if ( num_valid_pairs == 0 ) {
      _domain [ DSZ_IDX() ] = 0;
      _domain [ EVT_IDX() ] = INT_FAIL_EVT;
      return;
    }
    _domain [ REP_IDX() ] = num_valid_pairs;
    
    // Decrease the total number of elements
    _domain [ DSZ_IDX() ] -= (num_lw + num_up);
    
    //Fix pairs w.r.t. each case.
    if ( lb_case == 2 ) {
      /*
       * Lower bound is 23:
       * ..., {20, 23}, {42, 47}, ... -> ..., {23, 23}, {42, 47}, ...
       */
      _domain [ DSZ_IDX() ] -= ( _domain[ lb_idx + 1 ] - _domain[ lb_idx ]);
      _domain[ lb_idx ] = _domain[ lb_idx + 1 ];
    }
    if ( lb_case == 3 ) {
      /*
       * Lower bound is 22:
       * ..., {20, 23}, {42, 47}, ... -> ..., {22, 23}, {42, 47}, ...
       */
      _domain [ DSZ_IDX() ] -= ( _domain[ LB_IDX() ] - _domain[ lb_idx ] );
      _domain[ lb_idx ] = _domain[ LB_IDX() ];
    }
    
    if ( ub_case == 1 ) {
      /*
       * Upper bound is 42:
       * ..., {20, 23}, {42, 47}, ... -> ..., {20, 23}, {42, 42}, ...
       */
      _domain [ DSZ_IDX() ] -= (_domain[ ub_idx + 1 ] - _domain[ ub_idx ]);
      _domain[ ub_idx + 1 ] = _domain[ ub_idx ];
    }
    if ( ub_case == 3 ) {
      /*
       * Upper bound is 45:
       * ..., {20, 23}, {42, 47}, ... -> ..., {20, 23}, {42, 45}, ...
       */
      _domain [ DSZ_IDX() ] -= (_domain[ ub_idx + 1 ] - _domain[ UB_IDX() ]);
      _domain[ ub_idx + 1 ] = _domain[ UB_IDX() ];
    }
    
    /*
     * Copy on the left all the elements from the
     * current pair to upper bound.
     * @note this is done only if lb_idx > 0
     * otherwise the bounds are already in their final positon
     * and there is no case 4 to consider.
     * In what follows we consider the following example:
     * {5, 10}, {20, 23}, {42, 47}, {50, 53}.
     */
    if ( lb_idx > 0 ) {
      /*
       * lower bound = 20, upper bound = 47 -> 
       * {20, 23}, {42, 47}
       */
      for ( int i = 0; i < num_valid_pairs; i++ ) {
        _domain[ BIT_IDX() + 2 * i     ] = _domain[ BIT_IDX() + 2 * (lb_idx + i) ];
        _domain[ BIT_IDX() + 2 * i + 1 ] = _domain[ BIT_IDX() + 2 * (ub_idx + i) + 1 ];
      }//i
    }//lb_idx
    
    // Set new lower bound / upper bound if domain is reduced
    if ( _domain [ LB_IDX() ] < _domain[ BIT_IDX()     ] ) {
      _domain [ LB_IDX() ] = _domain[ BIT_IDX() ];
    }
    if ( _domain [ UB_IDX() ] > _domain[ BIT_IDX() + 1 ] ) {
      _domain [ UB_IDX() ] = _domain[ BIT_IDX() + 1 ];
    }
    
  }//num_pairs
  
  /*
   * Check new domain size:
   * if the sum of elements is <= VECTOR_MAX ->
   * switch representation to bitmap list.
   */
  if ( _domain [ DSZ_IDX() ] <= VECTOR_MAX ) {
    switch_list_to_bitmaplist ();
  }
}//update_list

void
CudaDomain::switch_list_to_bitmaplist () {
  
  // Consistency check
  if ( _domain [ REP_IDX() ] <= 0 ) return;
  
  // Set number of bitmaps
  int num_pairs = _domain [ REP_IDX() ];
  _domain [ REP_IDX() ] *= -1;
  
  // Avoid useless copies
  if ( num_pairs == 1 ) {
    prepare_bit_list ( _domain[ LB_IDX () ],
                       _domain[ UB_IDX () ],
                       BIT_IDX() );
    return;
  }
  
  // Copy pairs before updating BIT field.
  int * all_pairs = new int[ 2 * num_pairs ];
  memcpy( all_pairs, &_domain[ BIT_IDX()], 2 * num_pairs * sizeof(int) );
  
  // Prepare the corresponding bitmap list: first pair
  prepare_bit_list ( _domain[ LB_IDX () ],
                     _domain[ UB_IDX () ],
                     BIT_IDX() );
  
  // Prepare the corresponding bitmap list: remaining pairs
  int list_size = _domain[ UB_IDX () ] - _domain[ LB_IDX () ] + 1;
  int start_idx = BIT_IDX() + 2 + num_chunks( list_size );
  for ( int i = 0; i < num_pairs; i++ ) {
    prepare_bit_list ( all_pairs[ 2*i     ],
                       all_pairs[ 2*i + 1 ],
                       start_idx );
    list_size = all_pairs[ 2*i + 1 ] - all_pairs[ 2*i ] + 1;
    start_idx += 2 + num_chunks( list_size );
  }//num_pairs
}//switch_list_to_bitmap

void
CudaDomain::prepare_bit_list ( int min , int max, int idx ) {
  // Set lower bound, and upper bowund at the correct index
  _domain [ idx ]     = min;
  _domain [ idx + 1 ] = max;

  // Prepare bitlist
  int d_size = max - min + 1;
  int n_chunks = num_chunks( d_size );
  for ( int i = idx + 2; i <= idx + 2 + n_chunks; i++ ) {
    _domain[ i ] = 0xffffffff;
  }
  
  /* 
   * Update bitmap.
   * @note min is the lower bound for the current
   * bitmap representing the pair {min, max}, i.e.,
   * the pair has an offset of min from 0.
   * Therefore, {min, max} is "shifted back"
   * of min numbers in the bitmap representation.
   * {min, max} -> {0, max - min}
   */
  update_bitmap ( 0, max - min, idx + 2 );
}//prepare_bit_list

bool
CudaDomain::set_singleton ( int ) {
  return true;
}//set_singleton

bool
CudaDomain::subtract ( int ) {
  return true;
}//subtract

void
CudaDomain::add_element ( int ) {
  
}//add_element

void
CudaDomain::in_min ( int ) {
  
}//in_min

void
CudaDomain::in_max ( int ) {
  
}//in_max

void
CudaDomain::print () const {
  cout << "- CudaDomain:\n";
  cout << "EVT:\t";
  switch ( _domain[ EVT_IDX() ] ) {
    case INT_NO_EVT:
      cout << "NO Event\n";
      break;
    case INT_SINGLETON_EVT:
      cout << "Singleton Event\n";
      break;
    case INT_BOUNDS_EVT:
      cout << "Bounds Event\n";
      break;
    case INT_CHANGE_EVT:
      cout << "Change Event\n";
      break;
    case INT_FAIL_EVT:
      cout << "Fail Event\n";
      break;
    default:
      cout << "Other Event\n";
      break;
  }
  cout << "REP:\t";
  if ( get_representation()  == CudaDomainRepresenation::BITMAP ) {
    cout << "Bitmap\n";
  }
  else if ( get_representation()  == CudaDomainRepresenation::BITMAP_LIST ){
    cout << "Bitmap lists\n";
  }
  else {
    cout << "List\n";
  }
  cout << "LB-UB:\t";
  cout << "[" << _lower_bound << ".." << _upper_bound << "]\n";
  cout << "DSZ:\t";
  cout << get_size() << "\n";
  cout << "Bytes:\t";
  if ( get_allocated_bytes () < 1024 )
    cout << get_allocated_bytes () << "Bytes\n";
  else
    cout << (get_allocated_bytes () / 1024) << "kB\n";
  print_domain ();
}//print

void
CudaDomain::print_domain () const {
  cout << "|| ";
  for ( int i = 0; i < BIT_IDX(); i++ ) {
    cout << _domain[ i ];
    if ( i < BIT_IDX() - 1 ) cout << " | ";
    else cout << " || ";
  }
  // Print according to the internal representatation of domain
  if ( get_representation()  == CudaDomainRepresenation::BITMAP ) {
    for ( int i = BIT_IDX(); i <  BIT_IDX() + _num_int_chunks; i++ ) {
      if ( _domain [ i ] ) {
        CudaBitUtils::print_bit_rep ( _domain [ i ] );
        cout << " ";
      }
    }
  }
  else if ( get_representation() == CudaDomainRepresenation::BITMAP_LIST ) {
    int num_bitmap = -1 * _domain[ REP_IDX() ];
    for ( int i = 0; i < num_bitmap; i++ ) {
      cout << "{" << _domain [ BIT_IDX() + 2 * i ] << ", " <<
      _domain [ BIT_IDX() + 2 * i + 1 ] << "}: ";
      int size_bitmap = num_chunks ( _domain [ BIT_IDX() + 2 * i + 1 ] -
                                     _domain [ BIT_IDX() + 2 * i     ] + 1 );
      for ( int j = 0; j < size_bitmap; j++ ) {
        CudaBitUtils::print_0x_rep ( _domain [ BIT_IDX() + 2 * i + 2 + j ] );
        cout << " ";
      }
    }
  }
  else {
    if ( _domain[ REP_IDX() ] == 1 ) {
      cout << "{" << _domain[ LB_IDX() ] << ", " <<
      _domain[ UB_IDX() ] << "}";
    }
    else {
      for ( int i = 1; i < _domain[ REP_IDX() ]; i++ ) {
        cout << "{" << _domain [ BIT_IDX() + 2 * i ] << ", " <<
        _domain [ BIT_IDX() + 2 * i + 1 ] << "} ";
      }
    }
  }
  cout << "||\n";
}//print_domain





