//
//  cuda_concrete_list.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 15/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_concrete_list.h"

CudaConcreteDomainList::CudaConcreteDomainList ( size_t size, int min, int max ) :
CudaConcreteDomain ( size ),
_num_pairs         ( 0 ),
_domain_size       ( 0 ) {
  _dbg = "CudaConcreteDomainList - ";
  
  // Empty domain if not consistent
  if ( max < min ) {
    set_empty ();
    return;
  }
  
  _max_allowed_pairs = _num_chunks / 2;
  if ( _max_allowed_pairs < 1 ) {
    set_empty ();
    return;
  }
  
  // Initialize bitmap
  for ( int i = 0; i < _num_chunks; i++ ) {
    _concrete_domain[ i ] = 0x0;
  }
  
  // Set current lower/upper bounds
  _lower_bound = min;
  _upper_bound = max;
  
  // Set initial lower/upper bounds
  _init_lower_bound = _lower_bound;
  _init_upper_bound = _upper_bound;
  
  // Add the new pair of bounds to the list
  add( min, max );
}//CudaConcreteDomainList

unsigned int
CudaConcreteDomainList::size () const {
  return _domain_size;
}//size

int
CudaConcreteDomainList::find_pair ( int val ) const {
  // Scan the pairs to check for val
  for ( int pair_idx = 0; pair_idx < _num_pairs; pair_idx++ ) {
    if ( val >= _concrete_domain [ 2 * pair_idx     ] &&
         val <= _concrete_domain [ 2 * pair_idx + 1 ]) {
      return  pair_idx;
    }
  }
  return -1;
}//find_pair

int
CudaConcreteDomainList::find_prev_pair ( int val ) const {
  for ( int pair_idx = 0; pair_idx < _num_pairs - 1; pair_idx++ ) {
    if ( (_concrete_domain[ 2 * pair_idx + 1   ] < val) &&
         (_concrete_domain[ 2 * (pair_idx + 1) ] > val) ) {
      return  pair_idx;
    }
  }
  return -1;
}//find_prev_pair

int
CudaConcreteDomainList::find_next_pair ( int val ) const {
  for ( int pair_idx = 1; pair_idx < _num_pairs; pair_idx++ ) {
    if ( (_concrete_domain[ 2 * pair_idx           ] > val) &&
         (_concrete_domain[ 2 * (pair_idx - 1) + 1 ] < val) ) {
      return  pair_idx;
    }
  }
  return -1;
}//find_prev_pair

void
CudaConcreteDomainList::shrink ( int min, int max ) {
  
  // Empty domain if not consistent
  if ( max < min ) {
    flush_domain ();
    return;
  }
  
  // Check min/max value consistency
  if ( min == _lower_bound && max == _upper_bound ) return;
  
  // Return if no chages in the domain
  if ( (min <= _lower_bound) &&
       (max >= _upper_bound) ) {
    return;
  }
  
  // Set min/max w.r.t. the current bounds
  if ( (min < _lower_bound) &&
       (max < _upper_bound) ) {
    min = _lower_bound;
  }
  
  if ( (min > _lower_bound) &&
       (max > _upper_bound) ) {
    max = _upper_bound;
  }

  /*
   * There are _num_pairs pairs of contiguous elements.
   * Four cases to consider for both the new lower/upper bounds:
   * 1) new bound is equal to a lower bound
   * 2) new bound is equal to an upper bound
   * 3) new bound is within a pair {lower bound, upper bound}
   * 4) new bound is betweent to pair of bounds
   */
  // Size (num of elements) of the current pair
  int pair_size = 0;
  
  // Prev (next) sum of vals w.r.t. min (max)
  int  sum_prev_elements = 0;
  int  sum_next_elements = 0;
  
  // Index of the pair containing min (max)
  int pair_with_min = -1;
  int pair_with_max = -1;
  
  /*
   * State whether the sum of vals < min (vals > max)
   * is calculated (needed for calc. domain's size)
   */
  bool sum_prev_vals_complete = false;
  bool sum_next_vals_complete = true;
  
  // Number of valid pairs
  int num_pairs_in_domain = 0;
  for ( int pair_idx = 0; pair_idx <= _num_pairs; pair_idx++ ) {
    if ( (min >= _concrete_domain[ 2 * pair_idx     ]) &&
         (min <= _concrete_domain[ 2 * pair_idx + 1 ]) ) {
      pair_with_min = pair_idx;
      sum_prev_vals_complete = true;
    }
    if ( !sum_prev_vals_complete ) {
      sum_prev_elements += pair_size;
    }
    
    if ( sum_prev_vals_complete && (!sum_next_vals_complete)) {
      num_pairs_in_domain += 1;
    }
    
    pair_size =
    _concrete_domain[ 2 * pair_idx + 1 ] -
    _concrete_domain[ 2 * pair_idx     ] + 1;
    
    if ( (max >= _concrete_domain[ 2 * pair_idx     ]) &&
         (max <= _concrete_domain[ 2 * pair_idx + 1 ]) ) {
      pair_with_max = pair_idx;
      sum_next_vals_complete = false;
    }
    if ( !sum_next_vals_complete ) {
      sum_next_elements += pair_size;
    }
    
    /*
     * Check when min (max) is between to pairs 
     * (add current size if it is the case and stop adding).
     */
    if ( pair_idx < _num_pairs ) {
      if ( (min > _concrete_domain[ 2 * pair_idx + 1   ]) &&
           (min < _concrete_domain[ 2 * (pair_idx + 1) ]) ) {
        pair_with_min = pair_idx + 1;
        sum_prev_elements += pair_size;
        sum_prev_vals_complete = true;
      }
      
      if ( (max > _concrete_domain[ 2 * pair_idx + 1   ]) &&
           (max < _concrete_domain[ 2 * (pair_idx + 1) ]) ) {
        pair_with_max = pair_idx;
        sum_next_vals_complete = false;
      }
    }
  }//pair
  
  /*
   * Fail (empty domain) if no pairs are valid.
   * For example: min = 8, max = 10 in
   * {2, 7}, {15, 20}.
   */
  if ( num_pairs_in_domain == 0 ) {
    flush_domain ();
    return;
  }
  // Set new number of valid pairs
  _num_pairs = num_pairs_in_domain;
  
  // Reduce total size
  _domain_size -= (sum_prev_elements + sum_next_elements);
  /*
   * Reduce further if min(max) is within a range
   * and change the pair itself.
   * For example: min = 4 in {2, 8} -> {4, 8} and 
   * reduce domain's size by 2.
   */
  if ( (min >  _concrete_domain[ 2 * pair_with_min    ]) &&
       (min <= _concrete_domain[ 2 * pair_with_min + 1])) {
    _domain_size -= (min - _concrete_domain[ 2 * pair_with_min ] );
    _concrete_domain[ 2 * pair_with_min ] = min;
  }
  
  if ( (max <  _concrete_domain[ 2 * pair_with_max + 1]) &&
       (min >= _concrete_domain[ 2 * pair_with_max    ])) {
    _domain_size -= (_concrete_domain[ 2 * pair_with_max + 1] - max );
    _concrete_domain[ 2 * pair_with_max + 1 ] = max;
  }
  
  /*
   * Translate the pairs on the left:
   * {1, 4} {6, 10} {22, 25} (min = 8, max = 23) ->
   * {8, 10}, {23, 25}.
   */
  if ( pair_with_min > 0 ) {
    memcpy( _concrete_domain,
            &_concrete_domain[ 2 * pair_with_min ],
            2 * _num_pairs * sizeof( int ) );
  }
  
  // Update bounds
  _lower_bound = _concrete_domain[ 0 ];
  _upper_bound = _concrete_domain[ 2 * _num_pairs + 1 ];
}//shrink

void
CudaConcreteDomainList::in_min ( int min ) {
  if ( min <= _lower_bound ) return;
  shrink ( min, _upper_bound );
}//in_min

void
CudaConcreteDomainList::in_max ( int max ) {
  if ( max >= _upper_bound ) return;
  shrink ( _lower_bound, max );
}//in_min

void
CudaConcreteDomainList::add ( int value ) {
  
  // Consistency check
  if ( value < _init_lower_bound ) value = _init_lower_bound;
  if ( value > _init_upper_bound ) value = _init_upper_bound;
  
  // Return if the value is already set to 1
  if ( contains( value ) ) return;
  
  add ( value, value );
}//add

void
CudaConcreteDomainList::add ( int min, int max ) {
  
  // Consistency check
  if ( max < min ) return;
  
  // Find the pair containing min (max)
  int lower_pair = find_pair ( min );
  int upper_pair = find_pair ( max );
  
  /*
   * Example:
   * min = 13, max = 35
   * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   * {1, 4}, {10, 37}, {40, 50}
   * where
   * _num_pairs       = 5
   * lower_pair       = 1
   * upper_pair       = 3
   * num_pairs_within = 1 (i.e., {20, 25})
   * num_pairs_right  = 1 (i.e., {40, 50})
   */
  if ( lower_pair && upper_pair ) {
    
    // No action if min/max are in the same pair
    if ( lower_pair == upper_pair ) return;
    
    int num_pairs_within = upper_pair - lower_pair - 1;
    
    // Find the number of elements to add
    int sum_elements = 0;
    for ( int pair_idx = lower_pair + 1; pair_idx <= upper_pair; pair_idx++ ) {
      sum_elements +=
      _concrete_domain[ 2 * pair_idx           ] -
      _concrete_domain[ 2 * (pair_idx - 1) + 1 ] - 1;
    }//pair_idx
    
    // Reset domain's size
    _domain_size += sum_elements;
    
    // Reset pairs and move to the left pairs > upper_pair
    int value_to_set    = _concrete_domain [ 2 * upper_pair + 1 ];
    int num_pairs_right = _num_pairs - upper_pair - 1;
    if ( num_pairs_right )
      memcpy( &_concrete_domain [ 2 * (lower_pair + 1) ],
              &_concrete_domain [ 2 * (upper_pair + 1) ] ,
              2 * num_pairs_right * sizeof( int ) );
    
    _concrete_domain [ 2 * lower_pair + 1 ] = value_to_set;
    
    // Fix total num pairs
    _num_pairs -= (num_pairs_within + 1);
    return;
  }
  
  /*
   * Example:
   * min = 13, max = 38
   * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   * {1, 4}, {10, 38}, {40, 50}
   * where
   * _num_pairs       = 5
   * prev_pair_idx    = 3
   * lower_pair       = 1
   * num_pairs_within = 1 (i.e., {20, 25})
   */
  if ( lower_pair ) {
    
    // Find prev pair and add elements to it
    int prev_pair_idx = find_prev_pair ( max );
    
    // Find the number of elements to add
    int sum_elements = 0;
    for ( int pair_idx = lower_pair + 1; pair_idx < prev_pair_idx; pair_idx++ ) {
      sum_elements +=
      _concrete_domain[ 2 * pair_idx           ] -
      _concrete_domain[ 2 * (pair_idx - 1) + 1 ] - 1;
    }//pair_idx
    
    sum_elements += max - _concrete_domain[ 2 * prev_pair_idx + 1 ];
    
    // Reset domain's size
    _domain_size += sum_elements;
 
    // Reset pairs and move to the left pairs > upper_pair
    int num_pairs_within = prev_pair_idx - lower_pair - 1;
    int num_pairs_right = _num_pairs - prev_pair_idx - 1;
    
    if ( num_pairs_right )
      memcpy( &_concrete_domain [ 2 * (lower_pair + 1)  ],
              &_concrete_domain [ 2 * (prev_pair_idx + 1) ] ,
              2 * num_pairs_right * sizeof( int ) );
    
    _concrete_domain [ 2 * lower_pair + 1 ] = max;
    
    // Fix total num pairs
    _num_pairs -= (num_pairs_within + 1);
    return;
  }
  
  /*
   * Example:
   * min = 18, max = 36
   * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   * {1, 4}, {10, 15}, {18, 37}, {40, 50}
   * where
   * _num_pairs       = 5
   * next_pair_idx    = 2
   * upper_pair       = 3
   * num_pairs_right  = 1
   * num_pairs_within = 0
   */
  if ( upper_pair ) {
    int next_pair_idx = find_next_pair ( min );
    
    // Find the number of elements to add
    int sum_elements = 0;
    for ( int pair_idx = next_pair_idx + 1; pair_idx <= upper_pair; pair_idx++ ) {
      sum_elements +=
      _concrete_domain[ 2 * pair_idx           ] -
      _concrete_domain[ 2 * (pair_idx - 1) + 1 ] - 1;
    }//pair_idx
    
    sum_elements += _concrete_domain[ 2 * next_pair_idx ] - min;
    
    // Reset domain's size
    _domain_size += sum_elements;
    
    // Reset pairs and move to the left pairs > upper_pair
    int num_pairs_within = upper_pair - next_pair_idx - 1;
    int num_pairs_right = _num_pairs - upper_pair - 1;
    
    // Copy only if there are pairs on the right to copy
    if ( num_pairs_right )
      memcpy( &_concrete_domain [ 2 * (next_pair_idx + 1) ],
              &_concrete_domain [ 2 * (upper_pair + 1)    ] ,
              2 * num_pairs_right * sizeof( int ) );
    
    _concrete_domain [ 2 * next_pair_idx ] = min;
    
    // Fix total num pairs
    _num_pairs -= (num_pairs_within + 1);
    return;
  }
  
  /*
   * Example:
   * min = 18, max = 38
   * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   * {1, 4}, {10, 15}, {18, 38}, {40, 50}
   * where
   * _num_pairs       = 5
   * prev_pair_idx    = 3
   * next_pair_idx    = 2
   * num_pairs_within = 1
   * num_pairs_right  = 1
   */
  if ( lower_pair < 0 && upper_pair < 0 ) {
    int next_pair_idx = find_next_pair ( min );
    int prev_pair_idx = find_next_pair ( max );
    
    // Check if min/max between the same two pairs
    if ( next_pair_idx > prev_pair_idx ) {
      /* Add a new pair
       * min = 18, max = 19
       * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
       * {1, 4}, {10, 15}, {18, 19}, {20, 25}, {30, 37}, {40, 50}
       */
      if ( _num_pairs + 1 > _max_allowed_pairs ) {
        logger->error( _dbg + "Can't add another pair",
                       __FILE__, __LINE__ );
        set_empty ();
        return;
      }
      
      // Shift domain on the right
      int num_pairs_right = _num_pairs - prev_pair_idx - 1;
      if ( num_pairs_right )
        memcpy( &_concrete_domain[ 2 * (next_pair_idx + 1) ],
                &_concrete_domain[ 2 * (prev_pair_idx + 1) ],
                2 * num_pairs_right * sizeof( int ) );
      
      _concrete_domain[ 2 * next_pair_idx     ] = min;
      _concrete_domain[ 2 * next_pair_idx + 1 ] = max;
      
      _num_pairs++;
      _domain_size += max - min + 1;
      return;
    }
    
    // Find the number of elements to add
    int sum_elements = 0;
    for ( int pair_idx = next_pair_idx + 1;
          pair_idx <= upper_pair; pair_idx++ ) {
      sum_elements +=
      _concrete_domain[ 2 * pair_idx           ] -
      _concrete_domain[ 2 * (pair_idx - 1) + 1 ] - 1;
    }//pair_idx
    
    sum_elements += _concrete_domain[ 2 * next_pair_idx ] - min;
    sum_elements +=  max - _concrete_domain[ 2 * prev_pair_idx ];
    
    // Reset domain's size
    _domain_size += sum_elements;
    
    // Reset pairs and move to the left pairs > upper_pair
    int num_pairs_within = prev_pair_idx - next_pair_idx;
    int num_pairs_right = _num_pairs - prev_pair_idx - 1;
    
    // Copy only if there are pairs on the right to copy
    if ( num_pairs_right )
      memcpy( &_concrete_domain [ 2 * (next_pair_idx + 1) ],
              &_concrete_domain [ 2 * (prev_pair_idx + 1) ] ,
              2 * num_pairs_right * sizeof( int ) );
    
    _concrete_domain [ 2 * next_pair_idx     ] = min;
    _concrete_domain [ 2 * next_pair_idx + 1 ] = max;
    
    // Fix total num pairs
    _num_pairs -= num_pairs_within;
  }
  
  // Update bounds
  if ( min < _lower_bound ) _lower_bound = min;
  if ( max > _upper_bound ) _upper_bound = max;
}//add

bool
CudaConcreteDomainList::contains ( int val ) const {
  return  ( find_pair( val ) != -1 );
}//contains

bool
CudaConcreteDomainList::is_singleton () const {
  if ( _num_pairs == 1 ) {
    return (_lower_bound == _upper_bound);
  }
  return  false;
}//is_singleton

int
CudaConcreteDomainList::get_singleton () const {
  if ( !is_singleton() ) {
    throw  NvdException ( ( _dbg + "Domain not singleton" ).c_str() );
  }
  
  return _lower_bound;
}//get_singleton

const void *
CudaConcreteDomainList::get_representation () const {
  return (void *) _concrete_domain;
}//get_representation

void
CudaConcreteDomainList::print () const {
  for ( int pair_idx = 0; pair_idx < _num_pairs; pair_idx++ ) {
    std::cout <<
    "{" <<
    _concrete_domain [ 2 * pair_idx     ] <<
    ", " <<
    _concrete_domain [ 2 * pair_idx + 1 ] <<
    "} ";
  }
}//print



