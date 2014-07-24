//
//  cuda_concrete_bitmaplist.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 17/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_concrete_bitmaplist.h"
#include "cuda_utilities.h"

CudaConcreteBitmapList::CudaConcreteBitmapList ( size_t size,
                                                 std::vector< std::pair <int, int> > pairs ) :
CudaConcreteDomainBitmap ( size ) {
  _dbg = "CudaConcreteBitmapList - ";
  
  // Consistency check
  if ( pairs.size() == 0 ) {
    set_empty ();
    return;
  }
  
  /*
   * Find max span among all the pairs {LB, UB} and 
   * calculate the current domain's size.
   * This will be the fixed size of each bitmap.
   */
  _domain_size = 0;
  int max_bitmap_size = -1;
  for ( auto x : pairs ) {
    _domain_size += x.second - x.first + 1;
    if ( (x.second - x.first + 1) > max_bitmap_size )
      max_bitmap_size = (x.second - x.first + 1);
  }
  
  _bitmap_size = NUM_CHUNKS ( max_bitmap_size );
  
  // Check minimum space requirements
  if ( _num_chunks < pairs.size() * (2 + _bitmap_size) ) {
    set_empty ();
    return;
  }
  
  // Set current lower/upper bounds
  _lower_bound = pairs[ 0 ].first;
  _upper_bound = pairs[ pairs.size () - 1 ].second;
  
  _init_lower_bound = _lower_bound;
  _init_upper_bound = _upper_bound;
  
  // If everything went well, add the pairs to the bitmap list
  _num_bitmaps = 0;
  for ( auto x : pairs )
    add ( x.first , x.second );
}//CudaConcreteBitmapList

unsigned int
CudaConcreteBitmapList::size () const {
  return _domain_size;
}//size

int
CudaConcreteBitmapList::pair_to_idx ( int i ) const {
  return i * (2 + _bitmap_size);
}//pair_to_idx

int
CudaConcreteBitmapList::find_pair ( int val ) const {
  
  // Consistency check
  if ( val < _init_lower_bound || val > _init_upper_bound ) return -1;
  
  int idx;
  for ( int pair_idx = 0; pair_idx < _num_bitmaps; pair_idx++ ) {
    idx = pair_to_idx ( pair_idx );
    if ( (val >= _concrete_domain[ idx     ]) &&
         (val <= _concrete_domain[ idx + 1 ]) ) {
      return  pair_idx;
    }
  }
  return -1;
}//find_bitmap

int
CudaConcreteBitmapList::find_prev_pair ( int val ) const {
  
  // Consistency check
  if ( val < _init_lower_bound || val > _init_upper_bound ) return -1;
  
  int idx_prev;
  int idx_curr;
  for ( int pair_idx = 1; pair_idx < _num_bitmaps; pair_idx++ ) {
    idx_prev = pair_to_idx ( pair_idx - 1 );
    idx_curr = pair_to_idx ( pair_idx     );
    if ( (val > _concrete_domain[ idx_prev + 1 ]) &&
         (val < _concrete_domain[ idx_curr     ]) ) {
      return  pair_idx;
    }
  }
  return -1;
}//find_prev_pair

int
CudaConcreteBitmapList::find_next_pair ( int val ) const {
  
  // Consistency check
  if ( val < _init_lower_bound || val > _init_upper_bound ) return -1;
  
  int idx_curr;
  int idx_next;
  for ( int pair_idx = 0; pair_idx < _num_bitmaps - 1; pair_idx++ ) {
    idx_curr = pair_to_idx ( pair_idx     );
    idx_next = pair_to_idx ( pair_idx + 1 );
    if ( (val > _concrete_domain[ idx_curr + 1 ]) &&
         (val < _concrete_domain[ idx_next     ]) ) {
      return  pair_idx + 1;
    }
  }
  return -1;
}//find_prev_pair

void
CudaConcreteBitmapList::shrink ( int min, int max ) {
  
  // Empty domain if not consistent
  if ( max < min ) {
    flush_domain ();
    return;
  }
  
  // Return if no reduction must be performed
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
  
  // Find the pairs of bounds containing min/max
  int bitmap_idx_min = find_pair ( min );
  int bitmap_idx_max = find_pair ( max );
  
  // If present update bitmap with min
  if ( bitmap_idx_min ) {
    int idx = pair_to_idx ( bitmap_idx_min );
    
    /*
     * Find the chunk containing min
     * @note chunks from right to left (min is on the right).
     */
    int chunk = IDX_CHUNK ( min - _concrete_domain[ idx ] );//_bitmap_size - 1 - chunk
    chunk = _bitmap_size - 1 - chunk;
    
    // Set to 0 previous chunks (i.e., chuncks on the right)
    for ( int i = _num_bitmaps - 1; i > chunk; i-- )
      _concrete_domain[ idx + 2 + i ] = 0;

    // Set 0 bits on the right in the chunk^th chunk
    _concrete_domain[ idx + 2 + chunk ] =
    CudaBitUtils::clear_bits_i_through_0 ( (unsigned int) _concrete_domain[ chunk ],
                                           IDX_BIT( min ) - 1 );
  }
  
  // If present update bitmap with max
  if ( bitmap_idx_max ) {
    int idx = pair_to_idx ( bitmap_idx_max );
    
    // Find the chunk containing max
    int chunk = IDX_CHUNK ( _concrete_domain[ idx + 1 ] - max );
    chunk = _bitmap_size - 1 - chunk;
    
    // Set to 0 previous chunks (i.e., chuncks on the left)
    for ( int i = 0; i < chunk; i++ )
      _concrete_domain[ idx + 2 + i ] = 0;
    
    // Set 0 bits on the left in the chunk^th chunk
    _concrete_domain[ idx + 2 + chunk ] =
    CudaBitUtils::clear_bits_MSB_through_i ( (unsigned int)
                                              _concrete_domain[ idx + 2 + chunk ],
                                              IDX_BIT( max ) + 1 );
  }
  
  // If not present set next for min and prev for max
  if ( bitmap_idx_min < 0 ) bitmap_idx_min = find_next_pair( min );
  if ( bitmap_idx_max < 0 ) bitmap_idx_max = find_prev_pair( max );
  
  /*
   * Check whether min/max belong to the same empty set
   * in such a case, exit with empty domain.
   */
  if ( bitmap_idx_max < bitmap_idx_min ) {
    flush_domain ();
    return;
  }
  
  /*
   * Recompute domain's size:
   * count all the bits set to 1 in all the 
   * bitmaps between bitmap_idx_min and bitmap_idx_max.
   * @note shrink is performed considering min/max when consistency check on 
   *       this bounds is already been performed above. This implies that
   *       bitmap_idx_min/bitmap_idx_max are never -1;
   */
  assert( bitmap_idx_min >= 0 );
  assert( bitmap_idx_max >= 0 );
  
  _domain_size = 0;
  int idx;
  for ( int pair_idx = bitmap_idx_min; pair_idx <= bitmap_idx_max; pair_idx ++ ) {
    idx = pair_to_idx ( pair_idx );
    for ( int i = idx + 2; i < idx + 2 + _bitmap_size; i++ ) {
      if ( _concrete_domain[ i ] ) {
        _domain_size += CudaBitUtils::num_1bit ( (unsigned int) _concrete_domain[ i ] );
      }
    }
  }//idx
  
  // If domain's size is 0, empty domain
  if ( _domain_size == 0 ) {
    set_empty ();
    return;
  }
  
  // Update bounds
  if ( min > _lower_bound ) _lower_bound = min;
  if ( max < _upper_bound ) _upper_bound = max;
}//shrink

void
CudaConcreteBitmapList::in_min ( int min ) {
  shrink ( min, _upper_bound );
}//in_min

void
CudaConcreteBitmapList::in_max ( int max ) {
  shrink ( _lower_bound, max );
}//in_max

void
CudaConcreteBitmapList::add ( int value ) {
  
  // Consistency check
  if ( value < _init_lower_bound || value > _init_upper_bound ) return;
  
  // Find the pair containing the value
  int pair_idx = find_pair( value );
  
  /*
   * If such a pair exists, set the 
   * bit corresponding to value to 1.
   * Otherwise add a new pair {value, value}
   * with the correspondent bitmap and shift
   * to the right all the other values/bitmaps.
   */
  if ( pair_idx ) {
    int idx = pair_to_idx( pair_idx );
    
    // Find the chunk containing value
    int chunk = IDX_CHUNK ( value - _concrete_domain[ idx ] );
    
    // Set 1 bit corresponding to value on the right in the chunk^th chunk
    if ( CudaBitUtils::get_bit ( _concrete_domain[ idx + 2 + _bitmap_size - 1 - chunk ], IDX_BIT( value )) ) {
      return;
    }
    _concrete_domain[ idx + 2 + _bitmap_size - 1 - chunk ] =
    CudaBitUtils::set_bit ( (unsigned int) _concrete_domain[ idx + 2 + _bitmap_size - 1 - chunk ], IDX_BIT( value ));
    
    _domain_size++;
  }
  else {
    /*
     * Insert {value, value} beween two other pairs:
     * first copy in two pairs ahead everthing between
     * next pair and the end of BITMAP field.
     * Then replace next pair with the current singleton 
     * set plus its bitmap representation.
     */
    add ( value, value );
    
    
  }
  
  if ( value < _lower_bound ) _lower_bound = value;
  if ( value > _upper_bound ) _upper_bound = value;
}//add

void
CudaConcreteBitmapList::add ( int min, int max  ) {
  
  // Consistency check
  if ( min < _init_lower_bound || max > _init_upper_bound ) return;
  
  // Find position for {min, max}
  int pair_idx_min = find_pair( min );
  int pair_idx_max = find_pair( max );
  
  // Fix min/max
  if ( pair_idx_min >= 0 )
    min = _concrete_domain[ pair_to_idx( pair_idx_min ) ];
  
  if ( pair_idx_max >= 0 )
    max = _concrete_domain[ pair_to_idx( pair_idx_max ) + 1 ];

  // Prepare bitmap to substitute/add in the right position
  int * new_pair = new int[ 2 + _bitmap_size ];
  
  // Set its bounds
  new_pair[ 0 ] = min;
  new_pair[ 1 ] = max;
  for ( int i = 0; i < _bitmap_size; i++ ) new_pair[ 2 + i ] = 0;
  
  /*
   * Set values (bit = 1).
   * @note values are shifted back of min positions.
   */
  int chunk, bit;
  for ( int value = 0; value < max - min + 1; value++ ) {
    chunk = IDX_CHUNK ( value );
    bit   = IDX_BIT   ( value );
    new_pair[ 2 + _bitmap_size - 1 - chunk ] =
    CudaBitUtils::set_bit ( (unsigned int) new_pair[ 2 + _bitmap_size - 1 - chunk ], bit );
  }//idx
  
  // Copy everything on the right and replace with the new bitmap
  if ( (pair_idx_min < 0)  && (pair_idx_max < 0) ) {
    
    // Check next of min
    pair_idx_min = find_next_pair( min );
    if ( pair_idx_min < 0 ) {
      
      // There is nothing on the right: add at the end of the current list
      pair_idx_min = _num_bitmaps * ( 2 + _bitmap_size );
      memcpy( &_concrete_domain[ pair_idx_min ],
              new_pair, (2 + _bitmap_size) * sizeof( int ) );
      
      // Add num bitmaps and size of the current domain
      _num_bitmaps++;
      _domain_size += ( max -  min + 1 );
      
      if ( min < _lower_bound ) _lower_bound = min;
      if ( max > _upper_bound ) _upper_bound = max;
      
      delete [] new_pair;
      
      return;
    }
    else {
      // There is already something greater than min
      delete [] new_pair;
      throw NvdException ( "Not yet implemented!" );
    }
  }
  // Other cases to consider
  delete [] new_pair;
  throw NvdException ( "Not yet implemented!" );
}//add

bool
CudaConcreteBitmapList::contains ( int val ) const {
  if ( val < _lower_bound || val > _upper_bound ) return false;
  if ( find_pair ( val ) ) return true;
  return false;
}//contains

void
CudaConcreteBitmapList::print () const {
  int idx_pair;
  for ( int idx = 0; idx < _num_bitmaps; idx++ ) {
    idx_pair = pair_to_idx( idx );
    std::cout << "{" <<
    _concrete_domain [ idx_pair     ] << ", " <<
    _concrete_domain [ idx_pair + 1 ] << "}: ";
    for ( int j = 0; j < _bitmap_size; j++ ) {
      CudaBitUtils::print_0x_rep ( _concrete_domain [ idx_pair + 2 + j ] );
      std::cout << " ";
    }
  }//idx
}//print





