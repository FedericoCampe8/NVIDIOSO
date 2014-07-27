//
//  cuda_concrete_bitmaplist.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 17/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_concrete_bitmaplist.h"
#include "cuda_utilities.h"

using namespace std;

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
      return  pair_idx - 1;
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
  if ( min == _lower_bound &&
       max == _upper_bound ) return;
  
  // Return if no chages in the domain
  if ( (min <= _lower_bound) &&
       (max >= _upper_bound) ) {
    return;
  }
  
  // Set min w.r.t. the current bounds
  if ( (min < _lower_bound) &&
       (max < _upper_bound) ) {
    min = _lower_bound;
  }
  
  // Set max w.r.t. the current bounds
  if ( (min > _lower_bound) &&
       (max > _upper_bound) ) {
    max = _upper_bound;
  }
  
  // Find the pairs of bounds containing min/max
  int bitmap_idx_min = find_pair ( min );
  int bitmap_idx_max = find_pair ( max );
  
  // If min is within some bitmap: update bitmap
  if ( bitmap_idx_min >= 0 ) {
    int idx = pair_to_idx ( bitmap_idx_min );
    
    /*
     * Find the chunk containing min
     * @note chunks from right to left (min is on the right) for min >= 0, on the left for min < 0.
     * Example:
     * {0, 32}  -> |   63..32 |  31..0 |
     * {-32, 0} -> | -63..-32 | -31..0 |
     */
    int chunk = IDX_CHUNK ( min >= 0 ? min - _concrete_domain[ idx ] : abs ( min ) - abs ( _concrete_domain[ idx + 1 ] ) );
    
    // Find chunk starting from lsb
    chunk = _bitmap_size - 1 - chunk;

    // Set bitmap according to the new min value on the correspondend chunk
    if ( min >= 0 ) {

      // Set to 0 previous chunks (i.e., chuncks on the right)
      for ( int i = _bitmap_size - 1; i > chunk; i-- )
        _concrete_domain[ idx + 2 + i ] = 0;
      
      // Set 0 bits on the right in the chunk^th chunk
      _concrete_domain[ idx + 2 + chunk ] =
      CudaBitUtils::clear_bits_i_through_0 ( (unsigned int) _concrete_domain[ idx + 2 + chunk ],
                                             IDX_BIT( min - _concrete_domain[ idx ] ) - 1 );
    }
    else {

      /*
       * Min is on the left, clean bits starting from MSB
       * Set to 0 previous chunks (i.e., chuncks on the left)
       */
      for ( int i = 0; i < chunk; i++ )
        _concrete_domain[ idx + 2 + i ] = 0;

      // Set 0 bits on the left MSB through i in the chunk^th chunk
      int offset;
      if ( _concrete_domain[ idx ] < 0 && _concrete_domain[ idx + 1 ] < 0 ) {
        offset = _concrete_domain[ idx + 1 ];
      }
      else {
        offset = _concrete_domain[ idx ];
      }
        
      _concrete_domain[ idx + 2 + chunk ] =
      CudaBitUtils::clear_bits_MSB_through_i ( (unsigned int) _concrete_domain[ idx + 2 + chunk ],
                                               IDX_BIT( abs ( min - offset ) ) + 1 );
    }
  }//bitmap_idx_min
  
  // If min is within some bitmap: update bitmap
  if ( bitmap_idx_max >= 0 ) {
    int idx = pair_to_idx ( bitmap_idx_max );

    /*
     * Find the chunk containing max
     * @note chunks from right to left (max is on the left) for max >= 0, on the right for max < 0.
     * Example:
     * {0, 32}  -> |   63..32 |  31..0 |
     * {-32, 0} -> | -63..-32 | -31..0 |
     */
    int chunk = IDX_CHUNK ( _concrete_domain[ idx + 1 ] - max );

    // Find chunk starting from MSB
    chunk = _bitmap_size - 1 - chunk;
    
    // Set bitmap according to the new min value on the correspondend chunk
    if ( max >= 0 ) {
      
      // Set to 0 previous chunks (i.e., chuncks on the left)
      for ( int i = 0; i < chunk; i++ )
        _concrete_domain[ idx + 2 + i ] = 0;
      
      // Set 0 bits on the left in the chunk^th chunk
      _concrete_domain[ idx + 2 + chunk ] =
      CudaBitUtils::clear_bits_MSB_through_i ( (unsigned int)
                                               _concrete_domain[ idx + 2 + chunk ],
                                               IDX_BIT( max - _concrete_domain[ idx ] ) + 1 );
    }
    else {
      
      // Set to 0 previous chunks (i.e., chuncks on the right lsb)
      for ( int i = _bitmap_size - 1; i > chunk; i-- )
        _concrete_domain[ idx + 2 + i ] = 0;
      
      // Set 0 bits on the left in the chunk^th chunk
      int offset;
      if ( _concrete_domain[ idx ] < 0 && _concrete_domain[ idx + 1 ] < 0 ) {
        offset = _concrete_domain[ idx + 1 ];
      }
      else {
        offset = _concrete_domain[ idx ];
      }
      _concrete_domain[ idx + 2 + chunk ] =
      CudaBitUtils::clear_bits_i_through_0 ( (unsigned int) _concrete_domain[ idx + 2 + chunk ],
                                             IDX_BIT( abs ( max - offset ) ) - 1 );
      
    }
  }//bitmap_idx_max
  
  // If not present set next for min and prev for max
  if ( bitmap_idx_min < 0 ) bitmap_idx_min = find_next_pair( min );
  if ( bitmap_idx_max < 0 ) bitmap_idx_max = find_prev_pair( max );
  
  /*
   * Check whether min/max belong to the same empty set
   * in such a case, exit with empty domain.
   */
  if ( bitmap_idx_max < bitmap_idx_min ) {
    _domain_size = 0;
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
  
  // Recompute size from scratch
  _domain_size = 0;
  int idx;
  for ( int pair_idx = bitmap_idx_min; pair_idx <= bitmap_idx_max; pair_idx++ ) {
    idx = pair_to_idx ( pair_idx );
    
    // Compute number of bits set to 1 in each chunk
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
CudaConcreteBitmapList::subtract ( int value ) {
  
  // Consistency check
  if ( value < _lower_bound || value > _upper_bound ) return;
  
  int pair_idx = find_pair ( value );
  if ( pair_idx < 0 ) return;
  
  int shift_value;
  int idx = pair_to_idx( pair_idx );
  if ( _concrete_domain[ idx ] < 0 && _concrete_domain[ idx + 1 ] < 0 ) {
    shift_value = abs( value - _concrete_domain[ idx + 1 ] );
  }
  else {
    shift_value = abs( value - _concrete_domain[ idx ] );
  }
  
  assert ( shift_value >= 0 );
  int chunk = IDX_CHUNK ( shift_value );
  chunk = _bitmap_size - 1 - chunk;
  
  if ( !CudaBitUtils::get_bit( _concrete_domain[ idx + 2 + chunk ],
                              IDX_BIT( shift_value ) ) )
    return;
  
  _concrete_domain[ idx + 2 + chunk ] = CudaBitUtils::clear_bit( _concrete_domain[ idx + 2 + chunk ], IDX_BIT( shift_value ) );
  
  // Decrease number of valid bits
  _domain_size -= 1;
  
  // Check for empty domain
  if ( _domain_size == 0 ) {
    set_empty ();
    return;
  }
  
  //Check for singleton
  if ( _domain_size == 1 ) {
    if ( _lower_bound == value ) {
      _lower_bound = _upper_bound;
    }
    if ( _upper_bound == value ) {
      _upper_bound = _lower_bound;
    }
    return;
  }
  
  // Set new lower/upper bound
  if ( value == _lower_bound ) {
    while ( true ) {
      value++;
      if ( contains ( value ) ) {
        _lower_bound = value;
        return;
      }
    }
  }
  
  if ( value == _upper_bound ) {
    while ( true ) {
      value--;
      if ( contains( value ) ) {
        _upper_bound = value;
        return;
      }
    }
  }
}//subtract

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
  
  // Check if val is set to 1 in some bitmap
  int pair_idx = find_pair ( val );
  if ( pair_idx >= 0 ) {
    int shift_value;
    int idx = pair_to_idx( pair_idx );
    if ( _concrete_domain[ idx ] < 0 && _concrete_domain[ idx + 1 ] < 0 ) {
      shift_value = abs( val - _concrete_domain[ idx + 1 ] );
    }
    else {
      shift_value = abs( val - _concrete_domain[ idx ] );
    }
    
    assert ( shift_value >= 0 );
    int chunk = IDX_CHUNK ( shift_value );
    chunk = _bitmap_size - 1 - chunk;
    
    if ( CudaBitUtils::get_bit( _concrete_domain[ idx + 2 + chunk ],
                               IDX_BIT( shift_value ) ) )
      return true;
  }
  
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
      CudaBitUtils::print_bit_rep ( _concrete_domain [ idx_pair + 2 + j ] );
      std::cout << " ";
    }
  }//idx
}//print





