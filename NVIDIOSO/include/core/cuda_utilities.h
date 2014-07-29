//
//  cuda_utilities.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 11/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This namespace implements some utilities and functions (especially on
//  bitmaps) that can be useful when operating in CUDA.
//

#ifndef NVIDIOSO_cuda_utilities_h
#define NVIDIOSO_cuda_utilities_h

typedef unsigned int uint;

namespace CudaBitUtils {
  
  /**
   * Check whether a given number n
   * is a power of 2.
   * @param n unsigned integer
   * @result true iff n = 2^x for some x >= 0
   */
  inline bool is_2pow ( uint n ) {
    return ( (n & (n-1)) == 0 );
  }//is_2pow
  
  /**
   * Counts the number of bits that are set to 1
   * given a unsigned int as input.
   */
  inline uint num_1bit ( uint n ) {
    uint c = 0;
    for ( c = 0; n; c++ )
      n &= n - 1; // clear the least significant bit set
    return c;
  }//num_ones
  
  /**
   * Counts the number of bits that are set to 1 
   * from bit 0 to i (inclusive).
   */
  inline uint num_1bit_0_through_i ( uint n, int i ) {
    if (  i < 0 || i > 31 ) return -1;
    uint c = 0;
    uint sum = 0;
    for ( c = 0; c <= i; c++ ) {
      if ( (n & (1 << i)) != 0 ) sum++;
    }
    return sum;
  }//num_ones
  
  /**
   * Counts the consecutive zero (trailing)
   * on the right in parallel.
   */
  inline uint trailing_right ( uint n ) {
    uint c = 32; // c will be the number of zero bits on the right
    n &= -signed(n);
    if (n) c--;
    if (n & 0x0000FFFF) c -= 16;
    if (n & 0x00FF00FF) c -= 8;
    if (n & 0x0F0F0F0F) c -= 4;
    if (n & 0x33333333) c -= 2;
    if (n & 0x55555555) c -= 1;
    return c;
  }//trailing_right
  
  /**
   * Get the bit value of the bit in position i of
   * the given unsigned int.
   * @param n usigned int to check the bit
   * @param i position of the bit to check
   * @return true iff n[i] = 1
   */
  inline bool get_bit ( uint n, int i ) {
    if (  i < 0 || i > 31 ) return false;
    return ( (n & (1 << i)) != 0 );
  }//get_bit
  
  //! Set the i^th bit to 1 and return the modified input val
  inline uint set_bit ( uint n, int i ) {
    if ( i < 0 || i > 31 ) return n;
    return n | ( 1 << i );
  }//set_bit
  
  //! Clear the i^th bit and return the modified input val
  inline uint clear_bit ( uint n, int i ) {
    if ( i < 0 || i > 31 ) return i;
    return n &  (~(1 << i));
  }//clear_bit
  
  //! Clear all bits from the MSB through i (inclusive)
  inline uint clear_bits_MSB_through_i ( uint n, int i ) {
    if (  i < 0 || i > 31 ) return n;
    return n & ( (1 << i) - 1 );
  }//clear_bits_MSB_through_i
  
  //! Clear all bits from i through 0 (inclusive)
  inline uint clear_bits_i_through_0 ( uint n, int i ) {
    if (  i < 0 || i > 31 ) return n;
    int mask = ~((1 << (i+1)) - 1);
    return n & mask;
  }//clear_bits_i_through_0
  
  /**
   * It updates a given bit in an unsigned int.
   * @param n the unsigned int to update
   * @param i the bit to update
   * @param v the value to set to the i^th bit of n
   * @return n' = (n[i] = v)
   */
  inline uint update_bit ( uint n, int i, int v ) {
    if (  i < 0 || i > 31 )         return n;
    if ( v < 0 || v > 1 ) return n;
    int mask = ~(1 << i );
    return  ( n & mask ) | ( v << i );
  }//update_bit
  
  /**
   * Print the bitmap representation of the
   * number given in input.
   * @param n a 32bit value
   */
  inline void print_bit_rep ( uint n ) {
    for ( int i = 0; i < 32; i++ ) {
      if ( n & 1<< (32 - i - 1) ) std::cout << "1";
      else                        std::cout << "0";
    }
  }//print_bitmap
  
  /**
   * Print the 0x representation of the
   * number given in input.
   * @note not used itoa here for compatibility reasons.
   * @param n a 32bit value
   */
  inline void print_0x_rep ( uint n ) {
    char buffer [ 50 ];
    sprintf( buffer, "0x%x", n );
    std::cout << buffer;
  }//print_bitmap
  
  
}//CudaBitUtils


#endif
