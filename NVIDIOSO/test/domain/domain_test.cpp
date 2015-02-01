//
//  domain_test.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 24/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "globals.h"
#include "cuda_domain.h"
#include "cuda_utilities.h"
#include "cuda_concrete_bitmap.h"
#include "cuda_concrete_list.h"
#include "cuda_concrete_bitmaplist.h"

using namespace std;

void test_bitmap                ( int lb, int ub );
void test_list                  ( int lb, int ub );
void test_bitmap_list           ( int lb, int ub );
void test_multi_bitmap_list     ();
void test_multi_bitmap_list_aux ();

int main ( int argc, char * argv[] ) {
  std::string dbg = "domain_test - ";
  
  if ( argc < 3 || argc > 4 ) {
    cout << "Usage:\n";
    cout << "./nvidioso <lower> <upper> [test]\n";
    cout << "where lower upper are integers values representing {lower, upper}\n";
    cout << "test is optional:\n";
    cout << "\t1 - test bitmap\n";
    cout << "\t2 - test list\n";
    cout << "\t3 - test bitmap\n";
    cout << "\t4 - test cuda_domain\n";
    cout << "\t0 (default) - all.\n";
    exit ( 0 );
  }
  
  int lb   = atoi ( argv [ 1 ] );
  int ub   = atoi ( argv [ 2 ] );
  int test = 0;
  if ( argc == 4) test = atoi ( argv [ 3 ] );
  
  if (test == 0 || test == 1 )
    cout << dbg << " Test bitmap...\n";
  try {
    if (test == 0 || test == 1 )
      test_bitmap ( lb, ub );
  } catch (...) {
    cout << dbg << " ...test bitmap failed.\n";
  }
  if (test == 0 || test == 1 )
    cout << dbg << " ...test bitmap completed successfully.\n";
  
  if (test == 0 || test == 2 )
    cout << dbg << " Test list...\n";
  try {
    if (test == 0 || test == 2 )
      test_list ( lb, ub  );
  } catch (...) {
    cout << dbg << " ...test list failed.\n";
  }
  if (test == 0 || test == 2 )
    cout << dbg << " ...test list completed successfully.\n";
  
  if (test == 0 || test == 3 )
    cout << dbg << " Test bitmaplist...\n";
  try {
    if (test == 0 || test == 3 ) {
      test_bitmap_list ( lb, ub  );
      //test_multi_bitmap_list ();
      //test_multi_bitmap_list_aux ();
    }
  } catch (...) {
    cout << dbg << " ...test bitmaplist failed.\n";
  }
  if (test == 0 || test == 3 )
    cout << dbg << " ...test bitmaplist completed successfully.\n";
  
  if ( test == 0 || test == 4 ) {
    cout << "Test Cuda Domain...\n";
    DomainPtr cuda_domain = make_shared<CudaDomain> ();
    ( static_pointer_cast<CudaDomain>( cuda_domain ) )->init_domain ( lb, ub );
    cuda_domain->print ();
  }
  
  cout << dbg << "Exit from domain test.\n";
  
  return 0;
}//main

void test_bitmap ( int lb, int ub ) {
  
  // Define a pointer to bitmap domain
  CudaConcreteDomainPtr bitmap_domain;
  
  int min      = lb;
  int max      = ub;
  int offset   = 1;
  int add_val  = 8;
  int dom_size = max - min + 1;
  int n_bytes  = ceil ( 1.0 * dom_size / 8 );
  
  cout << "Test on:\n";
  cout << "Domain:\t{" << min << ", " << max << "}\n";
  cout << "Size:\t" << dom_size << "\n";
  cout << "Bytes:\t" << n_bytes << "\n";
  
  // Instantiate bitmap domain
  bitmap_domain = make_shared<CudaConcreteDomainBitmap>( n_bytes, min, max );
  
  // Print bitmap domain
  cout << "Init: {" << bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "}\n";
  bitmap_domain->print ();
  cout << endl;
  cout << "Size: " << bitmap_domain->size () << endl;
  
  cout << "SHRINK Domain:\n";
  cout << "{" << bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "} -> {" <<
  bitmap_domain->lower_bound() + offset << ", " <<
  bitmap_domain->upper_bound() - offset << "}\n";
  bitmap_domain->shrink ( bitmap_domain->lower_bound() + offset, bitmap_domain->upper_bound() - offset );
  
  // Print bitmap domain
  bitmap_domain->print ();
  cout << endl;
  cout << "Size: " << bitmap_domain->size () << endl;

  cout << "IN_MIN Domain:\n";
  cout << "{" << bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "} -> {" <<
  bitmap_domain->lower_bound() + offset << ", " <<
  bitmap_domain->upper_bound() << "}\n";
  bitmap_domain->in_min ( bitmap_domain->lower_bound() + offset );
  
  // Print bitmap domain
  bitmap_domain->print ();
  cout << endl;
  cout << "Size: " << bitmap_domain->size () << endl;
  
  cout << "IN_MAX Domain:\n";
  cout << "{" << bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "} -> {" <<
  bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() - 2 << "}\n";
  bitmap_domain->in_max ( bitmap_domain->upper_bound() - 2 );
  
  // Print bitmap domain
  bitmap_domain->print ();
  cout << endl;
  cout << "Size: " << bitmap_domain->size () << endl;
  
  cout << "SHRINK TO SINGLETON Domain:\n";
  cout << "{" <<
  bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "} -> {" <<
  bitmap_domain->lower_bound() + 1 << ", " <<
  bitmap_domain->lower_bound() + 1 << "}\n";
  bitmap_domain->shrink ( bitmap_domain->lower_bound() + 1, bitmap_domain->lower_bound() + 1 );
  bitmap_domain->print ();
  cout << endl;
  cout << "{" <<
  bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "}" << endl;
  
  cout << "Size: " << bitmap_domain->size () << endl;
  
  if ( bitmap_domain->is_singleton () ) {
    cout << "Singleton: {" << bitmap_domain->get_singleton () << "}\n";
  }
  else {
    cout << "Not singleton\n";
  }
  
  cout << "ADD VALUE " << add_val << endl;
  bitmap_domain->add ( add_val );
  cout << "{" <<
  bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "}" << endl;
  bitmap_domain->print ();
  cout << endl;
  cout << "Size: " << bitmap_domain->size () << endl;
  
  cout << "ADD VALUE FROM " << 5 <<  " TO " << 7 << endl;
  bitmap_domain->add ( 5, 7 );
  cout << "{" <<
  bitmap_domain->lower_bound() << ", " <<
  bitmap_domain->upper_bound() << "}" << endl;
  bitmap_domain->print ();
  cout << endl;
  cout << "Size: " << bitmap_domain->size () << endl;
  
  if ( bitmap_domain->contains ( 9 ) ) {
    cout << "Domain contains " << 9 << endl;
  }
  else {
    cout << "Domain don't contain " << 9 << endl;
  }
  
  if ( bitmap_domain->contains ( 5 ) ) {
    cout << "Domain contains " << 5 << endl;
  }
  else {
    cout << "Domain does not contain " << 5 << endl;
  }
}//test_bitmap

void test_list ( int lb, int ub ) {
  
  // Define a pointer to bitmap domain
  CudaConcreteDomainPtr list_domain;
  
  int min      = lb;
  int max      = ub;
  int offset   = 1;
  int add_val  = 8;
  int dom_size = max - min + 1;
  int n_bytes  = 47 * 1024;
  
  cout << "Test on:\n";
  cout << "Domain:\t{" << min << ", " << max << "}\n";
  cout << "Size:\t" << dom_size << "\n";
  cout << "Bytes:\t" << n_bytes << "\n";
  
  // Instantiate bitmap domain
  list_domain = make_shared<CudaConcreteDomainList>( n_bytes, min, max );
  
  // Print list domain
  cout << "Init: {" << list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "}\n";
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  
  cout << "SHRINK Domain:\n";
  cout << "{" << list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "} -> {" <<
  list_domain->lower_bound() + offset << ", " <<
  list_domain->upper_bound() - offset << "}\n";
  list_domain->shrink ( list_domain->lower_bound() + offset, list_domain->upper_bound() - offset );
  
  // Print list domain
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  
  cout << "IN_MIN Domain:\n";
  cout << "{" << list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "} -> {" <<
  list_domain->lower_bound() + offset << ", " <<
  list_domain->upper_bound() << "}\n";
  list_domain->in_min ( list_domain->lower_bound() + offset );
  
  // Print list domain
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  
  cout << "IN_MAX Domain:\n";
  cout << "{" << list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "} -> {" <<
  list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() - 2 << "}\n";
  list_domain->in_max ( list_domain->upper_bound() - 2 );
  
  // Print list domain
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  
  
  cout << "SHRINK TO SINGLETON Domain:\n";
  cout << "{" <<
  list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "} -> {" <<
  list_domain->lower_bound() + 1 << ", " <<
  list_domain->lower_bound() + 1 << "}\n";
  list_domain->shrink ( list_domain->lower_bound() + 1, list_domain->lower_bound() + 1 );
  list_domain->print ();
  cout << endl;
  cout << "{" <<
  list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "}" << endl;
  
  cout << "Size: " << list_domain->size () << endl;
  
  if ( list_domain->is_singleton () ) {
    cout << "Singleton: {" << list_domain->get_singleton () << "}\n";
  }
  else {
    cout << "Not singleton\n";
  }
  
  
  cout << "ADD VALUE " << add_val << endl;
  list_domain->print ();
  cout << "-> ";
  list_domain->print ();
  cout << "{" << add_val << ", " << add_val << "}\n";
  list_domain->add ( add_val );
  cout << "{" <<
  list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "}" << endl;
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  

  cout << "ADD VALUE FROM " << 1 <<  " TO " << 4 << endl;
  list_domain->add ( 1, 4 );
  cout << "{" <<
  list_domain->lower_bound() << ", " <<
  list_domain->upper_bound() << "}" << endl;
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  
  cout << "ADD VALUES...\n";
  list_domain->add ( 10, 15 );
  list_domain->add ( 20, 25 );
  list_domain->add ( 30, 37 );
  list_domain->add ( 40, 50 );
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  
  list_domain->add ( 18, 38 );
  list_domain->print ();
  cout << endl;
  cout << "Size: " << list_domain->size () << endl;
  
  
  cout << "CONTAINS: " << 8 << endl;
  if ( list_domain->contains ( 8 ) ) {
    cout << "List contains " << 8 << endl;
  }
  else {
    cout << "List doesn't contain " << 8 << endl;
  }
  cout << "CONTAINS: " << 5 << endl;
  if ( list_domain->contains ( 5 ) ) {
    cout << "List contains " << 5 << endl;
  }
  else {
    cout << "List doesn't contain " << 5 << endl;
  }
}//test_list

void test_bitmap_list ( int lb, int ub ) {
  
  // Define a pointer to bitmap domain
  CudaConcreteDomainPtr bitmapList_domain;
  
  int min      = lb;
  int max      = ub;
  int offset   = 1;
  int add_val  = 8;
  int dom_size = max - min + 1;
  int n_bytes  = 47 * 1024;
  
  cout << "Test on:\n";
  cout << "Domain:\t{" << min << ", " << max << "}\n";
  cout << "Size:\t" << dom_size << "\n";
  cout << "Bytes:\t" << n_bytes << "\n";
  
  // Instantiate bitmap domain
  vector < pair <int, int> > pairs;
  pairs.push_back ( make_pair ( min, max ) );
  bitmapList_domain = make_shared<CudaConcreteBitmapList>( n_bytes, pairs );
  
  // Print bitmap domain
  cout << "Init: {" << bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "}\n";
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size () << endl;

  cout << "SHRINK Domain:\n";
  cout << "{" << bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "} -> {" <<
  bitmapList_domain->lower_bound() + offset << ", " <<
  bitmapList_domain->upper_bound() - offset << "}\n";
  bitmapList_domain->shrink ( bitmapList_domain->lower_bound() + offset, bitmapList_domain->upper_bound() - offset );
  
  // Print bitmaplist domain
  cout << "{" <<
  bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "}" << endl;
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size () << endl;
  
  // Print bitmaplist domain
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size () << endl;
  
  cout << "IN_MIN Domain:\n";
  cout << "{" << bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "} -> {" <<
  bitmapList_domain->lower_bound() + offset << ", " <<
  bitmapList_domain->upper_bound() << "}\n";
  bitmapList_domain->in_min ( bitmapList_domain->lower_bound() + offset );
  
  // Print list domain
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size () << endl;
  
  cout << "IN_MAX Domain:\n";
  cout << "{" << bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "} -> {" <<
  bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() - 2 << "}\n";
  bitmapList_domain->in_max ( bitmapList_domain->upper_bound() - 2 );
  
  // Print list domain
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size () << endl;
  
  
  cout << "SHRINK TO SINGLETON Domain:\n";
  cout << "{" <<
  bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "} -> {" <<
  bitmapList_domain->lower_bound() + 1 << ", " <<
  bitmapList_domain->lower_bound() + 1 << "}\n";
  bitmapList_domain->shrink ( bitmapList_domain->lower_bound() + 1, bitmapList_domain->lower_bound() + 1 );
  bitmapList_domain->print ();
  cout << endl;
  cout << "{" <<
  bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "}" << endl;
  
  cout << "Size: " << bitmapList_domain->size () << endl;
  
  if ( bitmapList_domain->is_singleton () ) {
    cout << "Singleton: {" << bitmapList_domain->get_singleton () << "}\n";
  }
  else {
    cout << "Not singleton\n";
  }
}//test_bitmap_list

void test_multi_bitmap_list () {
  
  // Define a pointer to bitmap domain
  CudaConcreteDomainPtr bitmapList_domain;
  
  int n_bytes  = 47 * 1024;
  
  cout << "Test on:\n";
  cout << "Bytes:\t" << n_bytes << "\n";
  
  // Instantiate bitmap domain
  vector < pair <int, int> > pairs;
  pairs.push_back ( make_pair ( 3, 19 ) );
  pairs.push_back ( make_pair ( 23, 28 ) );
  pairs.push_back ( make_pair ( 30, 34 ) );
  bitmapList_domain = make_shared<CudaConcreteBitmapList>( n_bytes, pairs );
  
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size() << endl;
  
  bitmapList_domain->shrink( 28, 28 );
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size() << endl;
  
}////test_bitmap_list

void test_multi_bitmap_list_aux () {
  
  // Define a pointer to bitmap domain
  CudaConcreteDomainPtr bitmapList_domain;
  
  int n_bytes  = 47 * 1024;
  
  cout << "Test on:\n";
  cout << "Bytes:\t" << n_bytes << "\n";
  
  // Instantiate bitmap domain
  vector < pair <int, int> > pairs;
  pairs.push_back ( make_pair ( -15, -8 ) );
  pairs.push_back ( make_pair (  -5, -2 ) );
  pairs.push_back ( make_pair (   3,  7 ) );
  pairs.push_back ( make_pair (  15, 20 ) );
  bitmapList_domain = make_shared<CudaConcreteBitmapList>( n_bytes, pairs );
  
  bitmapList_domain->print ();
  
  cout << endl;
  cout << "Size: " << bitmapList_domain->size() << endl;
  
  cout << "{" <<
  bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "}" << endl;
  
  
  bitmapList_domain->shrink( -2, -2 );
  bitmapList_domain->print ();
  cout << endl;
  cout << "Size: " << bitmapList_domain->size() << endl;
  
  
  cout << "{" <<
  bitmapList_domain->lower_bound() << ", " <<
  bitmapList_domain->upper_bound() << "}" << endl;
  cout << "Size: " << bitmapList_domain->size() << endl;
}//test_multi_bitmap_list_aux

