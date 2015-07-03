//
//  input_data.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 26/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.

//  InputData class:
//  Singleton class used for reading input options given by the user.


#ifndef NVIDIOSO_input_data_h
#define NVIDIOSO_input_data_h

#include "globals.h"

class InputData {
  
private:
  //! Debug string
  std::string _dbg;
  
  // Other flags and params
  int _verbose;
  int _time;
  int _max_sol;
  double _timeout;
  
  std::string _in_file;
  std::string _out_file;
  std::string _help_file;
  
  // Private methods
  void init();
  void print_help ();
  void print_gpu_info ();
  
protected:
  
  /**
   * Protected constructor: a client cannot instantiate
   * Singleton directly. 
   */
  InputData ( int argc, char* argv[] );
  
public:
	
  virtual ~InputData ();

  InputData ( const InputData& other ) 			  = delete; 
  InputData& operator= ( const InputData& other ) = delete; 
  
  //! Constructor to get the (static) InputData instance
  static InputData& get_instance ( int argc, char* argv[] ) 
  {
  	static InputData id_instance ( argc, argv );
    return id_instance;
  }//get_instance
  
  /**
   * Informs about the verbose option.
   * @return true if verbose is on.
   */
  bool verbose () const;
  
  /**
   * Informs about the time option.
   * @return true if timer is on.
   */
  bool timer () const;
  
  /**
   * Returns the timeout limit set by the user (default: inf).
   * @return the timeout limit.
   */
  double timeout () const;
  
  /**
   * Returns the limit on the number of solution
   * set by the user (default: 1).
   * @return the given limit on the number of solutions.
   */
  int max_n_sol () const;
  
  /**
   * Get input file (path to).
   * @return the path where the input file is located.
   */
  std::string get_in_file  () const;
  
  /**
   * Get output file (path to). If no path is given,
   * output will be printed on standard output.
   * @return the path to the file where the output results should be written.
   */
  std::string get_out_file () const;
};

#endif
