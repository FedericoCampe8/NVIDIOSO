/*
 *  input_data.cc
 *  iNVIDIOSO
 *
 *  Created by Federico Campeotto on 26/06/14.
 *  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
 */

#include "input_data.h"

#if WINDOWS_REL
	#include "getopt_win.h"
#else
	#include <getopt.h>
#endif 

using std::cout;
using std::endl;
using std::string;

void
InputData::init () {
  _dbg = "InputData - ";
  _verbose   = 0;
  _time      = 0;
  _max_sol   = 1;
  _timeout   = -1.0;
  _in_file   = "";
  _out_file  = "";
  _help_file = "config/.idata_help.txt";
}//init

InputData::InputData ( int argc, char* argv[] ) {
  
  // Init parameters
  init();
  
  // Read input
  while ( true ) {
    static struct option long_options[] =
    {
      // These options set a flag.
      {"verbose", no_argument, &_verbose,      1},
      {"watcher", no_argument, &_time,         1},
      /* These options don't set a flag.
        We distinguish them by their indices. */
      {"help",        no_argument,        0, 'h'}, // Print help message
      {"device",      no_argument,        0, 'd'}, // Print device info
      {"solutions",   required_argument,  0, 'n'}, // Set number of solutions
      {"timeout",     required_argument,  0, 't'}, // Set timeout in seconds
      {"input",       required_argument,  0, 'i'}, // Set input file
      {"output",      required_argument,  0, 'o'}, // Set output file
      {0, 0, 0, 0}
    };
    // getopt_long stores the option index here.
    int option_index = 0;
    
    int c = getopt_long (argc, argv, "hvdwn:t:i:o:",
                         long_options, &option_index);
    
    // Detect the end of the options.
    if ( c == -1 ) break;
    // Switch on the options read from input
    switch ( c ) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if ( long_options[ option_index ].flag != 0 )
          break;
        printf ( "option %s", long_options[ option_index ].name );
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;
        
      case 'h':
        print_help ();
        exit( 0 );
        
      case 'd':
        print_gpu_info ();
	      exit ( 0 );
       
      case 'w':
        _time = 1;
        break;
        
      case 'v':
        _verbose = 1;
        break;
        
      case 't':
        _timeout = atof ( optarg );
        break;
        
      case 'n':
        _max_sol = atoi( optarg );
        break;
        
      case 'i':
        _in_file = optarg;
        break;
        
      case 'o':
        _out_file = optarg;
        break;
        
      default:
        print_help ();
        break;
    }
  }//while
  
  // Exit without input file
  if ( _in_file.compare( "" ) == 0 ) 
  {
    LogMsgW << _dbg << " Missing input file" << std::endl;
  }
  else 
  {
    //Sanity check
    std::ifstream infile ( _in_file.c_str(), std::ifstream::in );
    if ( !infile.is_open() ) 
    {
      LogMsgW << _dbg << "Can't find input file " << _in_file << std::endl;
    }
    infile.close ();
  }
  
  if ( _verbose ) 
  {
    //puts ( "verbose flag is set" );
    LogMsg.set_verbose ( true );
  }
  
  // Sanity check
    if ( _max_sol < -1 )
    {
      LogMsg << "Maximum number of solutions set to: " << _max_sol << " automatically set to -1";
      _max_sol = -1;
    }
    
  // Print any remaining command line arguments (not options).
  if ( optind < argc ) {
    printf ("non-option ARGV-elements: ");
    while ( optind < argc )
      printf ("%s ", argv[ optind++ ]);
    putchar ('\n');
  }
}//-

InputData::~InputData () {
}

void
InputData::set_input_file ( std::string in_file )
{
  // Sanity check 
  if ( _in_file != "" )
  {
    LogMsgW << "Replacing current input file " << _in_file << " with " << in_file << std::endl;
  }
  _in_file = in_file;
  
  //Sanity check
  std::ifstream infile ( _in_file.c_str(), std::ifstream::in );
  if ( !infile.is_open() ) 
  {
    LogMsgW << "Can't find input file " << _in_file << std::endl;
  }
  infile.close ();
}//set_input_file

void
InputData::set_output_file ( std::string out_file )
{
  // Sanity check 
  if ( _out_file != "" )
  {
    LogMsgW << "Replacing current output file " << _out_file << " with " << out_file << std::endl;
  }
  _out_file = out_file;
}//set_output_file

bool
InputData::verbose () const 
{
  return _verbose == 1;
}//verbose

bool
InputData::timer () const 
{
  return _time == 1;
}//timer

int
InputData::max_n_sol () const 
{
    return _max_sol;
}//max_sol

double
InputData::timeout () const {
  return _timeout;
}//timeout

std::string
InputData::get_in_file () const {
  return _in_file;
}//get_in_file

std::string
InputData::get_out_file () const {
  return _out_file;
}//get_out_file

void
InputData::print_gpu_info () 
{
#if CUDAON
  int nDevices;
  cudaGetDeviceCount ( &nDevices );
  
  for ( int i = 0; i < nDevices; i++ ) 
  {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, i );
    cout << " Device characteristics:\n";
    cout << " Device Number:                 " << i+1 << endl;
    cout << " Device name:                   " << devProp.name << endl;
    cout << " Device's compute capability:   " << devProp.major << "." << 
    devProp.minor << endl;
    cout << " Total global memory:           " << devProp.totalGlobalMem << endl;
    cout << " Total shared memory per block: " << devProp.sharedMemPerBlock << endl;
    cout << " Total registers per block:     " << devProp.regsPerBlock << endl;
    cout << " Total constant memory:         " << devProp.totalConstMem << endl;
    cout << " Warp size:                     " << devProp.warpSize << endl;
    cout << " Maximum threads per block:     " << devProp.maxThreadsPerBlock << endl;
    cout << " Number of multiprocessors:     " << devProp.multiProcessorCount << endl;
    cout << " Memory Clock Rate (KHz):       " << devProp.memoryClockRate << endl;
    cout << " Memory Bus Width (bits):       " << devProp.memoryBusWidth << endl;
    cout << " Peak Memory Bandwidth (GB/s):  " <<
    2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6 << endl;
  }
#else
  cout << "Re-build iNVIDIOSO1.0 choosing the GPU version or" << endl;
  cout << "use \"make CFLAGS=-DCUDAON=true\" to enable CUDA capabilities." << endl;
#endif
}//print_help

void
InputData::print_help () 
{
	std::ifstream ifs ( _help_file );
	if ( !ifs.is_open() )
	{
		LogMsgW << "No help available\n";
		return;
	}
	string line;
	getline ( ifs, line );
	while ( ifs.good() )
	{	
		std::cout << line << endl;
		getline ( ifs, line );
	}
	ifs.close ();
}//print_gpu_info


