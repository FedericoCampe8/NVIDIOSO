//
//  cuda_cp_model.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 30/11/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents a base CP model on CUDA.
//  It holds the state concerning all the information needed to
//  explore the search space in order to find solutions.
//  It holds the following finite sets:
//    a) V of Variables, each associated with a finite domain D
//    b) C of constraints over the variables in V.
//       This set is represented by the correspondent Constraint Store.
//  This class represents general objects.
//  Specific implementations of the solver (i.e., using the CUDA framework)
//  should derive from this class and specilize the following methods.
//

#ifndef NVIDIOSO_cuda_cp_model_h
#define NVIDIOSO_cuda_cp_model_h

#include "cp_model.h"

using uint = unsigned int;
  
class CudaCPModel : public CPModel {
protected:
  
  std::string _dbg;
  
  //! Domain state on host
  uint * _h_domain_states;
  
  //! Domain state on device
  uint *  _d_domain_states;
  
  //! Size of (num. of bytes) all domains
  size_t _domain_state_size;
  
  //! Domain (begin) index
  int  *  _d_domain_index;
  /**
   * Information related to constraints:
   * 1 - Type of constraint
   * 2 - constraint's id
   * 3 - scope size
   * 4 - number of auxiliary arguments
   * 5 - list of variables ids
   * 6 - list of auxiliary arguments
   */
  int* d_constraint_description;
  
  //! Map from var id on host to var id on device
  std::map<int, int> _cuda_var_lookup;
  
  //! Allocate domains on device
  virtual bool alloc_variables ();
  
  //! Allocate constraints on device
  virtual bool alloc_constraints ();
  
public:
  CudaCPModel ();
  ~CudaCPModel();
  
  //! Mapping between constraint ids and the constraints on device
  std::unordered_map< size_t, size_t >constraint_mapping_h_d;
  
  //! Get function for domain states
  uint * const get_dev_domain_states_ptr () const;
  
  //! Get function for domain states indeces
  int * const get_dev_domain_index_ptr () const;
  
  /**
   * Finalizes the model.
   * This method actually allocates the structures on 
   * the device.
   */
  void finalize ();
  
  /**
   * Move the current state (set of domains)
   * from host to device.
   * @return true if the upload has been completed successfully. False otherwise.
   * @note update all variables into device.
   */
  virtual bool upload_device_state ();
  
  /**
   * Move the current state (set of domains)
   * from device to host.
   * @return true if the dowload has been completed successfully 
   *         AND no empty domains are present. False otherwise.
   * @note update all variables into host.
   */
  virtual bool download_device_state ();
  
};

#endif
