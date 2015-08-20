//
//  cuda_cp_model.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 30/11/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
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
  
  //! Domain states on host
  uint * _h_domain_states;
  
  /**
   * Domain states on device.
   * @note this array stores the current status of domains
   *       on global memory on device.
   */
  uint *  _d_domain_states;
  
  /**
   * Auxiliary array of domain states.
   * @note by default this is initialized to NULL.
   * @note this is used if some synchronization strategy on
   *       the original array is required and, for example,
   *       two arrays swapping their content are needed.
   */
   uint * _d_domain_states_aux;
  
  //! Size of (num. of bytes) all domains
  size_t _domain_state_size;
  
  /**
   * Domain (begin) index on device.
   * This array has an index for each variable allocated on device.
   * If _d_domain_index [ i ] = k, then D^{V_i} starts at 
   * _d_domain_states [ k ].
   */
  int * _d_domain_index;
  
  /**
   * Information related to base constraints:
   * 1 - Type of constraint
   * 2 - constraint's id
   * 3 - scope size
   * 4 - number of auxiliary arguments
   * 5 - list of variables ids
   * 6 - list of auxiliary arguments
   */
  int* _d_base_constraint_description;
  
  /**
   * Information related to global constraints:
   * 1 - Type of constraint
   * 2 - constraint's id
   * 3 - scope size
   * 4 - number of auxiliary arguments
   * 5 - list of variables ids
   * 6 - list of auxiliary arguments
   */
  int* _d_global_constraint_description;
  
  /**
   * Additional information required by some constraints (e.g., table constraint).
   * @note Information is stored in one single array of integers.
   * @note All tables and arrays are serialized into this array. 
   *       Each array/table starts with number of rows and number of columns as
   *       stored in CPModel class from CudaGenerator.
   */
  int * _d_additional_constraint_parameters;
  int * _d_additional_global_constraint_parameters;
  
  /**
   * Indexes into _d_additional_constraint_info for each constraint
   * to allocate on device.
   * @note if -1, then the constraint does not have any additional parameter.
   */
  int * _d_additional_constraint_parameters_index;
  int * _d_additional_global_constraint_parameters_index;
  
  /**
   * Lookup table from var id on host to var id on device.
   * For each variable id on host there is a corresponding variable id on device.
   * This is done since var id on host may by any integer value while variable ids on device
   * are sequential and correspond to the indexes on _map_vars_to_doms.
   */
  std::map<int, int> _cuda_var_lookup;
  
  /**
   * Map from var on device to index of corresponding domains on device.
   * This is used to copy indexes to _d_domain_index.
   * Moreover, it is also used by the function dev_var_mapping function
   * returning the indexes of the variables on device corresponding to a given 
   * set of (host) variables ids.
   */
  std::vector<int> _map_vars_to_doms;
  
  //! Allocate domains on device
  virtual bool alloc_variables ();
  
  //! Allocate constraints on device
  virtual bool alloc_constraints ();
  
  /*
   * @note the following two function behave the same.
   *       Both allocate and initialize constraint info on device.
   *       There are two functions to allow the user to derive them
   *       implementing new features only on one type of constraints 
   *       (e.g., using different cuda streams on global constraints,
   *        preserving the normal stream on base constraints).
   */
  // Allocate base constraints on device.
  virtual bool alloc_base_constraints ();
  
  // Allocate global constraints on device.
  virtual bool alloc_global_constraints ();
  
public:

  CudaCPModel ();
  ~CudaCPModel();
  
  //! Mapping between constraint ids and the constraints on device
  std::unordered_map< size_t, size_t > constraint_mapping_h_d;
  
  //! Get function for domain states
  uint * const get_dev_domain_states_ptr () const;
  
  //! Get function for domain states indeces
  int * const get_dev_domain_index_ptr () const;
  
  //! Get function for auxiliary states
  uint * const get_dev_domain_states_aux_ptr () const;

  /**
   * Converts the ids of the variables on the host
   * on the corresponding ids of the variables on device.
   * @param var_ids a (unordered) set of host variables ids
   * @return a vector of integers representing the indeces of the
   *         variables on devices corresponding to the var ids in var_ids.
   */
  std::vector<int> dev_var_mapping ( std::unordered_set<int> var_ids );
    
  /**
   * Finalizes the model.
   * This method actually allocates the structures on 
   * the device.
   */
  void finalize ();
  
  /**
   * Allocate memory for the auxiliary array of states.
   * @note if memory has been already allocated, it returns.
   */
   virtual void allocate_domain_states_aux ();
   
   //! Copy current states on the auxiliary array
   virtual void device_state_to_aux ();
   
   //! Copy aux states on the array of states
   virtual void device_aux_to_state ();
   
  /**
   * Reset the current state of the events associated with 
   * the variables.
   * This is done to notify constraint store about events that are 
   * actually changed by propagation.
   * @note after notifying the store, events are automatically reset.
   */
  virtual void reset_device_state ();
  	 
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
