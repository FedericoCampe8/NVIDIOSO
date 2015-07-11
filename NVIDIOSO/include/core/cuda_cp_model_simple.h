//
//  cuda_cp_model_simple.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/09/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  This class represents a concrete CP model on CUDA.
//  Its behaviour is the same as CudaCPModel (see "cuda_cp_model.h")
//  but it changes the way states are uploaded and downloaded on device.
//  Uploads and downloads are performed according to the representation of domains
//  used on CUDA, i.e., Boolean domains on Host are represented with 2 integers on Device
//  while standards domains on Host are represented either with a pair of bounds or
//  with a bitmap. 
//  Therefore this concrete class implements the above mapping.
//  Note that events on domains are set again before states are copied to device 
//  (normally there are no events on domains before constraint propagation).
//  This is done to allow device to refer directly to the event on a given domain
//  without the need of retrieving it from the current elements in the domain and,
//  therefore, reduce memory accesses on device.
//

#ifndef NVIDIOSO_cuda_cp_model_simple_h
#define NVIDIOSO_cuda_cp_model_simple_h

#include "cuda_cp_model.h"

using uint = unsigned int;
  
class CudaCPModelSimple : public CudaCPModel {
protected:
    //! Map for Boolean variables
    std::unordered_set<int> _bool_var_lookup;
    
    //! Allocate domains on device
    bool alloc_variables ();
	
public:
    CudaCPModelSimple ();
    ~CudaCPModelSimple();
  
    /**
     * Move the current state (set of domains) from host to device.
     * @return true if the upload has been completed successfully, False otherwise.
     * @note update all variables into device.
     * @note change domain representation to a domain representation used on Device
     */
    bool upload_device_state () override;
    
    /**
     * Move the current state (set of domains) from device to host.
     * @return true if the dowload has been completed successfully 
     *         AND no empty domains are present. False otherwise.
     * @note update all variables into host.
     * @note change domain representation to a domain representation used on Host
     */
    bool download_device_state () override;
};

#endif
