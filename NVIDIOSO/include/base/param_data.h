//
//  param_data.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/15/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  ParamData class:
//  Class used to read parameters options for the solver.
//


#ifndef NVIDIOSO_param_data_h
#define NVIDIOSO_param_data_h

#include "globals.h"

class ParamData;
extern ParamData* solver_params;

enum class CudaPropParam
{
    SEQUENTIAL,
    BLOCK_PER_CON,
    BLOCK_PER_VAR
};

class ParamData {
  
private:
    
    //! Debug string
    std::string _dbg;

    //! Input file
    std::string _in_file;

    //! Input file stream
    std::ifstream _ifs_params;
    
    // ======================= MACROS ======================
    static constexpr std::string SEPARATOR_KWD;
    static constexpr std::string PARAM_SEP_KWD;
    static constexpr std::string PARAM_YES_KWD;
    static constexpr std::string PARAM_NO_KWD;
    static constexpr std::string SEARCH_KWD;
    static constexpr std::string CSTORE_KWD;
    // =====================================================
    
    // ======================== SEARCH PARAMETERS ========================
    bool   _search_debug;
    bool   _search_trail_debug;
    bool   _search_time_watcher;
    int    _search_solution_limit;
    int    _search_backtrack_limit;
    int    _search_nodes_limit;
    int    _search_wrong_decisions_limit;
    double _search_timeout;
    static constexpr std::string SEARCH_DEBUG_KWD;
    static constexpr std::string SEARCH_TRAIL_DEBUG_KWD;
    static constexpr std::string SEARCH_TIME_WATCHER_KWD;
    static constexpr std::string SEARCH_TIMEOUT_KWD;
    static constexpr std::string SEARCH_SOL_LIM_KWD;
    static constexpr std::string SEARCH_BACKTRK_LIM_KWD;
    static constexpr std::string SEARCH_NODES_LIM_KWD;
    static constexpr std::string SEARCH_WRONG_DEC_KWD;
    // ====================================================================

    // ======================= CONSTRAINT STORE PARAMETERS =======================
    bool _cstore_consistency;
    bool _cstore_satisfiability;
    static constexpr std::string CSTORE_CONSISTENCY_KWD;
    static constexpr std::string CSTORE_SATISFIABILITY_KWD;

    // 					 ======== CONSTRAINT STORE CUDA ========
    
    int  _cstore_cuda_prop_loop_out;
    CudaPropParam _cstore_cuda_propagation_function;
    static constexpr std::string CSTORE_CUDA_PROP_KWD;
    static constexpr std::string CSTORE_CUDA_SEQ_KWD;
    static constexpr std::string CSTORE_CUDA_BPC_KWD;
    static constexpr std::string CSTORE_CUDA_BPV_KWD;
    static constexpr std::string CSTORE_CUDA_PROP_LOOP_OUT_KWD;
    // ===========================================================================

    //Print utilities
    void print_option ( bool         b,  bool new_line=true ) const;
    void print_option ( int          n,  bool new_line=true ) const;
    void print_option ( double       d,  bool new_line=true ) const;
    void print_option ( std::string  s,  bool new_line=true ) const;
    void print_option ( CudaPropParam p, bool new_line=true ) const;
    
protected:
    //! Open parameters file
    void open  ();

    //! Close parameters file
    void close ();

    //! Get parameter value
    std::string get_param_value ( std::string line );
    
    //! Set default parameters
    virtual void set_default_parameters ();

    //! Read parameters from file
    virtual void read_params ();

    //! Search engine parameters
    virtual void set_search_parameters ( std::string& line );

    //! Constraint store parameters
    virtual void set_constraint_engine_parameters ( std::string& line );
    
public:
    ParamData ();
    ParamData ( std::string in_file );
    
    virtual ~ParamData ();

    //! Set input (parameters) path
    void set_param_path ( std::string path );

    //! Read parameters from file
    virtual void set_parameters ();
    
    /**
     * Get path where parameters file is located.
     * @return the path where the parameters file is located.
     */
    std::string get_param_path  () const;

    //======================================
    
    // ========= SEARCH PARAMETERS =========
    bool   search_get_debug () const;
    bool   search_get_trail_debug () const;
    bool   search_get_time_watcher () const;
    int    search_get_solution_limit () const;
    int    search_get_backtrack_limit () const;
    int    search_get_nodes_limit () const;
    int    search_get_wrong_decision_limit () const;
    double search_get_timeout () const;
    // =====================================

    // ========= CSTORE PARAMETERS =========
    bool cstore_get_consistency () const;
    bool cstore_get_satisfiability () const;
    int	 cstore_get_dev_loop_out () const;
    int  cstore_type_to_int ( CudaPropParam ctype ) const;
    CudaPropParam cstore_int_to_type ( int ctype ) const;
    CudaPropParam cstore_get_dev_propagation () const;
    // =====================================
    
    //======================================
    
    virtual void print () const;
};

#endif
