//
//  solver_configuration_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/15/15.
//  Modified by Federico Campeotto on 09/15/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  SolverConfigManager class:
//  Class used to read parameters options for the solver.
//  First the class reads a JSON configuration file, then it parser
//  the file with all the parameters tunable by the user. 
//


#ifndef NVIDIOSO_solver_config_manager_h
#define NVIDIOSO_solver_config_manager_h

#include "globals.h"

class SolverConfigManager;
extern SolverConfigManager& solver_configurator;

class Configurable;
typedef std::unique_ptr<Configurable> ConfigurableUPtr;

/**
 * Proxy to choose the return value type
 * based on the signature of the input values of a function.
 */
 /*
struct ProxyReturnValue {
	operator bool()   const { return true; }
	operator double() const { return 1.1; }
  operator int()    const { return 2; }
};
*/

class Configurable {
friend class SolverConfigManager;
private:
	std::string _type;
	std::string _default_value;
	std::string _configured_value;
	std::vector<std::string> _configurable_options;
public:
	Configurable ( std::string type, std::string default_val, std::vector<std::string>& options );
	
	~Configurable ();
	
  //! Check if a value belongs to _configurable_options 
  bool is_config_value ( std::string config_value );
  
  //! Set chosen value 
	void set_configuration ( std::string config_value ); 
  
	//! Get type
	std::string get_configuration_type () const;
	
  //! Get chosen value
	std::string get_configuration () const;
  
  void print () const;
};

class SolverConfigManager {
 private:
 // ======================= MACROS ======================
	static const std::string PARAM_SEP_KWD;
	static const std::string PARAM_EQ_KWD;
	static const std::string PARAM_YES_KWD;
	static const std::string PARAM_NO_KWD;
	static const std::string PARAMS_DEFAULT_PATH;
	static const std::string CONFIG_SCHEME_DEFAULT_PATH;
	// =====================================================
  
	//! Debug string
	std::string _dbg;

	//! JSON scheme file
	std::string _json_scheme_file;

	//! Parameter data file
	std::string _params_data_file;

	//! Lookup table for configurables object
	std::unordered_map < std::string, ConfigurableUPtr > _configurable_lookup;

public:
  //! Default constructor uses default config files
	SolverConfigManager();

	/**
	 * Constructor.
	 * @param param_data_path path to the parameters file.
	 */
	SolverConfigManager(std::string param_data_path);

	/**
	 * Constructor.
	 * @param json_scheme_path path to the JSON scheme configuration file.
	 * @param param_data_path path to the parameters file.
	 */
	SolverConfigManager(std::string json_scheme_path, std::string param_data_path);

	//! Destructor
	virtual ~SolverConfigManager();
	
	/**
	 * Constructor get (static) instance.
	 * @note NOT a singleton.
	 */  
  static SolverConfigManager& get_default_instance () 
  {
    static SolverConfigManager instance;
    return instance;
  }//get_default_instance
	
	/**
	 * Loads the JSON configuration scheme to interpret
	 * the parameters data file.
	 * @param json_scheme_path path to the JSON scheme, if empty uses 
	 *        the internal path.
	 */
	virtual void load_scheme ( std::string json_scheme_path = "" );
  
  //! Set JSON scheme file path 
  void set_JSON_scheme_path ( std::string json_scheme_path );
  
  //! Set configuration file path 
  void set_config_file_path ( std::string config_path );
  
  //! Read parameters from file (either input path or internal path if given input is empty)
  virtual void load_configurations ( std::string config_path = "" );
	
	/**
	 * Get actual configuration value.
	 * @note return default value type if keyword is not found.
	 * @todo investigate use of overloading casting operator with proxy struct:
	 *       virtual ProxyReturnValue get_configuration ( std::string keyword );
	 */ 
	
	int get_configuration_int ( std::string keyword );
	bool get_configuration_bool ( std::string keyword ); 
	double get_configuration_double ( std::string keyword );
	std::size_t get_configuration_u_int ( std::string keyword );
	std::size_t get_configuration_size_t ( std::string keyword );
	std::string get_configuration_string ( std::string keyword );
	
  virtual void print () const;
};

#endif
