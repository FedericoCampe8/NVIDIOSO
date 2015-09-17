/*
 *  solver_configuration_manager.cpp
 *  iNVIDIOSO
 *
 *  Created by Federico Campeotto on 07/15/15.
 *  Modified by Federico Campeotto on 09/15/15.
 *  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
 */

#include "solver_configuration_manager.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using std::cout;
using std::endl;

SolverConfigManager& solver_configurator = SolverConfigManager::get_default_instance ();

// ======================= MACROS ======================
const std::string SolverConfigManager::PARAM_SEP_KWD = "_";
const std::string SolverConfigManager::PARAM_EQ_KWD = "=";
const std::string SolverConfigManager::PARAM_YES_KWD = "YES";
const std::string SolverConfigManager::PARAM_NO_KWD  = "NO";
const std::string SolverConfigManager::PARAMS_DEFAULT_PATH = "config/iNVIDIOSO.params";
const std::string SolverConfigManager::CONFIG_SCHEME_DEFAULT_PATH = "config/config_scheme.json";
// =====================================================

Configurable::Configurable ( std::string type, std::string default_val, std::vector<std::string>& options ) :
  _type             ( type ),
  _default_value    ( "" ),
  _configured_value ( "" ) {
  _configurable_options = options;
  if ( is_config_value ( default_val ) )
  {
    _default_value = default_val;
  }
}

Configurable::~Configurable () {
}

bool 
Configurable::is_config_value ( std::string config_value )
{
  if ( _configurable_options.size() == 0 ) return true;
  
  auto it = std::find ( _configurable_options.begin(), _configurable_options.end (), config_value );
  if ( it != _configurable_options.end () )
  {
    return true;
  }
  return false;
}//is_config_value 

void 
Configurable::set_configuration ( std::string config_value )
{
  // Check is the given value is one of the configurable options
  if ( is_config_value ( config_value ) )
  {
    _configured_value = config_value;
  }
}//set_configuration

std::string 
Configurable::get_configuration_type () const
{
  return _type;
}//get_configuration_type

std::string
Configurable::get_configuration () const 
{
  if ( _configured_value == "" )
  {
    return _default_value;
  }
  return _configured_value;
}//get_configuration

void 
Configurable::print () const
{
  std::cout << "Type:\t" << _type << std::endl;
  std::cout << "Default value:\t" << _default_value << std::endl;
  std::cout << "Configured value:\t" << _configured_value << std::endl;
}//print

SolverConfigManager::SolverConfigManager() :
	_dbg              ("SolverConfigManager - "),
	_json_scheme_file ( CONFIG_SCHEME_DEFAULT_PATH) ,
	_params_data_file ( PARAMS_DEFAULT_PATH ) {

	// Load scheme
	load_scheme ();
  
  // Load parameters
  load_configurations ();
}

/*
 * Default json scheme path.
 * @note change it using the load_scheme method.
 */
SolverConfigManager::SolverConfigManager ( std::string param_data_path ) :
	_dbg			        ( "SolverConfigManager - " ),
	_json_scheme_file ( CONFIG_SCHEME_DEFAULT_PATH ) {

	// Set input
	_params_data_file = param_data_path;

	// Load scheme
	load_scheme();
  
  // Load parameters
  load_configurations ();
}

SolverConfigManager::SolverConfigManager(std::string json_scheme_path, std::string param_data_path) :
	_dbg ("SolverConfigManager - ") {

	// Set scheme 
	_json_scheme_file = json_scheme_path; 

	// Set input
	_params_data_file = param_data_path;

	// Load scheme
	load_scheme();
  
  // Load parameters
  load_configurations ();
}

SolverConfigManager::~SolverConfigManager() {
}

void 
SolverConfigManager::load_scheme ( std::string json_scheme_path )
{
	std::string json_config_file{};
	if (json_scheme_path != "")
	{
		json_config_file = json_scheme_path;
	}
	else
	{// If empty input string, use internal path
		json_config_file = _json_scheme_file;
	}

	std::string line{};
	std::string json_str{};

	std::fstream json_file ( json_config_file );

	// Sanity check
	if (!json_file.is_open())
	{
		throw std::ios_base::failure ("SolverConfigManager::load_scheme - Cannot open JSON configuration file") ;
	}
	while (std::getline(json_file, line))
	{
		json_str += line;
	}

	const char* json = json_str.c_str();

	// Create a JSON document
	rapidjson::Document doc;
	doc.Parse(json);

	// Read the document and get corresponding values
	const rapidjson::Value& config = doc["config"];
	assert(config.IsArray());
	for (rapidjson::SizeType i = 0; i < config.Size(); i++)
	{
		// Class name: SEARCH, CSTORE, ...
		std::string class_id = (config[i]["class"]).GetString();

		// Get class parameters
		const rapidjson::Value& parameters = config[i]["parameters"];
		assert(parameters.IsArray());
		for (rapidjson::SizeType j = 0; j < parameters.Size(); j++)
		{
			// Class keyword option: TREE_DEBUG, SATISFIABILITY, ...
			std::string keyword_id = class_id + PARAM_SEP_KWD + (parameters[j]["keyword"]).GetString();
			
			// Class keyword option type: "string", "bool", "int", "u_int", ...
			std::string keyword_type = (parameters[j]["type"]).GetString();

			// Class keyword option default value: "abc", "true", "-1", "0", "1", ...
			std::string keyword_default = (parameters[j]["default"]).GetString();

			// Get options (if any)
			const rapidjson::Value& options = parameters[j]["options"];
			assert(options.IsArray());

			std::vector < std::string > options_values;
			for (rapidjson::SizeType z = 0; z < options.Size(); z++)
			{
				options_values.push_back((options[z]["value"]).GetString());
			}//z

			/*
       * Insert new value in lookup table.
       * Add only if not present, this will prevent overriding 
       * values previously added.
       */
      auto it = _configurable_lookup.find ( keyword_id );
      if ( it != _configurable_lookup.end () )
      {
        _configurable_lookup[keyword_id] = std::move ( ConfigurableUPtr ( new Configurable(keyword_type, keyword_default, options_values) ) );
      }
		}//j
	}//i
}//load_scheme

void 
SolverConfigManager::set_JSON_scheme_path ( std::string json_scheme_path )
{
  if ( json_scheme_path != "" )
    _json_scheme_file = json_scheme_path;
}//set_JSON_scheme_path

void 
SolverConfigManager::set_config_file_path ( std::string config_path )
{
  if ( config_path != "" )
    _params_data_file = config_path;
}//set_parameters_path

void 
SolverConfigManager::load_configurations ( std::string config_path )
{
  std::string file_to_open = _params_data_file;
  if ( config_path != "" )
    file_to_open = config_path;
  
  // Input file stream
	std::ifstream ifs ( file_to_open );
  if ( !ifs.is_open () )
  {
    LogMsgE << _dbg + "Cannot open configuration file " << file_to_open << std::endl;
    LogMsgE << _dbg + "Default parameters will be used" << std::endl;
    return;
  } 
  
  std::string line{};
  std::size_t pos;
  while ( std::getline ( ifs, line ) )
  {
    pos = line.find_first_of ( "#" );
    if ( pos != std::string::npos )
    {
      line = line.substr ( 0, pos + 1 );
    }
		
    //Skip empty lines
    if ( line.size() == 0 || pos == 0 ) continue;
		
    // Skip lines with spaces
    bool useful_char = false;
    for ( auto& c : line )
    {
      if ( c != ' ' || c != '\t' )
            {
                useful_char = true;
                break;
            }
        }
        if ( !useful_char ) continue;
        
        // Split the line in keyword = value
        std::string keyword{};
        std::string par_val{};
        std::size_t found = line.find( PARAM_EQ_KWD );
        
        // Sanity check
        if (found == std::string::npos) continue;
        keyword = line.substr ( 0, found );
        par_val = line.substr ( found + 1 );
        
        // Trim strings 
        //keyword = boost::algorithm::trim ( keyword );
        //par_val = boost::algorithm::trim ( par_val );
        
        // Translate YES/NO options in true/false 
        par_val = (par_val == "YES") ? "true" : "false";
        
        auto it = _configurable_lookup.find ( keyword );
        if ( it == _configurable_lookup.end () ) continue;
        _configurable_lookup [ keyword ]->set_configuration ( par_val );
  }
  ifs.close();
}//load_configurations

/*
ProxyReturnValue 
SolverConfigManager::get_configuration ( std::string keyword )
{
  return ProxyReturnValue();
}//get_configuration
*/

int 
SolverConfigManager::get_configuration_int ( std::string keyword )
{
  auto it = _configurable_lookup.find ( keyword );
  if ( it == _configurable_lookup.end () )
  {
    LogMsgE << _dbg << " get_configuration_int keyword " << keyword << 
    " not found: return default type value" << std::endl;
    int default_val {};
    return default_val;
  }
  
  if ( it->second->get_configuration_type () != "int" )
  {
    LogMsgE << _dbg << " get_configuration_int type not matching: return default type value" << std::endl;
    int default_val {};
    return default_val;
  }
  
  return std::atoi ( (it->second->get_configuration ()).c_str () );
}//get_configuration_int

bool 
SolverConfigManager::get_configuration_bool ( std::string keyword )
{
  auto it = _configurable_lookup.find ( keyword );
  if ( it == _configurable_lookup.end () )
  {
    LogMsgE << _dbg << " get_configuration_int keyword " << keyword << 
    " not found: return default type value" << std::endl;
    bool default_val {};
    return default_val;
  }
  
  if ( it->second->get_configuration_type () != "bool" )
  {
    LogMsgE << _dbg << " get_configuration_bool type not matching: return default type value" << std::endl;
    bool default_val {};
    return default_val;
  }
  
  if ( it->second->get_configuration () == "true" )
  {
    return true;
  }
  else
  {
    return false;
  }
}//get_configuration_bool

double 
SolverConfigManager::get_configuration_double ( std::string keyword )
{
  auto it = _configurable_lookup.find ( keyword );
  if ( it == _configurable_lookup.end () )
  {
    LogMsgE << _dbg << " get_configuration_int keyword " << keyword << 
    " not found: return default type value" << std::endl;
    double default_val {};
    return default_val;
  }
  
  if ( it->second->get_configuration_type () != "double" )
  {
    LogMsgE << _dbg << " get_configuration_double type not matching: return default type value" << std::endl;
    double default_val {};
    return default_val;
  }
  
  return std::atof ( (it->second->get_configuration ()).c_str () );
}//get_configuration_double

std::size_t 
SolverConfigManager::get_configuration_size_t ( std::string keyword )
{
  auto it = _configurable_lookup.find ( keyword );
  if ( it == _configurable_lookup.end () )
  {
    LogMsgE << _dbg << " get_configuration_int keyword " << keyword << 
    " not found: return default type value" << std::endl;
    return 0;
  }
  
  if ( it->second->get_configuration_type () != "u_int" )
  {
    LogMsgE << _dbg << " get_configuration_size_t type not matching: return default type value" << std::endl;
    return 0;
  }
  
  return static_cast<std::size_t> ( std::abs ( std::atoi ( (it->second->get_configuration ()).c_str () ) ) );
}//get_configuration_size_t

std::size_t 
SolverConfigManager::get_configuration_u_int ( std::string keyword )
{
  return get_configuration_size_t ( keyword );
}//get_configuration_u_int

std::string 
SolverConfigManager::get_configuration_string ( std::string keyword )
{
  auto it = _configurable_lookup.find ( keyword );
  if ( it == _configurable_lookup.end () )
  {
    LogMsgE << _dbg << " get_configuration_int keyword " << keyword << 
    " not found: return default type value" << std::endl;
    return "";
  }
  
  if ( it->second->get_configuration_type () != "string" )
  {
    LogMsgE << _dbg << " get_configuration_string type not matching: return default type value" << std::endl;
    return "";
  }
  
  return it->second->get_configuration ();
}//get_configuration_string

void
SolverConfigManager::print () const
{
    cout << "=========== SolverConfigManager ===========\n";
    for ( auto& pr : _configurable_lookup )
    {
      std::cout << pr.first << std::endl;
      std::cout << "\t" << pr.second->get_configuration () << std::endl;
    }
    cout << "===========================================\n";
}//print
