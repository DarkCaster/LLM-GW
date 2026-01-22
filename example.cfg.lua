-- example config file

-- gateway server options
server = {
	listen_v4 = "127.0.0.1:7777", -- ipv4 address to listen on, "none" to not start server listener
	listen_v6 = "none", -- ipv4 address to listen on, "none" to not start server listener
	health_check_timeout = 5.00, -- required param, must be > 0
	engine_startup_timeout = 60.0, -- required param, must be > 0
	engine_idle_timeout = 120.0, -- required param, must be > 0
}

-- NOTE: there are 2 helper functions available to help with tables and arrays (args)
-- r = concat_arrays(a1, a2) - will merge 2 arrays-tables together in a new array "r", appending contents of a2 to a1. will omit all non-indexed elements
-- r = merge_tables(t1, t2) - will merge 2 tables together in a new table "r", elements with same keys from t2 will replace elements from t1, non-indexed elements will be appended

-- model table example, needs to be included to the "models" table at the end
example_model = {
	engine = presets.engines.llamacpp,
	name = "example-model",
	connect = "http://127.0.0.1:8080", -- base url to forward queries incoming to server when using this model
	-- health_check_timeout = 5.00, -- optional param, if missing, it will use server.health_check_timeout, must be > 0
	-- engine_startup_timeout = 60.0, -- optional param, if missing, it will use server.engine_startup_timeout, must be > 0
	-- engine_idle_timeout = 120.0, -- optional param, if missing, it will use server.engine_startup_timeout, must be > 0
	tokenization = { -- manual tokenization
		-- used for initial context size requirements estimation when llama-server is not running
		-- this estimation is less presise than tokenize query with the running llama-server because chat-template is not applied to the messages
		binary = "/path/to/the/llama-tokenize/binary"
		-- base_args = { "--log-disable", "--stdin", "--ids" }, -- optional, only enable to override internal args
		extra_args = { "-m", "/path/to/model.gguf/file" }, -- extra arguments needed for llama-tokenize to work, use to pass model name
		extra_tokens_per_message = 8, -- add extra 10 tokens per each message to compensate chat-template absense
	}
	variants = {
		{
			binary = "/path/to/the/llama-server/binary", -- llama-server binary to launch, mandatory
			-- connect = "http://127.0.0.1:8080", -- optional, if missing will use value from <this model>
			args = {"-np","1","-ngl","999","-cmoe","-c","32000","-ctk","q8_0","-ctv","q8_0","-ub","4096","-b","4096","--mmap","-m", "/path/to/model.gguf/file"},
			context = 32000, -- context size provided by this variant, used to choose what variant to load to process incoming query
			-- engine_startup_timeout = 60.0, -- optional param, if missing, it will use <this model>.engine_startup_timeout
			-- health_check_timeout = 5.00, -- optional param, if missing, it will use <this model>.health_check_timeout, must be > 0
			-- engine_idle_timeout = 120.0, -- optional param, if missing, it will use <this model>.engine_startup_timeout, must be > 0
		},
		{
			binary = "/path/to/the/llama-server/binary", -- llama-server binary to launch, mandatory
			connect = "http://127.0.0.1:8080", -- base url to route incoming query from listen_v4/v6
			args = {"-np","1","-ngl","999","-cmoe","-c","64000","-ctk","q8_0","-ctv","q8_0","-ub","2048","-b","4096","--mmap","-m", "/path/to/model.gguf/file"},
			context = 64000, -- context size provided by this variant, used to choose what variant to load to process incoming query
		},
	},
}

-- You must compose all your models here in this table
models = { example_model, --[[ example_model2, example_mode3, etc... ]] }
