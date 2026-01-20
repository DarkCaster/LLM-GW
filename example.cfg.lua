-- example config file

-- gateway server options
server = {
	listen_v4 = "127.0.0.1:7777", -- ipv4 address to listen on, "none" to not start server listener
	listen_v6 = "none", -- ipv4 address to listen on, "none" to not start server listener
	health_check_timeout = 5.00, -- required param, must be > 0
	engine_startup_timeout = 60.0, -- required param, must be > 0
	engine_idle_timeout = 120.0, -- required param, must be > 0
}

-- model table example, needs to be included to the "models" table at the end
qwen3_30b_moe = {
	engine = presets.engines.llamacpp,
	name = "qwen3-30b-moe",
	health_check_timeout = 5.00, -- optional param, if missing, add parameter to table, set it equal to server.health_check_timeout, must be > 0
	engine_startup_timeout = 60.0, -- optional param, if missing, add parameter to table, set it equal to server.engine_startup_timeout, must be > 0
	engine_idle_timeout = 120.0, -- optional param, if missing, add parameter to table, set it equal to server.engine_startup_timeout, must be > 0
	variants = {
		{
			binary = "/path/to/the/llama-server/binary", -- llama-server binary to launch, mandatory
			connect = "http://127.0.0.1:8080", -- base url to route incoming query from listen_v4/v6
			args = {"-np","1","-ngl","999","-cmoe","-c","32000","-ctk","q8_0","-ctv","q8_0","-ub","4096","-b","4096","--mmap","-m", "/path/to/model.gguf/file"},
			tokenize = true, -- allow use this variant for tokenization if loaded, else it will stop/unload this variant and load variant where tokenize it true
			context = 32000, -- context size provided by this variant, used to choose what variant to load to process incoming query
			engine_startup_timeout = 60.0, -- optional param, if missing, add parameter to table, set it equal to <this model>.engine_startup_timeout
			health_check_timeout = 5.00, -- optional param, if missing, add parameter to table, set it equal to <this model>.health_check_timeout, must be > 0
			engine_idle_timeout = 120.0, -- optional param, if missing, add parameter to table, set it equal to <this model>.engine_startup_timeout, must be > 0
		},
		{
			binary = "/path/to/the/llama-server/binary", -- llama-server binary to launch, mandatory
			connect = "http://127.0.0.1:8080", -- base url to route incoming query from listen_v4/v6
			args = {"-np","1","-ngl","999","-cmoe","-c","64000","-ctk","q8_0","-ctv","q8_0","-ub","2048","-b","4096","--mmap","-m", "/path/to/model.gguf/file"},
			tokenize = true, -- allow use this variant for tokenization if loaded, else it will stop/unload this variant and load variant where tokenize it true
			context = 64000, -- context size provided by this variant, used to choose what variant to load to process incoming query
			engine_startup_timeout = 60.0, -- optional param, if missing, add parameter to table, set it equal to <this model>.engine_startup_timeout
			health_check_timeout = 5.00, -- optional param, if missing, add parameter to table, set it equal to <this model>.health_check_timeout, must be > 0
			engine_idle_timeout = 120.0, -- optional param, if missing, add parameter to table, set it equal to <this model>.engine_startup_timeout, must be > 0
		},
	},
}

models = { qwen3_30b_moe }
