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
		binary = "/path/to/the/llama-tokenize/binary",
		-- base_args = { "--log-disable", "--stdin", "--ids" }, -- optional, only enable to override internal args
		extra_args = { "-m", "/path/to/model.gguf/file" }, -- extra arguments needed for llama-tokenize to work, use to pass model name
		extra_tokens_per_message = 8, -- add extra 10 tokens per each message to compensate chat-template absense
	},
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

-- ###################################################################################################################
-- real world example below, you need to download corresponding model-files from HF and llama.cpp builds (+ cuda libs)
-- ###################################################################################################################

-- host/port for llama engine to listen to, used to forward request to
llama_host="127.0.0.1"
llama_port=7778
llama_url="http://"..llama_host..":"..llama_port

-- generic args for llama-server
llama_default_args = {
	"--host", tostring(llama_host),
	"--port", tostring(llama_port),
	"--no-warmup",
	"--no-webui", -- web ui is useless in a case when llama can be started/stopped/restarted any time
	"--verbosity", "2", -- less verbose console output than default 3
	"-cram", "0", "-np", "1", -- caching not needed normally, when we only use llama engine for short time
	"--mmap",
}

-- full path to llama binaries
llama_b7735_bin = [[C:\llama-b7735-bin-win-cuda-13.1-x64\llama-server.exe]]
llama_b7735_tokenize_bin = [[C:\llama-b7735-bin-win-cuda-13.1-x64\llama-tokenize.exe]]

-- helper function to construct params for MoE models, probably not very optimal in terms of performance:
-- only load the most essential MoE model parts into VRAM with "-cmoe" flag to lower VRAM pressure and leave some space for desktop use
function get_llama_moe_args(gguf, ctx_sz, ub, b, ctk, ctv)
	assert(type(ctx_sz)=="number", "ctx_sz param must be a number")
	assert(type(ub)=="number" or type(ub)=="nil", "ub param must be a number or nil")
	assert(type(b)=="number" or type(b)=="nil", "b param must be a number or nil")
	assert(type(ctk)=="string" or type(ctk)=="nil", "ctk param must be a string or nil")
	assert(type(ctv)=="string" or type(ctv)=="nil", "ctv param must be a string or nil")

	local args = concat_arrays(llama_default_args, {"-ngl","999","-cmoe","-c",tostring(ctx_sz)})
	if type(ub)=="number" then args = concat_arrays(args,{"-ub",tostring(ub)}) end
	if type(b)=="number" then args = concat_arrays(args,{"-b",tostring(b)}) end
	if type(ctk)=="string" then args = concat_arrays(args,{"-ctk",ctk}) end
	if type(ctv)=="string" then args = concat_arrays(args,{"-ctv",ctv}) end
	return concat_arrays(args, {"-m", gguf})
end

qwen3_30b_instruct_gguf = [[C:\Qwen\Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf]]
qwen3_30b_coder_gguf = [[C:\Qwen\Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf]]

-- Qwen3-MoE model examples suitable for HW configs with 32G RAM + 8G VRAM
qwen3_30b_instruct_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-30b-instruct",
	connect = llama_url,
	tokenization = { binary = llama_b7735_tokenize_bin, extra_args = { "-m", qwen3_30b_instruct_gguf }, extra_tokens_per_message = 8 },
	variants = {
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_instruct_gguf,10000,4096,4096), context = 10000 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_instruct_gguf,20000,4096,4096), context = 20000 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_instruct_gguf,30000,2048,2048), context = 30000 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_instruct_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_instruct_gguf,81920,1024,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_instruct_gguf,122880,512,2048,"q8_0","q8_0"), context = 122880 },
	},
}

qwen3_30b_coder_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-30b-coder",
	connect = llama_url,
	tokenization = { binary = llama_b7735_tokenize_bin, extra_args = { "-m", qwen3_30b_coder_gguf }, extra_tokens_per_message = 8 },
	variants = {
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_coder_gguf,10000,4096,4096), context = 10000 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_coder_gguf,20000,4096,4096), context = 20000 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_coder_gguf,30000,2048,2048), context = 30000 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_coder_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_coder_gguf,81920,1024,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_b7735_bin, args = get_llama_moe_args(qwen3_30b_coder_gguf,122880,512,2048,"q8_0","q8_0"), context = 122880 },
	},
}

models = { qwen3_30b_instruct_model, qwen3_30b_coder_model }
