-- example config file

-- gateway server options
server = {
	-- listen_v4 can be a single string or a table (array) of strings
	-- Examples:
	--   Single address: listen_v4 = "127.0.0.1:7777"
	--   All IPv4 addresses: listen_v4 = "0.0.0.0:7777"
	--   Two IPv4 addresses: listen_v4 = {"127.0.0.1:7777", "192.168.1.100:7777"}
	--   Disable IPv4: listen_v4 = "none"
	listen_v4 = "127.0.0.1:7777", -- ipv4 address to listen on, "none" to not start server listener

	-- listen_v6 can be a single string or a table (array) of strings
	-- Examples:
	--   Single address: listen_v6 = "[::1]:7777"
	--   All IPv6 addresses: listen_v6 = "[::]:7777"
	--   Two IPv6 addresses: listen_v6 = {"[::1]:7777", "[fe80::1]:7777"}
	--   Disable IPv6: listen_v6 = "none"
	listen_v6 = "none", -- ipv6 address to listen on, "none" to not start server listener

	-- Default timeouts, you may override it per model or per model-variant
	health_check_timeout = 5.00, -- required param, must be > 0
	engine_startup_timeout = 60.0, -- required param, must be > 0
	engine_idle_timeout = 120.0, -- required param, must be > 0

	-- Debug, optional, uncomment to enable
	-- dumps_dir = "dumps" -- dump incoming requests and answers and place it to logfiles inside this directory
	-- clear_dumps_on_start = false -- on startup, remove old dump-files from dumps_dir
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
	tokenization = { -- manual tokenization, this table is OPTIONAL
		-- used for initial context size requirements estimation when llama-server is not running
		-- this estimation is less precise than tokenize query with the running llama-server because chat-template is not applied to the messages
		binary = "/path/to/the/llama-tokenize/binary",
		-- base_args = { "--log-disable", "--stdin", "--ids" }, -- optional, only enable to override internal args
		extra_args = { "-m", "/path/to/model.gguf/file" }, -- extra arguments needed for llama-tokenize to work, use to pass model name
		extra_tokens_per_message = 8, -- add extra tokens per each message to compensate chat-template overhead
		extra_tokens = 0, -- add this number to the token count result, to compensate embedded system prompt if present
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
--	"--no-mmap",
--	"--no-direct-io"
}

-- full path to llama binaries, download from https://github.com/ggml-org/llama.cpp/releases
llama_bin = [[C:\llama-b7940-cuda-13.1-x64\llama-server.exe]]
llama_tokenize_bin = [[C:\llama-b7940-cuda-13.1-x64\llama-tokenize.exe]]

-- helper function to construct default params for llama-server,
-- trying to best fit models into VRAM and maximize performance, effective for MoE models
function get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	assert(type(ctx_sz)=="number", "ctx_sz param must be a number")
	assert(type(ub)=="number" or type(ub)=="nil", "ub param must be a number or nil")
	assert(type(b)=="number" or type(b)=="nil", "b param must be a number or nil")
	assert(type(ctk)=="string" or type(ctk)=="nil", "ctk param must be a string or nil")
	assert(type(ctv)=="string" or type(ctv)=="nil", "ctv param must be a string or nil")

	local args = concat_arrays(llama_default_args, {"-c",tostring(ctx_sz),"-fit","on","--fit-ctx",tostring(ctx_sz),"--fit-target","256"})
	if type(ub)=="number" then args = concat_arrays(args,{"-ub",tostring(ub)}) end
	if type(b)=="number" then args = concat_arrays(args,{"-b",tostring(b)}) end
	if type(ctk)=="string" then args = concat_arrays(args,{"-ctk",ctk}) end
	if type(ctv)=="string" then args = concat_arrays(args,{"-ctv",ctv}) end
	return concat_arrays(args, {"-m", gguf})
end

-- https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/tree/main
-- https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/tree/main
-- https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/tree/main
-- https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF/tree/main
-- https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF/tree/main
qwen3_30b_instruct_gguf = [[C:\Qwen\Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf]]
qwen3_30b_thinking_gguf = [[C:\Qwen\Qwen3-30B-A3B-Thinking-2507-UD-Q3_K_XL.gguf]]
qwen3_30b_coder_gguf = [[C:\Qwen\Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf]]
qwen3_next_80b_instruct_gguf = [[C:\Qwen\Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf]]
qwen3_next_80b_thinking_gguf = [[C:\Qwen\Qwen3-Next-80B-A3B-Thinking-UD-IQ2_XXS.gguf]]
qwen3_next_coder_gguf = [[C:\Qwen\Qwen3-Coder-Next-UD-IQ2_XXS.gguf]]

function get_qwen3moe_instr_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--spec-type", "ngram-map-k", "--spec-ngram-size-n", "8", "--spec-ngram-size-m", "8", "--spec-ngram-min-hits", "2", "--jinja", "--temp", "0.7", "--min-p", "0.00", "--top-p", "0.80", "--top-k", "20", "--presence-penalty", "0.1", "--repeat-penalty", "1.05"})
end

-- Qwen3-MoE model examples suitable for HW configs with 32G RAM + 8G VRAM (will fill almost all RAM and VRAM)
-- you may lower RAM usage a bit by lowering 'ub' parameter from 2048 to 512, however this will slow down prompt processing
qwen3_30b_instruct_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-30b-instruct",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", qwen3_30b_instruct_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_qwen3moe_instr_args(qwen3_30b_instruct_gguf,10240,2048,2048), context = 10240 },
		{ binary = llama_bin, args = get_qwen3moe_instr_args(qwen3_30b_instruct_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_qwen3moe_instr_args(qwen3_30b_instruct_gguf,30720,2048,2048), context = 30720 },
		{ binary = llama_bin, args = get_qwen3moe_instr_args(qwen3_30b_instruct_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_bin, args = get_qwen3moe_instr_args(qwen3_30b_instruct_gguf,61440,1024,2048,"q8_0","q8_0"), context = 61440 },
		{ binary = llama_bin, args = get_qwen3moe_instr_args(qwen3_30b_instruct_gguf,81920,1024,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_qwen3moe_instr_args(qwen3_30b_instruct_gguf,102400,512,2048,"q8_0","q8_0"), context = 102400 },
	},
}

function get_qwen3moe_next_instr_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--swa-checkpoints", "0", "--jinja", "--temp", "0.7", "--min-p", "0.00", "--top-p", "0.80", "--top-k", "20", "--presence-penalty", "0.1", "--repeat-penalty", "1.05"})
end

qwen3_next_80b_instruct_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-next-80b-instruct",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", qwen3_next_80b_instruct_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,10240,1024,2048), context = 10240 },
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,20480,1024,2048), context = 20480 },
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,30720,1024,2048), context = 30720 },
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,40960,1024,2048), context = 40960 },
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,61440,1024,2048), context = 61440 },
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,81920,1024,2048), context = 81920 },
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,102400,1024,2048), context = 102400 },
		{ binary = llama_bin, args = get_qwen3moe_next_instr_args(qwen3_next_80b_instruct_gguf,122880,1024,2048), context = 122880 },
	},
}

function get_qwen3moe_next_coder_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--temp", "1.0", "--top-p", "0.95", "--min-p", "0.01", "--top-k", "40", "--jinja"})
end

qwen3_next_coder_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-next-coder",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", qwen3_next_coder_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,10240,1024,2048), context = 10240 },
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,20480,1024,2048), context = 20480 },
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,30720,1024,2048), context = 30720 },
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,40960,1024,2048), context = 40960 },
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,61440,1024,2048), context = 61440 },
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,81920,1024,2048), context = 81920 },
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,102400,1024,2048), context = 102400 },
		{ binary = llama_bin, args = get_qwen3moe_next_coder_args(qwen3_next_coder_gguf,122880,1024,2048), context = 122880 },
	},
}

function get_qwen3moe_think_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--spec-type", "ngram-map-k", "--spec-ngram-size-n", "8", "--spec-ngram-size-m", "8", "--spec-ngram-min-hits", "2", "--jinja", "--temp", "0.6", "--min-p", "0.00", "--top-p", "0.95", "--top-k", "20", "--presence-penalty", "0.1", "--repeat-penalty", "1.05"})
end

qwen3_30b_thinking_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-30b-thinking",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", qwen3_30b_thinking_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_qwen3moe_think_args(qwen3_30b_thinking_gguf,10240,2048,2048), context = 10240 },
		{ binary = llama_bin, args = get_qwen3moe_think_args(qwen3_30b_thinking_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_qwen3moe_think_args(qwen3_30b_thinking_gguf,30720,2048,2048), context = 30720 },
		{ binary = llama_bin, args = get_qwen3moe_think_args(qwen3_30b_thinking_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_bin, args = get_qwen3moe_think_args(qwen3_30b_thinking_gguf,61440,1024,2048,"q8_0","q8_0"), context = 61440 },
		{ binary = llama_bin, args = get_qwen3moe_think_args(qwen3_30b_thinking_gguf,81920,1024,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_qwen3moe_think_args(qwen3_30b_thinking_gguf,102400,512,2048,"q8_0","q8_0"), context = 102400 },
	},
}

function get_qwen3moe_next_think_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--swa-checkpoints", "0", "--jinja", "--temp", "0.6", "--min-p", "0.00", "--top-p", "0.95", "--top-k", "20", "--presence-penalty", "0.1", "--repeat-penalty", "1.05"})
end

qwen3_next_80b_thinking_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-next-80b-thinking",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", qwen3_next_80b_thinking_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,10240,1024,2048), context = 10240 },
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,20480,1024,2048), context = 20480 },
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,30720,1024,2048), context = 30720 },
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,40960,1024,2048), context = 40960 },
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,61440,1024,2048), context = 61440 },
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,81920,1024,2048), context = 81920 },
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,102400,1024,2048), context = 102400 },
		{ binary = llama_bin, args = get_qwen3moe_next_think_args(qwen3_next_80b_thinking_gguf,122880,1024,2048), context = 122880 },
	},
}

function get_qwen3moe_coder_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--spec-type", "ngram-map-k", "--spec-ngram-size-n", "8", "--spec-ngram-size-m", "8", "--spec-ngram-min-hits", "2", "--jinja", "--temp", "0.7", "--min-p", "0.0", "--top-p", "0.80", "--top-k", "20", "--repeat-penalty", "1.05"})
end

qwen3_30b_coder_model = {
	engine = presets.engines.llamacpp,
	name = "qwen3-30b-coder",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", qwen3_30b_coder_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_qwen3moe_coder_args(qwen3_30b_coder_gguf,10240,2048,2048), context = 10240 },
		{ binary = llama_bin, args = get_qwen3moe_coder_args(qwen3_30b_coder_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_qwen3moe_coder_args(qwen3_30b_coder_gguf,30720,2048,2048), context = 30720 },
		{ binary = llama_bin, args = get_qwen3moe_coder_args(qwen3_30b_coder_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_bin, args = get_qwen3moe_coder_args(qwen3_30b_coder_gguf,61440,1024,2048,"q8_0","q8_0"), context = 61440 },
		{ binary = llama_bin, args = get_qwen3moe_coder_args(qwen3_30b_coder_gguf,81920,1024,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_qwen3moe_coder_args(qwen3_30b_coder_gguf,102400,512,2048,"q8_0","q8_0"), context = 102400 },
	},
}

-- https://huggingface.co/bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF
ministral_3_3b_instruct_gguf = [[C:\Mistral\mistralai_Ministral-3-3B-Instruct-2512-Q4_K_L.gguf]]

function get_ministral_3_instr_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--jinja", "--temp", "0.1"})
end

ministral_3_3b_instruct_model = {
	engine = presets.engines.llamacpp,
	name = "ministral-3-3b-instruct",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", ministral_3_3b_instruct_gguf }, extra_tokens_per_message = 8, extra_tokens = 550 },
	variants = {
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,10240,4096,4096), context = 10240 },
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,30720,1024,2048), context = 30720 },
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,40960,1024,2048), context = 40960 },
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,61440,1024,2048,"q8_0","q8_0"), context = 61440 },
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,81920,512,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,102400,512,2048,"q8_0","q8_0"), context = 102400 },
		{ binary = llama_bin, args = get_ministral_3_instr_args(ministral_3_3b_instruct_gguf,122880,512,2048,"q8_0","q8_0"), context = 122880 },
	},
}

-- https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF
-- https://huggingface.co/noctrex/GLM-4.7-Flash-MXFP4_MOE-GGUF
glm_47_flash_gguf = [[C:\GLM\GLM-4.7-Flash-UD-Q5_K_XL.gguf]] -- heavier quant, seem to be a bit overkill for my tasks
glm_47_flash_mxfp4_gguf = [[C:\GLM\GLM-4.7-Flash-MXFP4_MOE.gguf]]  -- small fast static quant, seem to work better with non-english texts for me than other 4bit quants with imatrix

-- "--temp", "0.7", "--top-p", "1.0", -- for tool calling
-- "--temp", "1.0", "--top-p", "0.95", -- for general use-case
-- "--spec-type", "ngram-mod", "--spec-ngram-size-n", "24", "--draft-min", "48", "--draft-max", "64"
function get_glm_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_llama_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--spec-type", "ngram-map-k", "--spec-ngram-size-n", "8", "--spec-ngram-size-m", "8", "--spec-ngram-min-hits", "2", "--jinja", "--temp", "0.7", "--repeat-penalty", "1.0"})
end

function get_glm_instruct_args(gguf, ctx_sz, ub, b, ctk, ctv)
	local args = get_glm_args(gguf, ctx_sz, ub, b, ctk, ctv)
	return concat_arrays(args, {"--chat-template-kwargs", [[{"enable_thinking":false}]]})
end

glm_47_flash_model = {
	engine = presets.engines.llamacpp,
	name = "glm-47-flash",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", glm_47_flash_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,10240,2048,2048), context = 10240 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,30720,2048,2048), context = 30720 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,61440,2048,2048), context = 61440 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,81920,2048,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,102400,2048,2048,"q8_0","q8_0"), context = 102400 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_gguf,122880,2048,2048,"q8_0","q8_0"), context = 122880 },
	},
}

glm_47_flash_instruct_model = {
	engine = presets.engines.llamacpp,
	name = "glm-47-flash-instruct",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", glm_47_flash_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,10240,2048,2048), context = 10240 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,30720,2048,2048), context = 30720 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,61440,2048,2048), context = 61440 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,81920,2048,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,102400,2048,2048,"q8_0","q8_0"), context = 102400 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_gguf,122880,2048,2048,"q8_0","q8_0"), context = 122880 },
	},
}

glm_47_flash_mxfp4_model = {
	engine = presets.engines.llamacpp,
	name = "glm-47-flash-mxfp4",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", glm_47_flash_mxfp4_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,10240,2048,2048), context = 10240 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,30720,2048,2048), context = 30720 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,61440,2048,2048), context = 61440 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,81920,2048,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,102400,2048,2048,"q8_0","q8_0"), context = 102400 },
		{ binary = llama_bin, args = get_glm_args(glm_47_flash_mxfp4_gguf,122880,2048,2048,"q8_0","q8_0"), context = 122880 },
	},
}

glm_47_flash_mxfp4_instruct_model = {
	engine = presets.engines.llamacpp,
	name = "glm-47-flash-mxfp4-instruct",
	connect = llama_url,
	tokenization = { binary = llama_tokenize_bin, extra_args = { "-m", glm_47_flash_mxfp4_gguf }, extra_tokens_per_message = 8, extra_tokens = 0 },
	variants = {
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,10240,2048,2048), context = 10240 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,20480,2048,2048), context = 20480 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,30720,2048,2048), context = 30720 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,40960,2048,2048), context = 40960 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,61440,2048,2048), context = 61440 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,81920,2048,2048,"q8_0","q8_0"), context = 81920 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,102400,2048,2048,"q8_0","q8_0"), context = 102400 },
		{ binary = llama_bin, args = get_glm_instruct_args(glm_47_flash_mxfp4_gguf,122880,2048,2048,"q8_0","q8_0"), context = 122880 },
	},
}

-- https://huggingface.co/Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf/tree/main
snowflake_arctic_embed_gguf = [[C:\Embedding\snowflake-arctic-embed-l-v2.0-f16.gguf]]

function get_arctic_args(gguf)
	return concat_arrays(llama_default_args,{"--embedding", "-c", "8192", "-ub", "4096", "-b", "4096", "-fit", "on", "--fit-ctx", "8192", "--fit-target", "256", "-m", gguf})
end

snowflake_arctic_embed_model = {
	engine = presets.engines.llamacpp,
	name = "snowflake-arctic-embed",
	connect = llama_url,
	variants = {
		{ binary = llama_bin, args = get_arctic_args(snowflake_arctic_embed_gguf), context = 8192 },
	},
}

models = {
	qwen3_30b_instruct_model,
	qwen3_30b_thinking_model,
	qwen3_30b_coder_model,
	qwen3_next_80b_instruct_model,
	qwen3_next_80b_thinking_model,
	qwen3_next_coder_model,
	glm_47_flash_model,
	glm_47_flash_instruct_model,
	glm_47_flash_mxfp4_model,
	glm_47_flash_mxfp4_instruct_model,
	ministral_3_3b_instruct_model,
	snowflake_arctic_embed_model,
}
