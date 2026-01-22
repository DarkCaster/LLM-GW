-- config file checks and asserts


-- Helper function to check if a value is of expected type
local function assert_type(value, expected_type, path)
	if value ~= nil then
		local actual_type = type(value)
		if actual_type ~= expected_type then
			error(string.format("Configuration error at '%s': expected %s, got %s", path, expected_type, actual_type))
		end
	end
end

-- Helper function to check if a value exists
local function assert_exists(value, path)
	if value == nil then
		error(string.format("Configuration error: missing required field '%s'", path))
	end
end

-- Helper function to check if a value is a positive number
local function assert_positive_number(value, path)
	if value == nil then
		error(string.format("Configuration error: missing required field '%s'", path))
	end
	if type(value) ~= "number" or value <= 0 then
		error(string.format("Configuration error at '%s': must be a positive number, got %s", path, value))
	end
end

-- Check llama.cpp engine variant
local function check_llamacpp_variant(variant, model_name, variant_index, model)
	local base_path = string.format("models.%s.variants[%d]", model_name, variant_index)
	-- Check binary (mandatory)
	assert_exists(variant.binary, base_path .. ".binary")
	assert_type(variant.binary, "string", base_path .. ".binary")
	-- Check connect string, add it from parent model table if missing
	variant.connect = variant.connect or model.connect
	assert_exists(variant.connect, base_path .. ".connect")
	assert_type(variant.connect, "string", base_path .. ".connect")
	-- Check args (mandatory)
	assert_exists(variant.args, base_path .. ".args")
	assert_type(variant.args, "table", base_path .. ".args")
	-- Check context (mandatory)
	assert_exists(variant.context, base_path .. ".context")
	assert_type(variant.context, "number", base_path .. ".context")
	-- Set defaults for optional timeout parameters
	variant.health_check_timeout = variant.health_check_timeout or model.health_check_timeout
	variant.engine_startup_timeout = variant.engine_startup_timeout or model.engine_startup_timeout
	variant.engine_idle_timeout = variant.engine_idle_timeout or model.engine_idle_timeout
	-- Validate timeout parameters are positive numbers
	assert_positive_number(variant.health_check_timeout, base_path .. ".health_check_timeout")
	assert_positive_number(variant.engine_startup_timeout, base_path .. ".engine_startup_timeout")
	assert_positive_number(variant.engine_idle_timeout, base_path .. ".engine_idle_timeout")
end

-- Check model configuration based on engine type
local function check_model(model, model_name)
	local base_path = string.format("models.%s", model_name)
	-- Check engine field (mandatory)
	assert_exists(model.engine, base_path .. ".engine")
	assert_type(model.engine, "string", base_path .. ".engine")
	-- Set defaults for optional timeout parameters
	model.health_check_timeout = model.health_check_timeout or server.health_check_timeout
	model.engine_startup_timeout = model.engine_startup_timeout or server.engine_startup_timeout
	model.engine_idle_timeout = model.engine_idle_timeout or server.engine_idle_timeout
	-- Validate timeout parameters are positive numbers
	assert_positive_number(model.health_check_timeout, base_path .. ".health_check_timeout")
	assert_positive_number(model.engine_startup_timeout, base_path .. ".engine_startup_timeout")
	assert_positive_number(model.engine_idle_timeout, base_path .. ".engine_idle_timeout")
	-- Check variants (mandatory)
	assert_exists(model.variants, base_path .. ".variants")
	assert_type(model.variants, "table", base_path .. ".variants")
	-- Check that variants is not empty
	if #model.variants == 0 then
		error(string.format("Configuration error at '%s.variants': must contain at least one variant", base_path))
	end
	-- Check each variant based on engine type
	for i, variant in ipairs(model.variants) do
		assert_type(variant, "table", base_path .. string.format(".variants[%d]", i))
		if model.engine == presets.engines.llamacpp then
			check_llamacpp_variant(variant, model_name, i, model)
		else
			error(string.format("Configuration error at '%s.engine': unknown engine type '%s'", base_path, model.engine))
		end
	end
end

-- Check server configuration
assert_exists(server, "server")
assert_type(server, "table", "server")

-- Check listen_v4 (mandatory)
assert_exists(server.listen_v4, "server.listen_v4")
assert_type(server.listen_v4, "string", "server.listen_v4")

-- Check listen_v6 (mandatory)
assert_exists(server.listen_v6, "server.listen_v6")
assert_type(server.listen_v6, "string", "server.listen_v6")

-- Check health_check_timeout (mandatory)
assert_positive_number(server.health_check_timeout, "server.health_check_timeout")

-- Check engine_startup_timeout (mandatory)
assert_positive_number(server.engine_startup_timeout, "server.engine_startup_timeout")

-- Check engine_idle_timeout (mandatory)
assert_positive_number(server.engine_idle_timeout, "server.engine_idle_timeout")

-- Check models configuration
assert_exists(models, "models")
assert_type(models, "table", "models")

-- Check that models table is not empty
if #models == 0 then
    error("Configuration error: 'models' table must contain at least one model")
end

-- Check each model in the models table
-- Track model names to ensure uniqueness
local seen_names = {}
for i, model in ipairs(models) do
	local base_path = string.format("models[%d]", i)
	assert_type(model, "table", base_path)
	-- Check name field in model table
	assert_exists(model.name, base_path..".name")
	assert_type(model.name, "string", base_path..".name")
	-- Check for duplicate model names
	if seen_names[model.name] then
		error(string.format("Configuration error: duplicate model name '%s' found. Model names must be unique.", model.name))
	end
	seen_names[model.name] = true
	-- Try to find model table name by searching in global scope
	local model_name = nil
	for name, value in pairs(_G) do
		if value == model and name ~= "models" and type(name) == "string" then
			model_name = name
			break
		end
	end
	if not model_name then
		model_name = string.format("model_%d", i)
	end
	check_model(model, model_name)
end
