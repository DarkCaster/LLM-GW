-- define some helper constants

presets={}

-- engines, supported by this LLM-GW software
presets.engines={}
presets.engines.llamacpp="llama.cpp"
presets.engines.llamacpp_sideload="llama.cpp.sideload"

function concat_arrays(a1, a2)
	local result = {}
	for _,v in ipairs(a1) do
		table.insert(result, v)
	end
	for _,v in ipairs(a2) do
		table.insert(result, v)
	end
	return result
end

function merge_tables(t1, t2)
	local function is_positive_integer(n)
		return type(n) == "number" and n >= 1 and n == math.floor(n)
	end
	local result = {}
	for k, v in pairs(t1) do
		result[k] = v
	end
	for k, v in pairs(t2) do
		if is_positive_integer(k) then
			table.insert(result, v)
		else
			result[k] = v
		end
	end
	return result
end
