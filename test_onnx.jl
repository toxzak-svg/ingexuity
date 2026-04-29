using ONNX
# Check if it has inference functions
functions = [n for n in names(ONNX) if !startswith(string(n), "#") && !startswith(string(n), "Symbol")]
println("Public functions: ", functions)

# Check if there's a sub-module for runtime
if isdefined(ONNX, :Runtime)
    println("Has Runtime module")
end

# Check docstring
println("\nONNX module doc: ", @doc ONNX)