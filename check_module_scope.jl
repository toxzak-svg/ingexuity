module Test
include("src/modules/Types.jl")
println("Test has Types: ", isdefined(Test, :Types))
println("Test.Types = ", Test.Types)
println("Main has Types: ", isdefined(Main, :Types))
end
