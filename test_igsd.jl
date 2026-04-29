using IngExuity
IGSD = IngExuity.IGSDCore
println("IngExuity loaded with IGSDCore!")
println("\nIGSDCore info:")
println("  Config: $(IGSD.DEFAULT_CONFIG)")
params = IGSD.param_count(IGSD.DEFAULT_CONFIG)
println("  Params: ~$(round(params/1e6, digits=1))M")
println("  Vocab: $(IGSD.DEFAULT_CONFIG.vocab_size)")
println("  State: $(IGSD.DEFAULT_CONFIG.d_state)")
println("  Layers: $(IGSD.DEFAULT_CONFIG.n_layers)")

println("\nTesting IGSDCore.chat()...")
result = IGSD.chat("Hello, how are you?")
println("Input: Hello, how are you?")
println("Output: $result")