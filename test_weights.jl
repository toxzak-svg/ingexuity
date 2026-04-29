using IngExuity
IGSD = IngExuity.IGSDCore

println("Saving weights...")
IngExuity.IGSDCore.save_weights(IGSD.DEFAULT_CORE, "test_weights.bin")
println("Saved! Size: $(filesize("test_weights.bin") / 1e6) MB")

println("\nLoading weights...")
loaded = IngExuity.IGSDCore.IGSDCore("test_weights.bin")
println("Loaded config: $(loaded.config)")

println("\nVerifying weights match...")
orig = IGSD.DEFAULT_CORE
println("wte match: $(orig.wte == loaded.wte)")
println("lm_head match: $(orig.lm_head == loaded.lm_head)")
println("layers match: $(orig.layers[1].wu == loaded.layers[1].wu)")