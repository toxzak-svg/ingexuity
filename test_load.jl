#!/usr/bin/env julia
using Pkg
Pkg.resolve()
using IngExuity
println("IngExuity loaded successfully!")
println("Gemma LLM: ", IngExuity.GemmaProvider.is_loaded(IngExuity.GEMMA_LLM))