using Test

# Load the Julia-native model module directly so the package can keep it
# experimental without selecting it as the default chat backend.
include(joinpath(@__DIR__, "..", "src", "modules", "NanoGPT.jl"))
using .NanoGPT

@testset "Julia-native transformer baseline" begin
    config = NanoGPT.TransformerConfig(
        vocab_size=264,
        max_seq_len=16,
        n_embed=8,
        n_heads=2,
        n_layers=1,
        n_hidden=16,
        dropout=0.0f0,
    )

    tokenizer = NanoGPT.SimpleTokenizer(config.vocab_size)
    tokens = NanoGPT.encode(tokenizer, "Julia")

    @test !isempty(tokens)
    @test NanoGPT.decode(tokenizer, tokens) == "Julia"

    first_model = NanoGPT.TransformerModel(config)
    second_model = NanoGPT.TransformerModel(config)

    first_logits = first_model(tokens)
    second_logits = second_model(tokens)

    @test length(first_logits) == config.vocab_size
    @test all(isfinite, first_logits)
    @test first_logits == second_logits
    @test NanoGPT.param_count(config) > 0
end

@testset "Julia-native tokenizer declared domain" begin
    tokenizer = NanoGPT.SimpleTokenizer(264)

    # The current SimpleTokenizer is byte-oriented only for the first 256
    # scalar values. Keep the supported Phase-0 fixture explicit rather than
    # implying full Unicode conformance.
    for sample in ("hello", "Rust and Julia", "123 !?")
        @test NanoGPT.decode(tokenizer, NanoGPT.encode(tokenizer, sample)) == sample
    end
end
