using Test
using Random
using Statistics

# Load the Julia-native model module directly so the package can keep it
# experimental without selecting it as the default chat backend.
include(joinpath(@__DIR__, "..", "src", "modules", "NanoGPT.jl"))
using .NanoGPT

@testset "Julia-native transformer correctness" begin
    config = NanoGPT.TransformerConfig(
        vocab_size=264,
        max_seq_len=16,
        n_embed=8,
        n_heads=2,
        n_layers=2,
        n_hidden=16,
        dropout=0.0f0,
    )

    tokenizer = NanoGPT.SimpleTokenizer(config.vocab_size)
    tokens = NanoGPT.encode(tokenizer, "Julia")

    @test !isempty(tokens)
    @test all(token -> 0 <= token < config.vocab_size, tokens)
    @test NanoGPT.decode(tokenizer, tokens) == "Julia"

    first_model = NanoGPT.TransformerModel(config)
    second_model = NanoGPT.TransformerModel(config)
    all_logits = NanoGPT.forward_all(first_model, tokens)
    next_logits = first_model(tokens)

    @test size(all_logits) == (config.vocab_size, length(tokens))
    @test length(next_logits) == config.vocab_size
    @test all(isfinite, all_logits)
    @test next_logits == all_logits[:, end]
    @test next_logits == second_model(tokens)

    # A single seeded generator is threaded through model construction: model
    # creation is reproducible, but separate layers no longer start identically.
    @test first_model.blocks[1].attn.Wq != first_model.blocks[2].attn.Wq
    @test first_model.lm_head === first_model.token_embedding
    @test NanoGPT.param_count(config) == 3456

    token_batch = hcat(tokens, tokens)
    batched_logits = first_model(token_batch)
    @test size(batched_logits) == (config.vocab_size, length(tokens), 2)
    @test batched_logits[:, :, 1] == batched_logits[:, :, 2]

    targets = mod.(tokens .+ 1, config.vocab_size)
    loss = NanoGPT.cross_entropy_loss(first_model, tokens, targets)
    @test isfinite(loss)
    @test loss > 0.0f0
end

@testset "LayerNorm normalizes each token" begin
    norm = NanoGPT.LayerNorm(4)
    inputs = Float32[
        1 4 2;
        2 1 5;
        3 3 1;
        4 2 4
    ]
    outputs = norm(inputs)

    @test all(isapprox.(vec(mean(outputs; dims=1)), 0.0f0; atol=1f-5))
    variances = vec(sum(abs2, outputs; dims=1) ./ size(outputs, 1))
    @test all(isapprox.(variances, 1.0f0; atol=1f-4))
end

@testset "Causal attention is stable and future-blind" begin
    attention = NanoGPT.CausalAttention(8, 2; rng=Random.Xoshiro(7))
    inputs = randn(Random.Xoshiro(9), Float32, 8, 5)
    baseline = attention(inputs)

    changed_future = copy(inputs)
    changed_future[:, end] .+= 100.0f0
    modified = attention(changed_future)

    @test all(isfinite, baseline)
    @test isapprox(baseline[:, 1:4], modified[:, 1:4]; atol=1f-6, rtol=1f-6)
    @test !isapprox(baseline[:, end], modified[:, end]; atol=1f-6, rtol=1f-6)

    probabilities = NanoGPT.softmax(Float32[-1000, 0, 1000])
    @test all(isfinite, probabilities)
    @test all(probability -> probability >= 0.0f0, probabilities)
    @test isapprox(sum(probabilities), 1.0f0; atol=1f-6)
end

@testset "Cross entropy uses zero-based target IDs" begin
    logits = zeros(Float32, 4, 3)
    targets = [0, 1, 3]
    @test isapprox(NanoGPT.cross_entropy_loss(logits, targets), log(4.0f0); atol=1f-6)
    @test_throws ArgumentError NanoGPT.cross_entropy_loss(logits, [0, 1, 4])
end

@testset "Julia-native tokenizer byte domain" begin
    tokenizer = NanoGPT.SimpleTokenizer(264)
    for sample in ("hello", "Rust and Julia", "123 !?", "Julia ∞")
        @test NanoGPT.decode(tokenizer, NanoGPT.encode(tokenizer, sample)) == sample
    end

    trained = NanoGPT.train(["abababab", "abab"], 264)
    @test NanoGPT.decode(trained, NanoGPT.encode(trained, "abab")) == "abab"
end

@testset "Configuration and bias validation" begin
    @test_throws ArgumentError NanoGPT.TransformerModel(
        NanoGPT.TransformerConfig(n_embed=7, n_heads=2),
    )

    no_bias = NanoGPT.TransformerConfig(
        vocab_size=264,
        max_seq_len=8,
        n_embed=8,
        n_heads=2,
        n_layers=1,
        n_hidden=16,
        bias=false,
    )
    model = NanoGPT.TransformerModel(no_bias)
    @test all(isfinite, model([0, 1, 2]))
end
