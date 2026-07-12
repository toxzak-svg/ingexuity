using Test
using Random

include(joinpath(@__DIR__, "..", "src", "modules", "J1Transformer.jl"))
using .J1Transformer

@testset "J1 63M configuration" begin
    config = J1Transformer.j1_63m_config()
    @test J1Transformer.param_count(config) == 63_252_992
    @test config.d_model == 512
    @test config.n_query_heads == 8
    @test config.n_kv_heads == 2
    @test config.d_ff == 1_365
end

@testset "J1 deterministic modern baseline" begin
    config = J1Transformer.tiny_config()
    first_model = J1Transformer.J1Model(config)
    second_model = J1Transformer.J1Model(config)
    tokens = [74, 117, 108, 105, 97]

    logits = J1Transformer.forward_all(first_model, tokens)
    @test size(logits) == (config.vocab_size, length(tokens))
    @test all(isfinite, logits)
    @test first_model(tokens) == logits[:, end]
    @test first_model(tokens) == second_model(tokens)
    @test first_model.blocks[1].attention.Wq != first_model.blocks[2].attention.Wq

    changed = copy(tokens)
    changed[end] = 98
    changed_logits = J1Transformer.forward_all(first_model, changed)
    @test isapprox(logits[:, 1:(end - 1)], changed_logits[:, 1:(end - 1)]; atol=2f-5, rtol=2f-5)
    @test !isapprox(logits[:, end], changed_logits[:, end]; atol=1f-6, rtol=1f-6)
end

@testset "RMSNorm and RoPE invariants" begin
    norm = J1Transformer.RMSNorm(8)
    inputs = randn(Random.Xoshiro(11), Float32, 8, 4)
    normalized = norm(inputs)
    rms = sqrt.(vec(sum(abs2, normalized; dims=1) ./ size(normalized, 1)))
    @test all(isapprox.(rms, 1.0f0; atol=2f-5))

    values = randn(Random.Xoshiro(12), Float32, 8, 3)
    rotated = J1Transformer.apply_rope(values, 2, [0, 1, 2])
    @test isapprox(vec(sum(abs2, values; dims=1)), vec(sum(abs2, rotated; dims=1)); atol=2f-5, rtol=2f-5)
    @test rotated[:, 1] == values[:, 1]
    @test rotated[:, 2] != values[:, 2]
end

@testset "Grouped-query attention shape and causality" begin
    config = J1Transformer.tiny_config()
    attention = J1Transformer.GroupedQueryAttention(config; rng=Random.Xoshiro(21))
    inputs = randn(Random.Xoshiro(22), Float32, config.d_model, 6)
    baseline = attention(inputs, collect(0:5))

    changed = copy(inputs)
    changed[:, end] .+= 50.0f0
    modified = attention(changed, collect(0:5))

    @test size(baseline) == size(inputs)
    @test attention.n_query_heads == 4
    @test attention.n_kv_heads == 2
    @test all(isfinite, baseline)
    @test isapprox(baseline[:, 1:5], modified[:, 1:5]; atol=2f-5, rtol=2f-5)
end

@testset "Incremental KV cache matches full forward" begin
    config = J1Transformer.tiny_config(max_seq_len=12)
    model = J1Transformer.J1Model(config; rng=Random.Xoshiro(31))
    tokens = [1, 4, 9, 16, 25, 36]
    cache = J1Transformer.init_cache(model)

    for position in eachindex(tokens)
        incremental = J1Transformer.decode_step!(model, tokens[position], cache)
        full = J1Transformer.forward_all(model, tokens[1:position])[:, end]
        @test isapprox(incremental, full; atol=3f-5, rtol=3f-5)
        @test cache.length == position
    end

    prefill_logits, prefill_cache = J1Transformer.prefill(model, tokens)
    @test isapprox(prefill_logits, model(tokens); atol=3f-5, rtol=3f-5)
    @test prefill_cache.length == length(tokens)

    short_model = J1Transformer.J1Model(J1Transformer.tiny_config(max_seq_len=2))
    short_cache = J1Transformer.init_cache(short_model)
    J1Transformer.decode_step!(short_model, 1, short_cache)
    J1Transformer.decode_step!(short_model, 2, short_cache)
    @test_throws ArgumentError J1Transformer.decode_step!(short_model, 3, short_cache)
end

@testset "J1 loss and configuration validation" begin
    logits = zeros(Float32, 5, 3)
    @test isapprox(J1Transformer.cross_entropy_loss(logits, [0, 2, 4]), log(5.0f0); atol=1f-6)
    @test_throws ArgumentError J1Transformer.cross_entropy_loss(logits, [0, 2, 5])

    @test_throws ArgumentError J1Transformer.J1Model(
        J1Transformer.J1Config(d_model=18, n_query_heads=4, n_kv_heads=2),
    )
    @test_throws ArgumentError J1Transformer.J1Model(
        J1Transformer.J1Config(d_model=24, n_query_heads=6, n_kv_heads=4),
    )
    @test_throws ArgumentError J1Transformer.J1Model(
        J1Transformer.J1Config(d_model=12, n_query_heads=4, n_kv_heads=2),
    )
end
