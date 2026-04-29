# ============================================================================
# Transformer.jl — Pure Julia Transformer for IngExuity
# Inference-only implementation (no training dependencies)
# Based on GPT-2 architecture
# ============================================================================
module Transformer

using Random
using Statistics

export TransformerModel, TransformerConfig, generate_text

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef mutable struct TransformerConfig
    vocab_size::Int = 50257
    max_seq_len::Int = 1024
    n_embed::Int = 384
    n_heads::Int = 6
    n_layers::Int = 6
    n_hidden::Int = 1536
    dropout::Float32 = 0.0f0
    bias::Bool = true
end

function param_count(cfg::TransformerConfig)::Int
    vocab = cfg.vocab_size * cfg.n_embed
    pos = cfg.max_seq_len * cfg.n_embed
    attn = cfg.n_layers * (
        3 * cfg.n_embed * cfg.n_embed +  # Wq, Wk, Wv
        cfg.n_embed * cfg.n_embed +        # Wo
        2 * cfg.n_embed                    # layer norms
    )
    ffn = cfg.n_layers * (
        2 * cfg.n_embed * cfg.n_hidden +   # W1, W2
        2 * cfg.n_embed                    # layer norms
    )
    head = cfg.vocab_size * cfg.n_embed
    total = vocab + pos + attn + ffn + head + 2 * cfg.n_embed
    return total
end

# ============================================================================
# Activation functions
# ============================================================================

gelu(x) = @. 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

function softmax!(out::AbstractVecOrMat, x::AbstractVecOrMat)
    if size(x) == size(out)
        copy!(out, x)
    else
        out .= x
    end
    out_exp = exp.(out .- maximum(out, dims=1))
    out ./= sum(out_exp, dims=1)
    return out
end

function softmax(x::AbstractVecOrMat)
    return softmax!(similar(x), x)
end

# ============================================================================
# Matrix multiplication helpers
# ============================================================================

matmul(x, w) = x * w

# ============================================================================
# Layer Normalization
# ============================================================================

struct LayerNorm
    weight::Vector{Float32}
    bias::Vector{Float32}
    eps::Float32
    n::Int
end

function LayerNorm(n::Int; bias::Bool=true, eps::Float32=1f-5)
    return LayerNorm(
        ones(Float32, n),
        bias ? zeros(Float32, n) : Float32[],
        eps,
        n
    )
end

function (ln::LayerNorm)(x::AbstractMatrix{Float32})::AbstractMatrix{Float32}
    # x: (n_embed, seq_len)
    mean = dropdims(mean(x, dims=1), dims=1)
    var = dropdims(var(x, dims=1, corrected=true), dims=1)
    x_norm = @. (x - mean) / sqrt(var + ln.eps)
    return @. ln.weight * x_norm + ln.bias
end

# ============================================================================
# Causal Self-Attention
# ============================================================================

struct CausalAttention
    Wq::Matrix{Float32}
    Wk::Matrix{Float32}
    Wv::Matrix{Float32}
    Wo::Matrix{Float32}
    n_heads::Int
    head_dim::Int
    scale::Float32
end

function CausalAttention(n_embed::Int, n_heads::Int)
    @assert n_embed % n_heads == 0
    head_dim = n_embed ÷ n_heads
    scale = Float32(1.0 / sqrt(head_dim))

    rng = Random.Xoshiro(42)
    Wq = randn(rng, Float32, n_embed, n_embed) * 0.02f0
    Wk = randn(rng, Float32, n_embed, n_embed) * 0.02f0
    Wv = randn(rng, Float32, n_embed, n_embed) * 0.02f0
    Wo = randn(rng, Float32, n_embed, n_embed) * 0.02f0

    return CausalAttention(Wq, Wk, Wv, Wo, n_heads, head_dim, scale)
end

function (attn::CausalAttention)(x::AbstractMatrix{Float32})::AbstractMatrix{Float32}
    # x: (n_embed, seq_len)
    n_embed, T = size(x)
    n_heads = attn.n_heads
    head_dim = attn.head_dim

    Q = attn.Wq' * x  # (n_embed, seq_len)
    K = attn.Wk' * x
    V = attn.Wv' * x

    Q_seq = reshape(Q, n_heads, head_dim, T)
    K_seq = reshape(K, n_heads, head_dim, T)
    V_seq = reshape(V, n_heads, head_dim, T)

    attn_scores = zeros(Float32, T, T, n_heads)
    for h in 1:n_heads
        for i in 1:T
            for j in 1:T
                if j <= i
                    attn_scores[i, j, h] = attn.scale * dot(Q_seq[h, :, i], K_seq[h, :, j])
                end
            end
        end
    end

    attn_weights = zeros(Float32, T, T, n_heads)
    for h in 1:n_heads
        for i in 1:T
            row_sum = sum(exp(attn_scores[i, j, h]) for j in 1:T if j <= i)
            for j in 1:T
                if j <= i
                    attn_weights[i, j, h] = exp(attn_scores[i, j, h]) / row_sum
                end
            end
        end
    end

    out = zeros(Float32, n_embed, T)
    for h in 1:n_heads
        for i in 1:T
            for j in 1:T
                if j <= i
                    out[h*head_dim-(head_dim-1):h*head_dim, i] .+=
                        attn_weights[i, j, h] .* V_seq[h, :, j]
                end
            end
        end
    end

    return attn.Wo' * out
end

# ============================================================================
# Feed-Forward Network
# ============================================================================

struct FeedForward
    W1::Matrix{Float32}
    b1::Vector{Float32}
    W2::Matrix{Float32}
    b2::Vector{Float32}
    activation::Function
end

function FeedForward(n_embed::Int, n_hidden::Int)
    rng = Random.Xoshiro(42)
    W1 = randn(rng, Float32, n_hidden, n_embed) * 0.02f0
    b1 = zeros(Float32, n_hidden)
    W2 = randn(rng, Float32, n_embed, n_hidden) * 0.02f0
    b2 = zeros(Float32, n_embed)
    return FeedForward(W1, b1, W2, b2, gelu)
end

function (ff::FeedForward)(x::AbstractMatrix{Float32})::AbstractMatrix{Float32}
    return ff.W2 * ff.activation.(ff.W1 * x .+ ff.b1) .+ ff.b2
end

# ============================================================================
# Transformer Block
# ============================================================================

struct TransformerBlock
    attn::CausalAttention
    ff::FeedForward
    ln1::LayerNorm
    ln2::LayerNorm
end

function TransformerBlock(n_embed::Int, n_heads::Int, n_hidden::Int)
    return TransformerBlock(
        CausalAttention(n_embed, n_heads),
        FeedForward(n_embed, n_hidden),
        LayerNorm(n_embed),
        LayerNorm(n_embed)
    )
end

function (block::TransformerBlock)(x::AbstractMatrix{Float32})::AbstractMatrix{Float32}
    x = x .+ block.attn(block.ln1(x))
    x = x .+ block.ff(block.ln2(x))
    return x
end

# ============================================================================
# Full Transformer Model
# ============================================================================

struct TransformerModel
    config::TransformerConfig
    token_embedding::Matrix{Float32}
    position_embedding::Matrix{Float32}
    blocks::Vector{TransformerBlock}
    final_ln::LayerNorm
    lm_head::Matrix{Float32}
end

function TransformerModel(config::TransformerConfig)
    rng = Random.Xoshiro(42)
    n = config.n_embed

    token_embedding = randn(rng, Float32, config.vocab_size, n) * 0.02f0
    position_embedding = randn(rng, Float32, config.max_seq_len, n) * 0.02f0

    blocks = [TransformerBlock(n, config.n_heads, config.n_hidden) for _ in 1:config.n_layers]
    final_ln = LayerNorm(n)

    lm_head = permutedims(token_embedding, [2, 1])

    return TransformerModel(
        config,
        token_embedding,
        position_embedding,
        blocks,
        final_ln,
        lm_head
    )
end

function (model::TransformerModel)(tokens::AbstractMatrix{Int})::AbstractMatrix{Float32}
    cfg = model.config
    T = size(tokens, 1)

    x = model.token_embedding[tokens .+ 1, :]'  # (n_embed, T)
    x = x .+ model.position_embedding[1:T, :]'

    for block in model.blocks
        x = block(x)
    end

    x = model.final_ln(x)
    logits = model.lm_head * x

    return logits
end

function (model::TransformerModel)(tokens::AbstractVector{Int})::AbstractVector{Float32}
    return model(reshape(tokens, length(tokens), 1))[:, 1]
end

function generate(model::TransformerModel, tokenizer, seed::String;
                  max_new_tokens::Int=100, temperature::Float32=1.0f0,
                  top_k::Int=40, rng=Random.Xoshiro(42))::String

    tokens = encode(tokenizer, seed)
    max_len = min(model.config.max_seq_len, length(tokens) + max_new_tokens)

    while length(tokens) < max_len
        context = tokens[max(1, end-model.config.max_seq_len+1):end]
        logits = model(context)
        logits = logits[end] ./ temperature

        if top_k > 0
            top_idx = partialsortperm(logits, 1:top_k, rev=true)
            mask = trues(length(logits))
            mask[top_idx] .= false
            logits[mask] .= -Inf
        end

        probs = softmax(logits)
        next_token = sample(rng, 1:length(probs), StatsBase.Weights(probs))

        push!(tokens, next_token)

        if next_token == tokenizer.eot_token
            break
        end
    end

    return decode(tokenizer, tokens)
end

function Base.show(io::IO, m::TransformerModel)
    cfg = m.config
    params = param_count(cfg)
    println(io, "TransformerModel($(cfg.n_layers) layers, $(cfg.n_heads) heads, $(cfg.n_embed) embed)")
    println(io, "  → ~$(round(params/1e6, digits=1))M params, $(cfg.vocab_size) vocab, $(cfg.max_seq_len) ctx")
end

end # module