# ============================================================================
# J1Transformer.jl — modern Julia-native control architecture for IngExuity
# ============================================================================
module J1Transformer

using Random

export J1Config, J1Model, KVCache
export tiny_config, j1_63m_config, param_count
export forward_all, cross_entropy_loss, init_cache, decode_step!, prefill
export RMSNorm, apply_rope, GroupedQueryAttention, SwiGLU

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef struct J1Config
    vocab_size::Int = 16_000
    max_seq_len::Int = 4_096
    d_model::Int = 512
    n_layers::Int = 20
    n_query_heads::Int = 8
    n_kv_heads::Int = 2
    d_ff::Int = 1_365
    rope_base::Float32 = 10_000.0f0
    norm_eps::Float32 = 1.0f-5
    bias::Bool = false
end

function validate_config(config::J1Config)
    config.vocab_size > 1 || throw(ArgumentError("vocab_size must exceed one"))
    config.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    config.d_model > 0 || throw(ArgumentError("d_model must be positive"))
    config.n_layers > 0 || throw(ArgumentError("n_layers must be positive"))
    config.n_query_heads > 0 || throw(ArgumentError("n_query_heads must be positive"))
    config.n_kv_heads > 0 || throw(ArgumentError("n_kv_heads must be positive"))
    config.d_ff > 0 || throw(ArgumentError("d_ff must be positive"))
    config.d_model % config.n_query_heads == 0 ||
        throw(ArgumentError("d_model must be divisible by n_query_heads"))
    config.n_query_heads % config.n_kv_heads == 0 ||
        throw(ArgumentError("n_query_heads must be divisible by n_kv_heads"))
    head_dim = config.d_model ÷ config.n_query_heads
    iseven(head_dim) || throw(ArgumentError("RoPE requires an even head dimension"))
    config.rope_base > 1.0f0 || throw(ArgumentError("rope_base must exceed one"))
    config.norm_eps > 0.0f0 || throw(ArgumentError("norm_eps must be positive"))
    return config
end

function tiny_config(; vocab_size::Int=264, max_seq_len::Int=32)
    return J1Config(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=16,
        n_layers=2,
        n_query_heads=4,
        n_kv_heads=2,
        d_ff=48,
        bias=false,
    )
end

j1_63m_config() = J1Config()

function param_count(config::J1Config)::Int
    validate_config(config)
    d = config.d_model
    head_dim = d ÷ config.n_query_heads
    kv_dim = config.n_kv_heads * head_dim
    bias_multiplier = config.bias ? 1 : 0

    embeddings = config.vocab_size * d
    attention = 2 * d * d + 2 * kv_dim * d +
                bias_multiplier * (2 * d + 2 * kv_dim)
    feed_forward = 3 * d * config.d_ff +
                   bias_multiplier * (2 * config.d_ff + d)
    norms = 2 * d
    final_norm = d
    return embeddings + config.n_layers * (attention + feed_forward + norms) + final_norm
end

# ============================================================================
# Numerical helpers
# ============================================================================

function add_bias(values::AbstractMatrix, bias::AbstractVector)
    isempty(bias) && return values
    length(bias) == size(values, 1) || throw(DimensionMismatch("bias width mismatch"))
    return values .+ reshape(bias, :, 1)
end

function stable_softmax(scores::AbstractVector{<:AbstractFloat})::Vector{Float32}
    maximum_score = maximum(scores)
    probabilities = exp.(Float32.(scores) .- Float32(maximum_score))
    denominator = sum(probabilities)
    isfinite(denominator) && denominator > 0.0f0 ||
        throw(ArgumentError("softmax denominator must be finite and positive"))
    probabilities ./= denominator
    return probabilities
end

function row_softmax!(scores::Matrix{Float32})
    maximums = maximum(scores; dims=2)
    scores .= exp.(scores .- maximums)
    totals = sum(scores; dims=2)
    all(total -> isfinite(total) && total > 0.0f0, totals) ||
        throw(ArgumentError("attention contains an invalid probability row"))
    scores ./= totals
    return scores
end

function causal_mask(sequence_length::Int)::Matrix{Float32}
    mask = zeros(Float32, sequence_length, sequence_length)
    @inbounds for query_position in 1:sequence_length
        for key_position in (query_position + 1):sequence_length
            mask[query_position, key_position] = -Inf32
        end
    end
    return mask
end

# ============================================================================
# RMSNorm and RoPE
# ============================================================================

struct RMSNorm
    weight::Vector{Float32}
    eps::Float32
end

RMSNorm(width::Int; eps::Float32=1.0f-5) = RMSNorm(ones(Float32, width), eps)

function (norm::RMSNorm)(inputs::AbstractMatrix{<:AbstractFloat})::Matrix{Float32}
    size(inputs, 1) == length(norm.weight) || throw(DimensionMismatch("RMSNorm width mismatch"))
    values = Float32.(inputs)
    mean_square = sum(abs2, values; dims=1) ./ Float32(size(values, 1))
    normalized = values ./ sqrt.(mean_square .+ norm.eps)
    return Matrix{Float32}(reshape(norm.weight, :, 1) .* normalized)
end

function apply_rope(
    values::AbstractMatrix{<:AbstractFloat},
    n_heads::Int,
    positions::AbstractVector{<:Integer};
    base::Float32=10_000.0f0,
)::Matrix{Float32}
    width, sequence_length = size(values)
    length(positions) == sequence_length || throw(DimensionMismatch("one RoPE position is required per token"))
    width % n_heads == 0 || throw(DimensionMismatch("head count does not divide projection width"))
    head_dim = width ÷ n_heads
    iseven(head_dim) || throw(ArgumentError("RoPE requires an even head dimension"))

    rotated = Matrix{Float32}(undef, width, sequence_length)
    source = Float32.(values)
    @inbounds for head in 1:n_heads
        head_start = (head - 1) * head_dim
        for pair_index in 0:(head_dim ÷ 2 - 1)
            first_dimension = head_start + 2 * pair_index + 1
            second_dimension = first_dimension + 1
            inverse_frequency = base ^ (-2.0f0 * Float32(pair_index) / Float32(head_dim))
            for token_index in 1:sequence_length
                angle = Float32(positions[token_index]) * inverse_frequency
                cosine = cos(angle)
                sine = sin(angle)
                first_value = source[first_dimension, token_index]
                second_value = source[second_dimension, token_index]
                rotated[first_dimension, token_index] = first_value * cosine - second_value * sine
                rotated[second_dimension, token_index] = first_value * sine + second_value * cosine
            end
        end
    end
    return rotated
end

# ============================================================================
# Grouped-query attention
# ============================================================================

struct GroupedQueryAttention
    Wq::Matrix{Float32}
    Wk::Matrix{Float32}
    Wv::Matrix{Float32}
    Wo::Matrix{Float32}
    bq::Vector{Float32}
    bk::Vector{Float32}
    bv::Vector{Float32}
    bo::Vector{Float32}
    n_query_heads::Int
    n_kv_heads::Int
    head_dim::Int
    scale::Float32
    rope_base::Float32
end

function GroupedQueryAttention(config::J1Config; rng::AbstractRNG=Random.default_rng())
    validate_config(config)
    d = config.d_model
    head_dim = d ÷ config.n_query_heads
    kv_dim = config.n_kv_heads * head_dim
    Wq = randn(rng, Float32, d, d) .* 0.02f0
    Wk = randn(rng, Float32, kv_dim, d) .* 0.02f0
    Wv = randn(rng, Float32, kv_dim, d) .* 0.02f0
    Wo = randn(rng, Float32, d, d) .* 0.02f0
    bq = config.bias ? zeros(Float32, d) : Float32[]
    bk = config.bias ? zeros(Float32, kv_dim) : Float32[]
    bv = config.bias ? zeros(Float32, kv_dim) : Float32[]
    bo = config.bias ? zeros(Float32, d) : Float32[]
    return GroupedQueryAttention(
        Wq, Wk, Wv, Wo, bq, bk, bv, bo,
        config.n_query_heads,
        config.n_kv_heads,
        head_dim,
        inv(sqrt(Float32(head_dim))),
        config.rope_base,
    )
end

function (attention::GroupedQueryAttention)(
    inputs::AbstractMatrix{<:AbstractFloat},
    positions::AbstractVector{<:Integer},
)::Matrix{Float32}
    d_model, sequence_length = size(inputs)
    d_model == attention.n_query_heads * attention.head_dim ||
        throw(DimensionMismatch("attention input width mismatch"))

    values = Float32.(inputs)
    queries = apply_rope(
        add_bias(attention.Wq * values, attention.bq),
        attention.n_query_heads,
        positions;
        base=attention.rope_base,
    )
    keys = apply_rope(
        add_bias(attention.Wk * values, attention.bk),
        attention.n_kv_heads,
        positions;
        base=attention.rope_base,
    )
    projected_values = add_bias(attention.Wv * values, attention.bv)

    output = zeros(Float32, d_model, sequence_length)
    mask = causal_mask(sequence_length)
    queries_per_kv_head = attention.n_query_heads ÷ attention.n_kv_heads

    @inbounds for query_head in 1:attention.n_query_heads
        kv_head = (query_head - 1) ÷ queries_per_kv_head + 1
        query_start = (query_head - 1) * attention.head_dim + 1
        kv_start = (kv_head - 1) * attention.head_dim + 1
        query_features = query_start:(query_start + attention.head_dim - 1)
        kv_features = kv_start:(kv_start + attention.head_dim - 1)

        query = @view queries[query_features, :]
        key = @view keys[kv_features, :]
        value = @view projected_values[kv_features, :]
        scores = Matrix{Float32}(transpose(query) * key)
        scores .*= attention.scale
        scores .+= mask
        row_softmax!(scores)
        output[query_features, :] .= value * transpose(scores)
    end

    return Matrix{Float32}(add_bias(attention.Wo * output, attention.bo))
end

# ============================================================================
# SwiGLU
# ============================================================================

silu(values) = @. values / (1.0f0 + exp(-values))

struct SwiGLU
    Wgate::Matrix{Float32}
    Wvalue::Matrix{Float32}
    Wdown::Matrix{Float32}
    bgate::Vector{Float32}
    bvalue::Vector{Float32}
    bdown::Vector{Float32}
end

function SwiGLU(config::J1Config; rng::AbstractRNG=Random.default_rng())
    d = config.d_model
    hidden = config.d_ff
    Wgate = randn(rng, Float32, hidden, d) .* 0.02f0
    Wvalue = randn(rng, Float32, hidden, d) .* sqrt(2.0f0 / Float32(d + hidden))
    Wdown = randn(rng, Float32, d, hidden) .* 0.02f0
    bgate = config.bias ? zeros(Float32, hidden) : Float32[]
    bvalue = config.bias ? zeros(Float32, hidden) : Float32[]
    bdown = config.bias ? zeros(Float32, d) : Float32[]
    return SwiGLU(Wgate, Wvalue, Wdown, bgate, bvalue, bdown)
end

function (ffn::SwiGLU)(inputs::AbstractMatrix{<:AbstractFloat})::Matrix{Float32}
    values = Float32.(inputs)
    gate = silu(add_bias(ffn.Wgate * values, ffn.bgate))
    content = add_bias(ffn.Wvalue * values, ffn.bvalue)
    return Matrix{Float32}(add_bias(ffn.Wdown * (gate .* content), ffn.bdown))
end

# ============================================================================
# Blocks and model
# ============================================================================

struct J1Block
    attention_norm::RMSNorm
    attention::GroupedQueryAttention
    ffn_norm::RMSNorm
    ffn::SwiGLU
end

function J1Block(config::J1Config; rng::AbstractRNG=Random.default_rng())
    return J1Block(
        RMSNorm(config.d_model; eps=config.norm_eps),
        GroupedQueryAttention(config; rng=rng),
        RMSNorm(config.d_model; eps=config.norm_eps),
        SwiGLU(config; rng=rng),
    )
end

function (block::J1Block)(
    inputs::AbstractMatrix{<:AbstractFloat},
    positions::AbstractVector{<:Integer},
)::Matrix{Float32}
    after_attention = Float32.(inputs) .+ block.attention(block.attention_norm(inputs), positions)
    return after_attention .+ block.ffn(block.ffn_norm(after_attention))
end

struct J1Model
    config::J1Config
    token_embedding::Matrix{Float32}
    blocks::Vector{J1Block}
    final_norm::RMSNorm
end

function J1Model(config::J1Config=j1_63m_config(); rng::AbstractRNG=Random.Xoshiro(42))
    validate_config(config)
    embedding = randn(rng, Float32, config.vocab_size, config.d_model) .* 0.02f0
    blocks = [J1Block(config; rng=rng) for _ in 1:config.n_layers]
    return J1Model(config, embedding, blocks, RMSNorm(config.d_model; eps=config.norm_eps))
end

function validate_tokens(model::J1Model, tokens::AbstractVector{<:Integer})
    isempty(tokens) && throw(ArgumentError("at least one token is required"))
    length(tokens) <= model.config.max_seq_len || throw(ArgumentError("sequence exceeds cache capacity"))
    all(token -> 0 <= token < model.config.vocab_size, tokens) ||
        throw(ArgumentError("token ID is outside the vocabulary"))
    return tokens
end

function forward_all(model::J1Model, tokens::AbstractVector{<:Integer})::Matrix{Float32}
    validate_tokens(model, tokens)
    positions = collect(0:(length(tokens) - 1))
    hidden = permutedims(model.token_embedding[Int.(tokens) .+ 1, :])
    for block in model.blocks
        hidden = block(hidden, positions)
    end
    hidden = model.final_norm(hidden)
    return Matrix{Float32}(model.token_embedding * hidden)
end

(model::J1Model)(tokens::AbstractVector{<:Integer}) = Vector{Float32}(forward_all(model, tokens)[:, end])

function cross_entropy_loss(
    logits::AbstractMatrix{<:AbstractFloat},
    targets::AbstractVector{<:Integer},
)::Float32
    size(logits, 2) == length(targets) || throw(DimensionMismatch("target count mismatch"))
    isempty(targets) && throw(ArgumentError("targets cannot be empty"))
    vocabulary = size(logits, 1)
    all(target -> 0 <= target < vocabulary, targets) || throw(ArgumentError("target outside vocabulary"))

    total = 0.0f0
    for position in eachindex(targets)
        column = @view logits[:, position]
        maximum_logit = maximum(column)
        log_partition = maximum_logit + log(sum(exp.(column .- maximum_logit)))
        total += Float32(log_partition - column[Int(targets[position]) + 1])
    end
    return total / Float32(length(targets))
end

cross_entropy_loss(model::J1Model, inputs, targets) =
    cross_entropy_loss(forward_all(model, inputs), targets)

# ============================================================================
# Incremental KV cache
# ============================================================================

mutable struct LayerKVCache
    keys::Matrix{Float32}
    values::Matrix{Float32}
end

mutable struct KVCache
    layers::Vector{LayerKVCache}
    length::Int
    max_seq_len::Int
end

function init_cache(model::J1Model)::KVCache
    head_dim = model.config.d_model ÷ model.config.n_query_heads
    kv_dim = model.config.n_kv_heads * head_dim
    layers = [
        LayerKVCache(
            zeros(Float32, kv_dim, model.config.max_seq_len),
            zeros(Float32, kv_dim, model.config.max_seq_len),
        ) for _ in model.blocks
    ]
    return KVCache(layers, 0, model.config.max_seq_len)
end

function attention_step(
    attention::GroupedQueryAttention,
    inputs::AbstractMatrix{<:AbstractFloat},
    cache::LayerKVCache,
    position::Int,
)::Matrix{Float32}
    size(inputs, 2) == 1 || throw(DimensionMismatch("incremental attention accepts one token"))
    values = Float32.(inputs)
    queries = apply_rope(
        add_bias(attention.Wq * values, attention.bq),
        attention.n_query_heads,
        [position];
        base=attention.rope_base,
    )
    key = apply_rope(
        add_bias(attention.Wk * values, attention.bk),
        attention.n_kv_heads,
        [position];
        base=attention.rope_base,
    )
    value = add_bias(attention.Wv * values, attention.bv)
    cache.keys[:, position + 1] .= key[:, 1]
    cache.values[:, position + 1] .= value[:, 1]

    context_length = position + 1
    d_model = attention.n_query_heads * attention.head_dim
    output = zeros(Float32, d_model, 1)
    queries_per_kv_head = attention.n_query_heads ÷ attention.n_kv_heads

    @inbounds for query_head in 1:attention.n_query_heads
        kv_head = (query_head - 1) ÷ queries_per_kv_head + 1
        query_start = (query_head - 1) * attention.head_dim + 1
        kv_start = (kv_head - 1) * attention.head_dim + 1
        query_features = query_start:(query_start + attention.head_dim - 1)
        kv_features = kv_start:(kv_start + attention.head_dim - 1)

        query = @view queries[query_features, 1]
        keys = @view cache.keys[kv_features, 1:context_length]
        cached_values = @view cache.values[kv_features, 1:context_length]
        scores = vec(transpose(query) * keys) .* attention.scale
        probabilities = stable_softmax(scores)
        output[query_features, 1] .= cached_values * probabilities
    end

    return Matrix{Float32}(add_bias(attention.Wo * output, attention.bo))
end

function block_step(
    block::J1Block,
    inputs::AbstractMatrix{<:AbstractFloat},
    cache::LayerKVCache,
    position::Int,
)::Matrix{Float32}
    after_attention = Float32.(inputs) .+
        attention_step(block.attention, block.attention_norm(inputs), cache, position)
    return after_attention .+ block.ffn(block.ffn_norm(after_attention))
end

function decode_step!(model::J1Model, token::Integer, cache::KVCache)::Vector{Float32}
    0 <= token < model.config.vocab_size || throw(ArgumentError("token ID is outside the vocabulary"))
    cache.length < cache.max_seq_len || throw(ArgumentError("KV cache is full"))
    length(cache.layers) == length(model.blocks) || throw(DimensionMismatch("cache layer count mismatch"))

    position = cache.length
    hidden = reshape(copy(@view model.token_embedding[Int(token) + 1, :]), :, 1)
    for layer_index in eachindex(model.blocks)
        hidden = block_step(model.blocks[layer_index], hidden, cache.layers[layer_index], position)
    end
    cache.length += 1
    hidden = model.final_norm(hidden)
    return Vector{Float32}(model.token_embedding * hidden[:, 1])
end

function prefill(model::J1Model, tokens::AbstractVector{<:Integer})
    validate_tokens(model, tokens)
    cache = init_cache(model)
    logits = Float32[]
    for token in tokens
        logits = decode_step!(model, token, cache)
    end
    return logits, cache
end

end # module
