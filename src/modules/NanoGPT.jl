# ============================================================================
# NanoGPT.jl — Pure Julia transformer research baseline for IngExuity
# ============================================================================
module NanoGPT

using Random
using Statistics
import StatsBase
using JSON

export TransformerModel, TransformerConfig, SimpleTokenizer, GPT2Tokenizer
export generate, chat_local, forward_all, cross_entropy_loss, param_count
export load_model, load_tokenizer, load_local_model, load_pretrained, is_loaded
export encode, decode, train

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef mutable struct TransformerConfig
    vocab_size::Int = 50257
    max_seq_len::Int = 1024
    n_embed::Int = 768
    n_heads::Int = 12
    n_layers::Int = 12
    n_hidden::Int = 3072
    dropout::Float32 = 0.0f0
    bias::Bool = true
end

function validate_config(cfg::TransformerConfig)
    cfg.vocab_size > 1 || throw(ArgumentError("vocab_size must be greater than one"))
    cfg.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    cfg.n_embed > 0 || throw(ArgumentError("n_embed must be positive"))
    cfg.n_heads > 0 || throw(ArgumentError("n_heads must be positive"))
    cfg.n_layers > 0 || throw(ArgumentError("n_layers must be positive"))
    cfg.n_hidden > 0 || throw(ArgumentError("n_hidden must be positive"))
    cfg.n_embed % cfg.n_heads == 0 ||
        throw(ArgumentError("n_embed must be divisible by n_heads"))
    0.0f0 <= cfg.dropout < 1.0f0 ||
        throw(ArgumentError("dropout must be in [0, 1)"))
    return cfg
end

"""Return the number of independently stored model parameters.

The token embedding and language-model head are tied and therefore counted once.
"""
function param_count(cfg::TransformerConfig)::Int
    validate_config(cfg)
    d = cfg.n_embed
    h = cfg.n_hidden
    bias_count = cfg.bias ? 1 : 0

    embeddings = cfg.vocab_size * d + cfg.max_seq_len * d
    attention_per_layer = 4 * d * d + bias_count * 4 * d
    ffn_per_layer = 2 * d * h + bias_count * (h + d)
    norms_per_layer = 2 * (d + bias_count * d)
    final_norm = d + bias_count * d

    return embeddings + cfg.n_layers * (
        attention_per_layer + ffn_per_layer + norms_per_layer
    ) + final_norm
end

# ============================================================================
# Numerics
# ============================================================================

gelu(x) = @. 0.5f0 * x * (1.0f0 + tanh(sqrt(2.0f0 / Float32(pi)) * (x + 0.044715f0 * x^3)))

function softmax!(out::AbstractVector, x::AbstractVector)
    axes(out) == axes(x) || throw(DimensionMismatch("softmax output must match input"))
    max_x = maximum(x)
    out .= exp.(x .- max_x)
    denominator = sum(out)
    isfinite(denominator) && denominator > zero(denominator) ||
        throw(ArgumentError("softmax denominator is not finite and positive"))
    out ./= denominator
    return out
end

softmax(x::AbstractVector) = softmax!(similar(x), x)

function row_softmax!(scores::AbstractMatrix)
    row_max = maximum(scores; dims=2)
    scores .= exp.(scores .- row_max)
    row_sums = sum(scores; dims=2)
    all(isfinite, row_sums) || throw(ArgumentError("attention softmax produced non-finite rows"))
    all(value -> value > zero(eltype(row_sums)), row_sums) ||
        throw(ArgumentError("attention softmax produced an empty row"))
    scores ./= row_sums
    return scores
end

function add_bias(x::AbstractMatrix, bias::AbstractVector)
    isempty(bias) && return x
    length(bias) == size(x, 1) || throw(DimensionMismatch("bias does not match projection"))
    return x .+ reshape(bias, :, 1)
end

# ============================================================================
# Layer normalization
# ============================================================================

struct LayerNorm
    weight::Vector{Float32}
    bias::Vector{Float32}
    eps::Float32
    n::Int
end

function LayerNorm(n::Int; bias::Bool=true, eps::Float32=1f-5)
    n > 0 || throw(ArgumentError("LayerNorm width must be positive"))
    return LayerNorm(
        ones(Float32, n),
        bias ? zeros(Float32, n) : Float32[],
        eps,
        n,
    )
end

function (ln::LayerNorm)(x::AbstractMatrix{<:AbstractFloat})::Matrix{Float32}
    size(x, 1) == ln.n ||
        throw(DimensionMismatch("LayerNorm expected $(ln.n) features, got $(size(x, 1))"))

    xf = Float32.(x)
    token_means = Statistics.mean(xf; dims=1)
    centered = xf .- token_means
    token_variances = sum(abs2, centered; dims=1) ./ Float32(size(xf, 1))
    normalized = centered ./ sqrt.(token_variances .+ ln.eps)
    scaled = reshape(ln.weight, :, 1) .* normalized
    return isempty(ln.bias) ? Matrix{Float32}(scaled) :
           Matrix{Float32}(scaled .+ reshape(ln.bias, :, 1))
end

# ============================================================================
# Causal self-attention
# ============================================================================

struct CausalAttention
    Wq::Matrix{Float32}
    Wk::Matrix{Float32}
    Wv::Matrix{Float32}
    Wo::Matrix{Float32}
    bq::Vector{Float32}
    bk::Vector{Float32}
    bv::Vector{Float32}
    bo::Vector{Float32}
    n_heads::Int
    head_dim::Int
    scale::Float32
end

function CausalAttention(
    n_embed::Int,
    n_heads::Int;
    rng::AbstractRNG=Random.default_rng(),
    bias::Bool=true,
)
    n_embed % n_heads == 0 || throw(ArgumentError("n_embed must be divisible by n_heads"))
    head_dim = n_embed ÷ n_heads
    scale = inv(sqrt(Float32(head_dim)))

    Wq = randn(rng, Float32, n_embed, n_embed) .* 0.02f0
    Wk = randn(rng, Float32, n_embed, n_embed) .* 0.02f0
    Wv = randn(rng, Float32, n_embed, n_embed) .* 0.02f0
    Wo = randn(rng, Float32, n_embed, n_embed) .* 0.02f0
    bq = bias ? zeros(Float32, n_embed) : Float32[]
    bk = bias ? zeros(Float32, n_embed) : Float32[]
    bv = bias ? zeros(Float32, n_embed) : Float32[]
    bo = bias ? zeros(Float32, n_embed) : Float32[]

    return CausalAttention(Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads, head_dim, scale)
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

function (attn::CausalAttention)(x::AbstractMatrix{<:AbstractFloat})::Matrix{Float32}
    n_embed, sequence_length = size(x)
    expected_width = attn.n_heads * attn.head_dim
    n_embed == expected_width ||
        throw(DimensionMismatch("attention expected $expected_width features, got $n_embed"))

    xf = Float32.(x)
    queries = add_bias(transpose(attn.Wq) * xf, attn.bq)
    keys = add_bias(transpose(attn.Wk) * xf, attn.bk)
    values = add_bias(transpose(attn.Wv) * xf, attn.bv)

    mask = causal_mask(sequence_length)
    output = zeros(Float32, n_embed, sequence_length)

    # The only explicit model loop is over heads. Token interactions and value
    # aggregation use matrix multiplication instead of scalar triple loops.
    @inbounds for head in 1:attn.n_heads
        first_feature = (head - 1) * attn.head_dim + 1
        features = first_feature:(first_feature + attn.head_dim - 1)
        q = @view queries[features, :]
        k = @view keys[features, :]
        v = @view values[features, :]

        scores = Matrix{Float32}(transpose(q) * k)
        scores .*= attn.scale
        scores .+= mask
        row_softmax!(scores)

        output[features, :] .= v * transpose(scores)
    end

    return Matrix{Float32}(add_bias(transpose(attn.Wo) * output, attn.bo))
end

# ============================================================================
# Feed-forward network
# ============================================================================

struct FeedForward
    W1::Matrix{Float32}
    b1::Vector{Float32}
    W2::Matrix{Float32}
    b2::Vector{Float32}
    activation::Function
end

function FeedForward(
    n_embed::Int,
    n_hidden::Int;
    rng::AbstractRNG=Random.default_rng(),
    bias::Bool=true,
)
    W1 = randn(rng, Float32, n_hidden, n_embed) .* 0.02f0
    b1 = bias ? zeros(Float32, n_hidden) : Float32[]
    W2 = randn(rng, Float32, n_embed, n_hidden) .* 0.02f0
    b2 = bias ? zeros(Float32, n_embed) : Float32[]
    return FeedForward(W1, b1, W2, b2, gelu)
end

function (ff::FeedForward)(x::AbstractMatrix{<:AbstractFloat})::Matrix{Float32}
    hidden = ff.activation(add_bias(ff.W1 * Float32.(x), ff.b1))
    return Matrix{Float32}(add_bias(ff.W2 * hidden, ff.b2))
end

# ============================================================================
# Transformer block
# ============================================================================

struct TransformerBlock
    attn::CausalAttention
    ff::FeedForward
    ln1::LayerNorm
    ln2::LayerNorm
end

function TransformerBlock(
    n_embed::Int,
    n_heads::Int,
    n_hidden::Int;
    rng::AbstractRNG=Random.default_rng(),
    bias::Bool=true,
)
    return TransformerBlock(
        CausalAttention(n_embed, n_heads; rng=rng, bias=bias),
        FeedForward(n_embed, n_hidden; rng=rng, bias=bias),
        LayerNorm(n_embed; bias=bias),
        LayerNorm(n_embed; bias=bias),
    )
end

function (block::TransformerBlock)(x::AbstractMatrix{<:AbstractFloat})::Matrix{Float32}
    after_attention = Float32.(x) .+ block.attn(block.ln1(x))
    return after_attention .+ block.ff(block.ln2(after_attention))
end

# ============================================================================
# Full transformer model
# ============================================================================

struct TransformerModel
    config::TransformerConfig
    token_embedding::Matrix{Float32}
    position_embedding::Matrix{Float32}
    blocks::Vector{TransformerBlock}
    final_ln::LayerNorm
    lm_head::Matrix{Float32}
end

function TransformerModel(
    config::TransformerConfig;
    rng::AbstractRNG=Random.Xoshiro(42),
)
    validate_config(config)
    n = config.n_embed

    token_embedding = randn(rng, Float32, config.vocab_size, n) .* 0.02f0
    position_embedding = randn(rng, Float32, config.max_seq_len, n) .* 0.02f0
    blocks = [
        TransformerBlock(
            n,
            config.n_heads,
            config.n_hidden;
            rng=rng,
            bias=config.bias,
        ) for _ in 1:config.n_layers
    ]
    final_ln = LayerNorm(n; bias=config.bias)

    # Weight tying is an alias, not a second independently stored matrix.
    lm_head = token_embedding
    return TransformerModel(
        config,
        token_embedding,
        position_embedding,
        blocks,
        final_ln,
        lm_head,
    )
end

function validate_tokens(model::TransformerModel, tokens::AbstractVector{<:Integer})
    isempty(tokens) && throw(ArgumentError("at least one token is required"))
    length(tokens) <= model.config.max_seq_len ||
        throw(ArgumentError("sequence exceeds max_seq_len=$(model.config.max_seq_len)"))
    all(token -> 0 <= token < model.config.vocab_size, tokens) ||
        throw(ArgumentError("token IDs must be in 0:$(model.config.vocab_size - 1)"))
    return tokens
end

"""Return logits for every input position as `vocab_size × sequence_length`."""
function forward_all(
    model::TransformerModel,
    tokens::AbstractVector{<:Integer},
)::Matrix{Float32}
    validate_tokens(model, tokens)
    sequence_length = length(tokens)
    embedding_rows = Int.(tokens) .+ 1

    x = permutedims(model.token_embedding[embedding_rows, :])
    x .+= permutedims(model.position_embedding[1:sequence_length, :])

    for block in model.blocks
        x = block(x)
    end

    x = model.final_ln(x)
    return Matrix{Float32}(model.lm_head * x)
end

"""Return next-token logits for the final input position."""
function (model::TransformerModel)(tokens::AbstractVector{<:Integer})::Vector{Float32}
    logits = forward_all(model, tokens)
    return Vector{Float32}(logits[:, end])
end

"""Evaluate a `sequence_length × batch_size` token matrix."""
function (model::TransformerModel)(tokens::AbstractMatrix{<:Integer})::Array{Float32,3}
    sequence_length, batch_size = size(tokens)
    sequence_length > 0 && batch_size > 0 || throw(ArgumentError("token batch cannot be empty"))
    outputs = Array{Float32}(undef, model.config.vocab_size, sequence_length, batch_size)
    for batch_index in 1:batch_size
        outputs[:, :, batch_index] .= forward_all(model, @view(tokens[:, batch_index]))
    end
    return outputs
end

function cross_entropy_loss(
    logits::AbstractMatrix{<:AbstractFloat},
    targets::AbstractVector{<:Integer},
)::Float32
    size(logits, 2) == length(targets) ||
        throw(DimensionMismatch("one target is required for each logit column"))
    isempty(targets) && throw(ArgumentError("targets cannot be empty"))
    vocab_size = size(logits, 1)
    all(target -> 0 <= target < vocab_size, targets) ||
        throw(ArgumentError("target token is outside the vocabulary"))

    total = 0.0f0
    for position in eachindex(targets)
        column = @view logits[:, position]
        max_logit = maximum(column)
        log_partition = max_logit + log(sum(exp.(column .- max_logit)))
        total += Float32(log_partition - column[Int(targets[position]) + 1])
    end
    return total / Float32(length(targets))
end

function cross_entropy_loss(
    model::TransformerModel,
    inputs::AbstractVector{<:Integer},
    targets::AbstractVector{<:Integer},
)::Float32
    return cross_entropy_loss(forward_all(model, inputs), targets)
end

function generate(
    model::TransformerModel,
    tokenizer,
    seed::String;
    max_new_tokens::Int=100,
    temperature::Float32=1.0f0,
    top_k::Int=40,
    rng::AbstractRNG=Random.Xoshiro(42),
)::String
    max_new_tokens >= 0 || throw(ArgumentError("max_new_tokens cannot be negative"))
    temperature > 0.0f0 || throw(ArgumentError("temperature must be positive"))
    top_k >= 0 || throw(ArgumentError("top_k cannot be negative"))

    tokens = encode(tokenizer, seed)
    isempty(tokens) && throw(ArgumentError("seed must encode to at least one token"))

    for _ in 1:max_new_tokens
        context_start = max(1, length(tokens) - model.config.max_seq_len + 1)
        context = @view tokens[context_start:end]
        logits = model(context) ./ temperature

        if 0 < top_k < length(logits)
            keep = partialsortperm(logits, 1:top_k; rev=true)
            filtered = fill(-Inf32, length(logits))
            filtered[keep] .= logits[keep]
            logits = filtered
        end

        probabilities = softmax(logits)
        sampled_index = StatsBase.sample(rng, eachindex(probabilities), StatsBase.Weights(probabilities))
        next_token = Int(sampled_index) - 1
        push!(tokens, next_token)

        next_token == tokenizer.eot_token && break
    end

    return decode(tokenizer, tokens)
end

function Base.show(io::IO, model::TransformerModel)
    cfg = model.config
    params = param_count(cfg)
    println(io, "TransformerModel($(cfg.n_layers) layers, $(cfg.n_heads) heads, $(cfg.n_embed) embed)")
    println(io, "  → ~$(round(params / 1e6; digits=1))M stored params, $(cfg.vocab_size) vocab, $(cfg.max_seq_len) ctx")
end

# ============================================================================
# Simple byte-level BPE tokenizer
# ============================================================================

struct SimpleTokenizer
    vocab::Dict{Vector{Int},Int}
    reverse_vocab::Vector{Vector{Int}}
    eot_token::Int
    eot_byte::Vector{Int}
end

function SimpleTokenizer(vocab_size::Int=5000; rng::AbstractRNG=Random.Xoshiro(42))
    vocab_size >= 257 || throw(ArgumentError("vocab_size must cover 256 bytes plus EOT"))
    _ = rng # Reserved for deterministic tokenizer-training extensions.

    vocab = Dict{Vector{Int},Int}()
    reverse_vocab = Vector{Vector{Int}}()
    for byte in 0:255
        vocab[[byte]] = byte
        push!(reverse_vocab, [byte])
    end

    eot_token = vocab_size - 1
    return SimpleTokenizer(vocab, reverse_vocab, eot_token, [256])
end

function get_pair_counts(sequences::Vector{Dict{Vector{Int},Int}})
    pairs = Dict{Tuple{Int,Int},Int}()
    for sequence_counts in sequences
        for (tokens, count) in sequence_counts
            for index in 1:(length(tokens) - 1)
                pair = (tokens[index], tokens[index + 1])
                pairs[pair] = get(pairs, pair, 0) + count
            end
        end
    end
    return pairs
end

function merge_pair!(
    sequences::Vector{Dict{Vector{Int},Int}},
    pair::Tuple{Int,Int},
    new_id::Int,
)
    for sequence_counts in sequences
        replacements = Dict{Vector{Int},Int}()
        for (tokens, count) in sequence_counts
            merged = Int[]
            index = 1
            while index <= length(tokens)
                if index < length(tokens) &&
                   tokens[index] == pair[1] && tokens[index + 1] == pair[2]
                    push!(merged, new_id)
                    index += 2
                else
                    push!(merged, tokens[index])
                    index += 1
                end
            end
            replacements[merged] = get(replacements, merged, 0) + count
        end
        empty!(sequence_counts)
        merge!(sequence_counts, replacements)
    end
    return sequences
end

function train(
    texts::Vector{String},
    vocab_size::Int=5000;
    rng::AbstractRNG=Random.Xoshiro(42),
)
    tokenizer = SimpleTokenizer(vocab_size; rng=rng)
    sequences = [Dict{Vector{Int},Int}() for _ in eachindex(texts)]
    for (index, text) in pairs(texts)
        bytes = Int.(collect(codeunits(text)))
        isempty(bytes) || (sequences[index][bytes] = 1)
    end

    current_id = 256
    final_merge_id = tokenizer.eot_token - 1
    while current_id <= final_merge_id
        pair_counts = get_pair_counts(sequences)
        isempty(pair_counts) && break
        best_pair = argmax(pair_counts)
        pair_counts[best_pair] >= 2 || break

        tokenizer.vocab[[best_pair[1], best_pair[2]]] = current_id
        push!(tokenizer.reverse_vocab, [best_pair[1], best_pair[2]])
        merge_pair!(sequences, best_pair, current_id)
        current_id += 1
    end
    return tokenizer
end

function encode(tokenizer::SimpleTokenizer, text::String)::Vector{Int}
    raw_tokens = Int.(collect(codeunits(text)))
    result = Int[]
    index = 1
    while index <= length(raw_tokens)
        longest = [raw_tokens[index]]
        for final_index in (index + 1):length(raw_tokens)
            candidate = raw_tokens[index:final_index]
            haskey(tokenizer.vocab, candidate) || break
            longest = candidate
        end
        push!(result, tokenizer.vocab[longest])
        index += length(longest)
    end
    return result
end

function expand_token!(bytes::Vector{UInt8}, tokenizer::SimpleTokenizer, token::Int)
    token == tokenizer.eot_token && return bytes
    0 <= token < length(tokenizer.reverse_vocab) || return bytes
    expansion = tokenizer.reverse_vocab[token + 1]
    if length(expansion) == 1 && 0 <= expansion[1] <= 255
        push!(bytes, UInt8(expansion[1]))
    else
        for child in expansion
            expand_token!(bytes, tokenizer, child)
        end
    end
    return bytes
end

function decode(tokenizer::SimpleTokenizer, tokens::Vector{Int})::String
    bytes = UInt8[]
    for token in tokens
        token == tokenizer.eot_token && break
        expand_token!(bytes, tokenizer, token)
    end
    return String(bytes)
end

# ============================================================================
# GPT-2 tokenizer
# ============================================================================

struct GPT2Tokenizer
    vocab::Dict{String,Int}
    reverse_vocab::Dict{Int,String}
    merges::Vector{Tuple{String,String}}
    byte_encoder::Dict{UInt8,String}
    byte_decoder::Dict{String,UInt8}
    eot_token::Int
end

function gpt2_byte_encoder()::Dict{UInt8,String}
    encoder = Dict{UInt8,String}()
    visible = UInt8[]
    append!(visible, UInt8.(0x21:0x7e))
    append!(visible, UInt8.(0xa1:0xac))
    append!(visible, UInt8.(0xae:0xff))
    next_codepoint = 0
    for byte in UInt8(0):UInt8(255)
        if byte in visible
            encoder[byte] = string(Char(byte))
        else
            encoder[byte] = string(Char(256 + next_codepoint))
            next_codepoint += 1
        end
    end
    return encoder
end

gpt2_byte_decoder(encoder::Dict{UInt8,String}) = Dict(value => key for (key, value) in encoder)

function GPT2Tokenizer(vocab_path::String, merges_path::String)
    parsed_vocab = JSON.parsefile(vocab_path)
    vocab = Dict{String,Int}(String(token) => Int(identifier) for (token, identifier) in parsed_vocab)
    reverse_vocab = Dict{Int,String}(identifier => token for (token, identifier) in vocab)

    merges = Tuple{String,String}[]
    for line in eachline(merges_path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        pieces = split(strip(line))
        length(pieces) == 2 && push!(merges, (String(pieces[1]), String(pieces[2])))
    end

    encoder = gpt2_byte_encoder()
    decoder = gpt2_byte_decoder(encoder)
    eot = get(vocab, "<|endoftext|>", 50256)
    return GPT2Tokenizer(vocab, reverse_vocab, merges, encoder, decoder, eot)
end

function bpe(tokenizer::GPT2Tokenizer, word::String)::Vector{String}
    symbols = [string(character) for character in word]
    length(symbols) <= 1 && return symbols
    merge_ranks = Dict(pair => rank for (rank, pair) in enumerate(tokenizer.merges))

    while length(symbols) > 1
        best_index = 0
        best_rank = typemax(Int)
        for index in 1:(length(symbols) - 1)
            rank = get(merge_ranks, (symbols[index], symbols[index + 1]), typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_index = index
            end
        end
        best_index == 0 && break
        merged = symbols[best_index] * symbols[best_index + 1]
        symbols = vcat(symbols[1:(best_index - 1)], [merged], symbols[(best_index + 2):end])
    end
    return symbols
end

function encode(tokenizer::GPT2Tokenizer, text::String)::Vector{Int}
    encoded_symbols = [tokenizer.byte_encoder[byte] for byte in codeunits(text)]
    token_ids = Int[]
    for token in bpe(tokenizer, join(encoded_symbols))
        identifier = get(tokenizer.vocab, token, nothing)
        identifier === nothing || push!(token_ids, identifier)
    end
    return token_ids
end

function encode_with_eot(tokenizer::GPT2Tokenizer, text::String)::Vector{Int}
    tokens = encode(tokenizer, text)
    push!(tokens, tokenizer.eot_token)
    return tokens
end

function decode(tokenizer::GPT2Tokenizer, token_ids::Vector{Int})::String
    encoded = join(
        get(tokenizer.reverse_vocab, identifier, "")
        for identifier in token_ids if identifier != tokenizer.eot_token
    )
    bytes = UInt8[]
    for character in encoded
        byte = get(tokenizer.byte_decoder, string(character), nothing)
        byte === nothing || push!(bytes, byte)
    end
    return String(bytes)
end

# ============================================================================
# Model loading and chat
# ============================================================================

const MODEL = Ref{Union{TransformerModel,Nothing}}(nothing)
const TOKENIZER = Ref{Union{SimpleTokenizer,GPT2Tokenizer,Nothing}}(nothing)
const WEIGHTS_PATH = Ref{Union{String,Nothing}}(nothing)

function load_model(config::TransformerConfig=TransformerConfig())
    model = TransformerModel(config)
    MODEL[] = model
    return model
end

function load_tokenizer(vocab_size::Int=5000)
    tokenizer = SimpleTokenizer(vocab_size)
    TOKENIZER[] = tokenizer
    return tokenizer
end

is_loaded()::Bool = MODEL[] !== nothing && TOKENIZER[] !== nothing

"""Load GPT-2 weights exported by `scripts/export_gpt2_weights.py`."""
function load_pretrained(
    weights_path::String;
    vocab_path::Union{String,Nothing}=nothing,
    merges_path::Union{String,Nothing}=nothing,
)
    @info "Loading pretrained GPT-2 weights from: $weights_path"
    if vocab_path !== nothing && merges_path !== nothing
        tokenizer = GPT2Tokenizer(vocab_path, merges_path)
        TOKENIZER[] = tokenizer
        @info "GPT-2 tokenizer loaded" vocab=length(tokenizer.vocab)
    end

    data = read(weights_path)
    null_position = findfirst(isequal(0x00), data)
    null_position === nothing && error("Invalid weight file: no null terminator found")
    tensor_info = JSON.parse(String(data[1:(null_position - 1)]))
    raw_floats = reinterpret(Float32, data[(null_position + 1):end])
    position = Ref(1)

    function read_tensor(name::String)::Array{Float32}
        info = tensor_info[name]
        shape = Int.(info["shape"])
        count = prod(shape)
        final_position = position[] + count - 1
        final_position <= length(raw_floats) || error("Weight file ended while reading $name")
        tensor = reshape(copy(raw_floats[position[]:final_position]), Tuple(reverse(shape)))
        position[] = final_position + 1
        return Array(tensor)
    end

    cfg = TransformerConfig()
    n = cfg.n_embed
    token_embedding = Matrix{Float32}(read_tensor("wte.weight"))
    position_embedding = Matrix{Float32}(read_tensor("wpe.weight"))

    blocks = TransformerBlock[]
    for layer_index in 0:(cfg.n_layers - 1)
        ln1_weight = vec(read_tensor("h.$layer_index.ln_1.weight"))
        ln1_bias = vec(read_tensor("h.$layer_index.ln_1.bias"))
        ln2_weight = vec(read_tensor("h.$layer_index.ln_2.weight"))
        ln2_bias = vec(read_tensor("h.$layer_index.ln_2.bias"))

        combined_attention_weight = read_tensor("h.$layer_index.attn.c_attn.weight")
        combined_attention_bias = vec(read_tensor("h.$layer_index.attn.c_attn.bias"))
        output_weight = read_tensor("h.$layer_index.attn.c_proj.weight")
        output_bias = vec(read_tensor("h.$layer_index.attn.c_proj.bias"))

        Wq = Matrix{Float32}(transpose(combined_attention_weight[1:n, :]))
        Wk = Matrix{Float32}(transpose(combined_attention_weight[(n + 1):(2n), :]))
        Wv = Matrix{Float32}(transpose(combined_attention_weight[(2n + 1):(3n), :]))
        bq = Vector{Float32}(combined_attention_bias[1:n])
        bk = Vector{Float32}(combined_attention_bias[(n + 1):(2n)])
        bv = Vector{Float32}(combined_attention_bias[(2n + 1):(3n)])
        Wo = Matrix{Float32}(transpose(output_weight))
        bo = Vector{Float32}(output_bias)

        W1 = Matrix{Float32}(read_tensor("h.$layer_index.mlp.c_fc.weight"))
        b1 = Vector{Float32}(vec(read_tensor("h.$layer_index.mlp.c_fc.bias")))
        W2 = Matrix{Float32}(read_tensor("h.$layer_index.mlp.c_proj.weight"))
        b2 = Vector{Float32}(vec(read_tensor("h.$layer_index.mlp.c_proj.bias")))

        head_dim = n ÷ cfg.n_heads
        attention = CausalAttention(
            Wq, Wk, Wv, Wo, bq, bk, bv, bo,
            cfg.n_heads, head_dim, inv(sqrt(Float32(head_dim))),
        )
        feed_forward = FeedForward(W1, b1, W2, b2, gelu)
        push!(
            blocks,
            TransformerBlock(
                attention,
                feed_forward,
                LayerNorm(Vector{Float32}(ln1_weight), Vector{Float32}(ln1_bias), 1f-5, n),
                LayerNorm(Vector{Float32}(ln2_weight), Vector{Float32}(ln2_bias), 1f-5, n),
            ),
        )
    end

    final_weight = Vector{Float32}(vec(read_tensor("ln_f.weight")))
    final_bias = Vector{Float32}(vec(read_tensor("ln_f.bias")))
    final_norm = LayerNorm(final_weight, final_bias, 1f-5, n)
    model = TransformerModel(cfg, token_embedding, position_embedding, blocks, final_norm, token_embedding)
    WEIGHTS_PATH[] = weights_path
    return model
end

function load_local_model(
    weights_path::String;
    vocab_path::Union{String,Nothing}=nothing,
    merges_path::Union{String,Nothing}=nothing,
)
    model = load_pretrained(weights_path; vocab_path=vocab_path, merges_path=merges_path)
    MODEL[] = model
    return model
end

function chat_local(input::String; session_id::Int64=0)::Dict{String,Any}
    if !is_loaded()
        return Dict("error" => "Model not loaded. Call load_local_model() first.", "text" => "")
    end

    response = generate(
        MODEL[],
        TOKENIZER[],
        input;
        rng=Random.Xoshiro(UInt64(time_ns())),
    )
    return Dict(
        "text" => response,
        "session_id" => session_id,
        "model" => "NanoGPT (Pure Julia)",
    )
end

chat(input::String; session_id::Int64=0) = chat_local(input; session_id=session_id)

end # module
