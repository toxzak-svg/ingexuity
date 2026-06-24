# ============================================================================
# NanoGPT.jl — Pure Julia NanoGPT for IngExuity
# Drop-in replacement for Flux-based NanoGPT
# ============================================================================
module NanoGPT

using Random
using Statistics
import StatsBase
using JSON

export TransformerModel, TransformerConfig, SimpleTokenizer, generate, chat_local

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

function param_count(cfg::TransformerConfig)::Int
    vocab = cfg.vocab_size * cfg.n_embed
    pos = cfg.max_seq_len * cfg.n_embed
    attn = cfg.n_layers * (
        3 * cfg.n_embed * cfg.n_embed +
        cfg.n_embed * cfg.n_embed +
        2 * cfg.n_embed
    )
    ffn = cfg.n_layers * (
        2 * cfg.n_embed * cfg.n_hidden +
        2 * cfg.n_embed
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
    m = Statistics.mean(x, dims=2)
    v = Statistics.var(x, dims=2, corrected=true)
    x_norm = (x .- m) ./ sqrt.(v .+ ln.eps)
    return ln.weight .* x_norm .+ ln.bias
end

# ============================================================================
# Causal Self-Attention
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

function CausalAttention(n_embed::Int, n_heads::Int)
    @assert n_embed % n_heads == 0
    head_dim = n_embed ÷ n_heads
    scale = Float32(1.0 / sqrt(head_dim))

    rng = Random.Xoshiro(42)
    Wq = randn(rng, Float32, n_embed, n_embed) * 0.02f0
    Wk = randn(rng, Float32, n_embed, n_embed) * 0.02f0
    Wv = randn(rng, Float32, n_embed, n_embed) * 0.02f0
    Wo = randn(rng, Float32, n_embed, n_embed) * 0.02f0
    bq = zeros(Float32, n_embed)
    bk = zeros(Float32, n_embed)
    bv = zeros(Float32, n_embed)
    bo = zeros(Float32, n_embed)

    return CausalAttention(Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads, head_dim, scale)
end

function (attn::CausalAttention)(x::AbstractMatrix{Float32})::AbstractMatrix{Float32}
    n_embed, T = size(x)
    n_heads = attn.n_heads
    head_dim = attn.head_dim

    Q = attn.Wq' * x .+ attn.bq
    K = attn.Wk' * x .+ attn.bk
    V = attn.Wv' * x .+ attn.bv

    Q_seq = reshape(Q, n_heads, head_dim, T)
    K_seq = reshape(K, n_heads, head_dim, T)
    V_seq = reshape(V, n_heads, head_dim, T)

    attn_scores = zeros(Float32, T, T, n_heads)
    for h in 1:n_heads
        for i in 1:T
            for j in 1:T
                if j <= i
                    attn_scores[i, j, h] = attn.scale * sum(Q_seq[h, :, i] .* K_seq[h, :, j])
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

    return attn.Wo' * out .+ attn.bo
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

    lm_head = token_embedding

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

    x = model.token_embedding[tokens .+ 1, :]'
    x = x .+ model.position_embedding[1:T, :]'

    for block in model.blocks
        x = block(x)
    end

    x = model.final_ln(x)
    logits = model.lm_head * x

    return logits
end

function (model::TransformerModel)(tokens::AbstractVector{Int})::AbstractVector{Float32}
    T = length(tokens)
    x = model.token_embedding[tokens .+ 1, :]'
    x = x .+ model.position_embedding[1:T, :]'

    for block in model.blocks
        x = block(x)
    end

    x = model.final_ln(x)
    logits = model.lm_head * x

    return logits[:, 1]
end

function generate(model::TransformerModel, tokenizer, seed::String;
                  max_new_tokens::Int=100, temperature::Float32=1.0f0,
                  top_k::Int=40, rng=Random.Xoshiro(42))::String

    tokens = encode(tokenizer, seed)
    max_len = min(model.config.max_seq_len, length(tokens) + max_new_tokens)

    while length(tokens) < max_len
        context = tokens[max(1, end-model.config.max_seq_len+1):end]
        logits = model(context) ./ temperature

        if top_k > 0 && top_k < length(logits)
            sorted_idx = sortperm(logits, rev=true)
            keep_idx = sorted_idx[1:top_k]
            filtered_logits = Float32[-Inf for _ in 1:length(logits)]
            for i in keep_idx
                filtered_logits[i] = logits[i]
            end
            logits = filtered_logits
        end

        probs = softmax(logits)
        probs = max.(probs, Float32(1e-10))
        probs = probs ./ sum(probs)
        next_token = StatsBase.sample(rng, 1:length(probs), StatsBase.Weights(probs))

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
    println(io, "  → ~$(round(params/1e6, digits=1))M params, $(cfg.vocab_size) vocab, $(cfg.max_seq_len) ctx)")
end

# ============================================================================
# Simple BPE Tokenizer
# ============================================================================

struct SimpleTokenizer
    vocab::Dict{Vector{Int}, Int}
    reverse_vocab::Vector{Vector{Int}}
    eot_token::Int
    eot_byte::Vector{Int}
end

function SimpleTokenizer(vocab_size::Int=5000; rng=Random.Xoshiro(42))
    @assert vocab_size >= 256

    vocab = Dict{Vector{Int}, Int}()
    reverse_vocab = Vector{Vector{Int}}()

    for i in 0:255
        vocab[[i]] = i + 1
        push!(reverse_vocab, [i])
    end

    eot_token = vocab_size - 1
    push!(reverse_vocab, [256])
    vocab[[256]] = eot_token

    return SimpleTokenizer(vocab, reverse_vocab, eot_token, [256])
end

function _get_stats(freqs::Vector{Dict{Vector{Int}, Int}})
    pairs = Dict{Vector{Int}, Int}()
    for freqs_dict in freqs
        for (pair, count) in freqs_dict
            pairs[pair] = get(pairs, pair, 0) + count
        end
    end
    return pairs
end

function _merge_pair!(freqs::Vector{Dict{Vector{Int}, Int}}, pair::Vector{Int}, new_id::Int)
    for freqs_dict in freqs
        new_freqs = Dict{Vector{Int}, Int}()
        for (tokens, count) in freqs_dict
            new_tokens = Int[]
            i = 1
            while i <= length(tokens)
                if i < length(tokens) && tokens[i] == pair[1] && tokens[i+1] == pair[2]
                    push!(new_tokens, new_id)
                    i += 2
                else
                    push!(new_tokens, tokens[i])
                    i += 1
                end
            end
            new_freqs[new_tokens] = count
        end
        empty!(freqs_dict)
        for (k, v) in new_freqs
            freqs_dict[k] = v
        end
    end
end

function train(texts::Vector{String}, vocab_size::Int=5000; rng=Random.Xoshiro(42))
    tokenizer = SimpleTokenizer(vocab_size, rng=rng)

    freqs = [Dict{Vector{Int}, Int}() for _ in 1:length(texts)]

    for (i, text) in enumerate(texts)
        tokens = vcat([[Int(c) for c in text]..., [256]])
        freqs[i][tokens] = 1
    end

    current_id = 257
    target_size = vocab_size - 1

    while current_id <= target_size
        pairs = _get_stats(freqs)
        if isempty(pairs)
            break
        end

        best_pair = argmax(pairs)
        if pairs[best_pair] < 2
            break
        end

        push!(tokenizer.reverse_vocab, best_pair)
        tokenizer.vocab[best_pair] = current_id

        _merge_pair!(freqs, best_pair, current_id)
        current_id += 1
    end

    return tokenizer
end

function encode(tok::SimpleTokenizer, text::String)::Vector{Int}
    tokens = [Int(c) for c in text]
    result = Int[]

    i = 1
    while i <= length(tokens)
        longest = [tokens[i]]
        for j in (i+1):length(tokens)
            prefix = tokens[i:j]
            if haskey(tok.vocab, prefix)
                longest = prefix
            else
                break
            end
        end
        push!(result, tok.vocab[longest])
        i += length(longest)
    end

    return result
end

function decode(tok::SimpleTokenizer, tokens::Vector{Int})::String
    bytes = Vector{Int}()

    for token in tokens
        if token == tok.eot_token
            break
        end
        if token > 0 && token <= length(tok.reverse_vocab)
            append!(bytes, tok.reverse_vocab[token])
        end
    end

    bytes = bytes[bytes .!= 256]
    return String([Char(b) for b in bytes])
end

# ============================================================================
# GPT-2 Tokenizer
# ============================================================================

struct GPT2Tokenizer
    vocab::Dict{String, Int}
    reverse_vocab::Dict{Int, String}
    merges::Vector{Tuple{String, String}}
    byte_encoder::Dict{UInt8, String}
    byte_decoder::Dict{String, UInt8}
    eot_token::Int
end

function gpt2_byte_encoder()::Dict{UInt8, String}
    encoder = Dict{UInt8, String}()
    bs = UInt8[]
    for b in 0x21:0x7e; push!(bs, b); end
    for b in 0xa1:0xac; push!(bs, b); end
    for b in 0xae:0xff; push!(bs, b); end
    n = 0
    for b in 0x00:0xff
        if !(b in bs)
            push!(bs, b)
            encoder[b] = string(Char(256 + n))
            n += 1
        else
            encoder[b] = string(Char(b))
        end
    end
    return encoder
end

function gpt2_byte_decoder(encoder::Dict{UInt8, String})::Dict{String, UInt8}
    return Dict(v => k for (k, v) in encoder)
end

function GPT2Tokenizer(vocab_path::String, merges_path::String)
    vocab = JSON.parsefile(vocab_path)
    reverse_vocab = Dict{Int, String}(v => k for (k, v) in vocab)

    merges = Tuple{String, String}[]
    for line in eachline(merges_path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        parts = split(strip(line))
        if length(parts) == 2
            push!(merges, (parts[1], parts[2]))
        end
    end

    encoder = gpt2_byte_encoder()
    decoder = gpt2_byte_decoder(encoder)
    eot = get(vocab, "<|endoftext|>", 50256)

    return GPT2Tokenizer(vocab, reverse_vocab, merges, encoder, decoder, eot)
end

function bpe(tok::GPT2Tokenizer, word::String)::Vector{String}
    chars = [string(c) for c in word]
    if length(chars) == 1
        return chars
    end

    pairs = Dict{Tuple{String,String}, Int}()
    for i in 1:length(chars)-1
        pairs[(chars[i], chars[i+1])] = get(pairs, (chars[i], chars[i+1]), 0) + 1
    end

    while true
        best_pair = nothing
        best_rank = typemax(Int)
        for (pair, _) in pairs
            idx = findfirst(isequal(pair), tok.merges)
            if idx !== nothing && idx < best_rank
                best_rank = idx
                best_pair = pair
            end
        end

        best_pair === nothing && break

        new_chars = String[]
        i = 1
        while i <= length(chars)
            if i < length(chars) && (chars[i], chars[i+1]) == best_pair
                push!(new_chars, chars[i] * chars[i+1])
                i += 2
            else
                push!(new_chars, chars[i])
                i += 1
            end
        end
        chars = new_chars

        if length(chars) == 1
            break
        end

        pairs = Dict{Tuple{String,String}, Int}()
        for i in 1:length(chars)-1
            pairs[(chars[i], chars[i+1])] = get(pairs, (chars[i], chars[i+1]), 0) + 1
        end
    end

    return chars
end

function encode(tok::GPT2Tokenizer, text::String)::Vector{Int}
    text_bytes = Vector{UInt8}(text)
    encoded = [tok.byte_encoder[b] for b in text_bytes]
    pre_tokenized = split(join(encoded), " ")

    token_ids = Int[]
    for word in pre_tokenized
        word = strip(word)
        isempty(word) && continue
        bpe_tokens = bpe(tok, word)
        for t in bpe_tokens
            id = get(tok.vocab, t, nothing)
            if id !== nothing
                push!(token_ids, id)
            end
        end
    end

    return token_ids
end

function encode_with_eot(tok::GPT2Tokenizer, text::String)::Vector{Int}
    tokens = encode(tok, text)
    push!(tokens, tok.eot_token)
    return tokens
end

function decode(tok::GPT2Tokenizer, token_ids::Vector{Int})::String
    tokens = String[]
    for id in token_ids
        id == tok.eot_token && break
        t = get(tok.reverse_vocab, id, nothing)
        t !== nothing && push!(tokens, t)
    end
    text = join(tokens)
    bytes = UInt8[]
    i = 1
    while i <= length(text)
        c = text[i]
        b = get(tok.byte_decoder, string(c), nothing)
        if b !== nothing
            push!(bytes, b)
        else
            push!(bytes, UInt8(c))
        end
        i += 1
    end
    return String(bytes)
end

# ============================================================================
# Model Loading & Chat
# ============================================================================

const MODEL = Ref{Union{TransformerModel, Nothing}}(nothing)
const TOKENIZER = Ref{Union{SimpleTokenizer, GPT2Tokenizer, Nothing}}(nothing)

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

function is_loaded()::Bool
    return MODEL[] !== nothing && TOKENIZER[] !== nothing
end

# ============================================================================
# Pretrained weight loading (GPT-2 binary format)
# ============================================================================

const WEIGHTS_PATH = Ref{Union{String, Nothing}}(nothing)

"""
    load_pretrained(weights_path::String)

Load GPT-2 pretrained weights from a binary file exported by
scripts/export_gpt2_weights.py. The file format is:
1. A JSON header string (null-terminated)
2. Raw float32 data for each tensor in order
"""
function load_pretrained(weights_path::String;
                         vocab_path::Union{String,Nothing}=nothing,
                         merges_path::Union{String,Nothing}=nothing)
    @info "Loading pretrained GPT-2 weights from: $weights_path"

    if vocab_path !== nothing && merges_path !== nothing
        tok = GPT2Tokenizer(vocab_path, merges_path)
        TOKENIZER[] = tok
        @info "GPT-2 tokenizer loaded (vocab: $(length(tok.vocab)) tokens)"
    end

    data = read(weights_path)

    null_pos = findfirst(isequal(0x00), data)
    if null_pos === nothing
        error("Invalid weight file: no null terminator found")
    end

    header_bytes = data[1:null_pos[1]-1]
    header_str = String(header_bytes)
    tensor_info = JSON.parse(header_str)

    offset = null_pos[1] + 1
    raw_floats = reinterpret(Float32, data[offset:end])
    pos = Ref(1)

    function read_tensor(name::String)
        info = tensor_info[name]
        shape = info["shape"]
        shape_t = Tuple(reverse(shape))
        count = prod(Int, shape)
        w = reshape(raw_floats[pos[]:pos[]+count-1], shape_t)
        pos[] += count
        return w
    end

    cfg = TransformerConfig()
    n = cfg.n_embed
    wte = read_tensor("wte.weight")
    wpe = read_tensor("wpe.weight")

    blocks = TransformerBlock[]
    for i in 0:11
        ln1_w = vec(read_tensor("h.$i.ln_1.weight"))
        ln1_b = vec(read_tensor("h.$i.ln_1.bias"))
        ln2_w = vec(read_tensor("h.$i.ln_2.weight"))
        ln2_b = vec(read_tensor("h.$i.ln_2.bias"))

        ca_w = read_tensor("h.$i.attn.c_attn.weight")
        ca_b = vec(read_tensor("h.$i.attn.c_attn.bias"))
        cp_w = read_tensor("h.$i.attn.c_proj.weight")
        cp_b = vec(read_tensor("h.$i.attn.c_proj.bias"))

        Wq = ca_w[1:n, :]'
        Wk = ca_w[n+1:2*n, :]'
        Wv = ca_w[2*n+1:3*n, :]'
        bq = ca_b[1:n]
        bk = ca_b[n+1:2*n]
        bv = ca_b[2*n+1:3*n]
        Wo = cp_w'
        bo = cp_b

        fc_w = read_tensor("h.$i.mlp.c_fc.weight")
        fc_b = vec(read_tensor("h.$i.mlp.c_fc.bias"))
        mp_w = read_tensor("h.$i.mlp.c_proj.weight")
        mp_b = vec(read_tensor("h.$i.mlp.c_proj.bias"))

        W1 = fc_w
        b1 = fc_b
        W2 = mp_w
        b2 = mp_b

        hd = n ÷ cfg.n_heads
        attn = CausalAttention(Wq, Wk, Wv, Wo, bq, bk, bv, bo, cfg.n_heads, hd, Float32(1.0 / sqrt(hd)))
        ff = FeedForward(W1, b1, W2, b2, gelu)
        push!(blocks, TransformerBlock(attn, ff, LayerNorm(ln1_w, ln1_b, 1f-5, n), LayerNorm(ln2_w, ln2_b, 1f-5, n)))
    end

    ln_f_w = vec(read_tensor("ln_f.weight"))
    ln_f_b = vec(read_tensor("ln_f.bias"))
    final_ln = LayerNorm(ln_f_w, ln_f_b, 1f-5, n)

    model = TransformerModel(cfg, wte, wpe, blocks, final_ln, wte)
    WEIGHTS_PATH[] = weights_path

    return model
end

function load_local_model(weights_path::String;
                          vocab_path::Union{String,Nothing}=nothing,
                          merges_path::Union{String,Nothing}=nothing)
    model = load_pretrained(weights_path; vocab_path=vocab_path, merges_path=merges_path)
    MODEL[] = model
    return model
end

function chat_local(input::String; session_id::Int64=0)::Dict{String,Any}
    if !is_loaded()
        return Dict("error" => "Model not loaded. Call load_local_model() first.", "text" => "")
    end

    model = MODEL[]
    tokenizer = TOKENIZER[]

    response = generate(model, tokenizer, input; rng=Random.Xoshiro(time_ns() % Int64))

    return Dict(
        "text" => response,
        "session_id" => session_id,
        "model" => "NanoGPT (Pure Julia)"
    )
end

function chat(input::String; session_id::Int64=0)::Dict{String,Any}
    return chat_local(input; session_id=session_id)
end

end # module