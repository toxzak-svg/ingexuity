# ============================================================================
# IGSDCore.jl — Integer Gated State Decoder Core for IngExuity
# A hybrid recurrent/state-space text generation core
# Designed for mobile/edge with integer-only inference
# ============================================================================
module IGSDCore

export IGSDConfig, IGSDLayer, IGSDCoreState, IGSDCore
export init_state, forward_step, generate, sample_next
export DEFAULT_CONFIG

using Random

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef mutable struct IGSDConfig
    vocab_size::Int32 = 12000
    d_model::Int32 = 320
    d_state::Int32 = 192
    n_layers::Int32 = 10
    local_kernel::Int32 = 16
    state_bits::Int32 = 16
    init_scale::Float32 = 0.02f0
end

function param_count(cfg::IGSDConfig)::Int
    vocab = cfg.vocab_size * cfg.d_model
    embed_out = cfg.vocab_size * cfg.d_model
    per_layer = 4 * cfg.d_model * cfg.d_state +
                3 * cfg.d_model * cfg.local_kernel +
                cfg.d_model * 4 +
                cfg.d_state * 2
    total = vocab + embed_out + cfg.n_layers * per_layer + cfg.d_model * 2
    return total
end

# ============================================================================
# Integer helpers
# ============================================================================

const INT8_MIN = Int8(-128)
const INT8_MAX = Int8(127)
const INT16_MAX = Int16(32767)
const INT16_MIN = Int16(-32768)

function clamp_int8(x::Int16)::Int8
    x = clamp(x, INT16_MIN, INT16_MAX)
    return Int8(clamp(x, INT8_MIN, INT8_MAX))
end

function clamp_int16(x::Int16)::Int16
    return clamp(x, INT16_MIN, INT16_MAX)
end

function clamp_int16(x::Int32)::Int16
    return Int16(clamp(x, -2147483648, 2147483647))
end

function safe_shift_right(x::Int32, bits::Int32)::Int16
    if bits >= 32
        return x < 0 ? INT16_MIN : Int16(0)
    elseif bits > 0
        shifted = x >> bits
        return clamp(shifted, Int32(INT16_MIN), Int32(INT16_MAX)) |> Int16
    else
        return clamp(x, Int32(INT16_MIN), Int32(INT16_MAX)) |> Int16
    end
end

function hard_sigmoid(x::Int16)::Int16
    x = clamp(x, Int16(-8), Int16(8)) + Int16(8)
    return safe_shift_right(Int32(x) * Int32(819), Int32(14))
end

function hard_tanh(x::Int16)::Int16
    if x > Int16(8)
        return INT16_MAX
    elseif x < Int16(-8)
        return INT16_MIN
    else
        return x * Int16(256)
    end
end

# ============================================================================
# Quantization helpers
# ============================================================================

function quantize_int8(w::Matrix{Float32}, scale::Float32)::Matrix{Int8}
    int32_arr = round.(Int32, w ./ scale)
    result = Matrix{Int8}(undef, size(w))
    for i in eachindex(int32_arr)
        result[i] = clamp_int8(Int16(int32_arr[i]))
    end
    return result
end

function dequantize_int8(w::Matrix{Int8}, scale::Float32)::Matrix{Float32}
    return Float32.(w) .* scale
end

# ============================================================================
# IGSD Layer — Core building block
# ============================================================================

struct IGSDLayer
    wu::Matrix{Int8}
    wg::Matrix{Int8}
    wr::Matrix{Int8}
    proj::Matrix{Int8}
    dwconv::Matrix{Int8}
    scale_u::Float32
    scale_g::Float32
    scale_r::Float32
    scale_out::Float32
    scale_dwconv::Float32
end

function create_igsd_layer(d_model::Int, d_state::Int, local_kernel::Int; rng=Random.Xoshiro(42), init_scale::Float32=0.02f0)
    scale = init_scale

    wu = quantize_int8(randn(rng, Float32, d_state, d_model) .* scale, scale)
    wg = quantize_int8(randn(rng, Float32, d_state, d_model) .* scale, scale)
    wr = quantize_int8(randn(rng, Float32, d_model + d_state + local_kernel, d_model + d_state + local_kernel) .* scale, scale)
    proj = quantize_int8(randn(rng, Float32, d_model, d_model + d_state + local_kernel) .* scale, scale)
    dwconv = quantize_int8(randn(rng, Float32, d_model, local_kernel) .* scale, scale)

    IGSDLayer(wu, wg, wr, proj, dwconv, scale, scale, scale, scale, scale)
end

function (layer::IGSDLayer)(
    x::Matrix{Int16},
    state::Vector{Int16},
    local_buf::Matrix{Int16}
)::Tuple{Matrix{Int16}, Vector{Int16}}
    d_model = size(layer.wu, 2)
    d_state = size(layer.wu, 1)
    local_kernel = size(layer.dwconv, 2)

    u = clamp_int16.(layer.wu * x)

    g_raw = clamp_int16.(layer.wg * x)
    g = hard_sigmoid.(g_raw)

    state_contrib = Int32(0)
    for i in 1:d_state
        state_contrib += Int32(state[i]) * Int32(g[i])
    end
    state_contrib = clamp_int16(state_contrib >> 8)

    u_contrib = Int32(0)
    for i in 1:d_state
        u_contrib += Int32(u[i]) * Int32(INT16_MAX - g[i])
    end
    u_contrib = safe_shift_right(u_contrib, Int32(8))

    new_state = clamp_int16.(fill(Int32(state_contrib + u_contrib), d_state))

    r_input = vcat(reshape(x, d_model, 1), reshape(state, d_state, 1), reshape(local_buf[1:local_kernel], local_kernel, 1))
    r = clamp_int16.(layer.wr * r_input)
    r = hard_tanh.(r)

    y = clamp_int16.(layer.proj * r)

    return y, new_state
end

# ============================================================================
# Local mixer — depthwise causal convolution
# ============================================================================

function causal_pad(x::Matrix{Int16}, kernel_size::Int)::Matrix{Int16}
    T = size(x, 2)
    d = size(x, 1)
    if T < kernel_size
        padding = kernel_size - T
        return hcat(fill(INT16_MIN, d, padding), x)
    end
    return x
end

function apply_dwconv(layer::IGSDLayer, x::Matrix{Int16})::Matrix{Int16}
    d_model, T = size(x)
    kernel_size = size(layer.dwconv, 2)

    x_padded = causal_pad(x, kernel_size)

    out = zeros(Int16, d_model, T)
    for t in 1:T
        for k in 1:kernel_size
            col_idx = t + k
            if col_idx <= size(x_padded, 2)
                for d in 1:d_model
                    out[d, t] += Int32(layer.dwconv[d, k]) * Int32(x_padded[d, col_idx])
                end
            end
        end
    end

    return clamp_int16.(out)
end

# ============================================================================
# Core state
# ============================================================================

mutable struct IGSDCoreState
    s::Vector{Int16}
    local_buf::Matrix{Int16}
    step::Int32
    discrete_tape::Matrix{Int8}
    discrete_codes::Matrix{Int8}
end

struct IGSDCore
    config::IGSDConfig
    wte::Matrix{Int8}
    lm_head::Matrix{Int8}
    layers::Vector{IGSDLayer}
    scale_wte::Float32
    scale_lm::Float32
end

function IGSDCore(cfg::IGSDConfig=IGSDConfig())
    rng = Random.Xoshiro(42)
    scale = cfg.init_scale

    wte = quantize_int8(randn(rng, Float32, cfg.vocab_size, cfg.d_model) .* scale, scale)
    lm_head = quantize_int8(randn(rng, Float32, cfg.vocab_size, cfg.d_model) .* scale, scale)

    layers = [
        create_igsd_layer(Int(cfg.d_model), Int(cfg.d_state), Int(cfg.local_kernel); rng=rng, init_scale=scale)
        for _ in 1:cfg.n_layers
    ]

    IGSDCore(cfg, wte, lm_head, layers, scale, scale)
end

function IGSDCore(weight_path::String)
    open(weight_path, "r") do f
        cfg = IGSDConfig(
            read(f, Int32),
            read(f, Int32),
            read(f, Int32),
            read(f, Int32),
            read(f, Int32),
            read(f, Int32),
            read(f, Float32)
        )
        scale_wte = read(f, Float32)
        scale_lm = read(f, Float32)
        wte = Matrix{Int8}(undef, cfg.vocab_size, cfg.d_model)
        read!(f, wte)
        lm_head = Matrix{Int8}(undef, cfg.vocab_size, cfg.d_model)
        read!(f, lm_head)
        layers = IGSDLayer[]
        for _ in 1:cfg.n_layers
            wu = Matrix{Int8}(undef, cfg.d_state, cfg.d_model)
            read!(f, wu)
            wg = Matrix{Int8}(undef, cfg.d_state, cfg.d_model)
            read!(f, wg)
            wr = Matrix{Int8}(undef, cfg.d_model, cfg.d_model + cfg.d_state + cfg.local_kernel)
            read!(f, wr)
            proj = Matrix{Int8}(undef, cfg.d_model, cfg.d_model + cfg.d_state + cfg.local_kernel)
            read!(f, proj)
            dwconv = Matrix{Int8}(undef, cfg.d_model, cfg.local_kernel)
            read!(f, dwconv)
            push!(layers, IGSDLayer(wu, wg, wr, proj, dwconv, cfg.init_scale, cfg.init_scale, cfg.init_scale, cfg.init_scale, cfg.init_scale))
        end
        IGSDCore(cfg, wte, lm_head, layers, scale_wte, scale_lm)
    end
end

function save_weights(core::IGSDCore, weight_path::String)
    open(weight_path, "w") do f
        write(f, core.config.vocab_size)
        write(f, core.config.d_model)
        write(f, core.config.d_state)
        write(f, core.config.n_layers)
        write(f, core.config.local_kernel)
        write(f, core.config.state_bits)
        write(f, core.config.init_scale)
        write(f, core.scale_wte)
        write(f, core.scale_lm)
        write(f, core.wte)
        write(f, core.lm_head)
        for layer in core.layers
            write(f, layer.wu)
            write(f, layer.wg)
            write(f, layer.wr)
            write(f, layer.proj)
            write(f, layer.dwconv)
        end
    end
end

function init_state(core::IGSDCore)::IGSDCoreState
    cfg = core.config
    n_slots = 8

    IGSDCoreState(
        zeros(Int16, cfg.d_state),
        zeros(Int16, cfg.d_model, cfg.local_kernel),
        Int32(0),
        zeros(Int8, cfg.n_layers, n_slots),
        zeros(Int8, cfg.n_layers, n_slots)
    )
end

# ============================================================================
# Forward step
# ============================================================================

function forward_step(
    core::IGSDCore,
    token_id::Int32,
    state::IGSDCoreState
)::Tuple{Vector{Int16}, IGSDCoreState}
    cfg = core.config

    token_emb_row = dequantize_int8(core.wte[token_id+1:token_id+1, :], core.scale_wte)
    x = clamp_int16.(round.(Int32, token_emb_row / cfg.init_scale))
    x = reshape(x, cfg.d_model, 1)

    new_local = zeros(Int16, cfg.d_model, cfg.local_kernel)
    for k in 1:cfg.local_kernel
        if k < size(state.local_buf, 2)
            new_local[:, k] = state.local_buf[:, k+1]
        end
    end
    new_local[:, end] = x

    new_state = init_state(core)
    y = zeros(Int16, cfg.d_model, 1)

    for (i, layer) in enumerate(core.layers)
        y, new_s = layer(x, state.s, new_local)
        new_state.s = new_s

        for j in 1:size(state.local_buf, 2)
            new_state.local_buf[:, j] = new_local[:, j]
        end
    end

    new_state.step = state.step + Int32(1)

    logits = clamp_int16.(core.lm_head * y)
    logits_vec = vec(logits)

    return logits_vec, new_state
end

# ============================================================================
# Sampling
# ============================================================================

function sample_next(
    logits::Vector{Int16},
    temperature::Float32=1.0f0,
    top_k::Int32=40
)::Int32
    if temperature > 0.0f0
        logits_scaled = Float32.(logits) ./ temperature
        logits_exp = exp.(logits_scaled .- maximum(logits_scaled))
        probs = logits_exp ./ sum(logits_exp)
        cumsum = 0.0f0
        r = rand(Float32)
        for (i, p) in enumerate(probs)
            cumsum += p
            if cumsum >= r
                return Int32(i - 1)
            end
        end
    end

    sorted_idx = sortperm(Vector{Int16}(logits), rev=true)
    top_idx = sorted_idx[1:min(top_k, length(sorted_idx))]
    best = argmax([logits[i] for i in top_idx])
    return top_idx[best] - Int32(1)
end

# ============================================================================
# Generation
# ============================================================================

function generate(
    core::IGSDCore,
    prompt_tokens::Vector{Int32};
    max_new_tokens::Int32=Int32(64),
    temperature::Float32=1.0f0,
    top_k::Int32=Int32(40)
)::Vector{Int32}
    state = init_state(core)
    all_tokens = copy(prompt_tokens)

    for token in prompt_tokens
        _, state = forward_step(core, token, state)
    end

    for _ in 1:max_new_tokens
        last_token = all_tokens[end]
        logits, state = forward_step(core, last_token, state)
        next_token = sample_next(logits, temperature, top_k)
        push!(all_tokens, next_token)

        if next_token == Int32(0)
            break
        end
    end

    return all_tokens
end

# ============================================================================
# Simple tokenizer interface
# ============================================================================

struct SimpleTokenizer
    vocab::Dict{Vector{UInt8}, Int32}
    reverse_vocab::Vector{Vector{UInt8}}
    eot_token::Int32
end

function SimpleTokenizer(vocab_size::Int32=12000; rng=Random.Xoshiro(42))
    vocab = Dict{Vector{UInt8}, Int32}()
    reverse_vocab = Vector{Vector{UInt8}}()

    for i in UInt8(0):UInt8(255)
        vocab[Vector{UInt8}([i])] = Int32(i)
        push!(reverse_vocab, Vector{UInt8}([i]))
    end

    eot_token = vocab_size - Int32(1)
    eot_bytes = Vector{UInt8}([UInt8(0xFF), UInt8(0xFE)])
    push!(reverse_vocab, eot_bytes)
    vocab[eot_bytes] = eot_token

    return SimpleTokenizer(vocab, reverse_vocab, eot_token)
end

function encode(tok::SimpleTokenizer, text::String)::Vector{Int32}
    tokens = Vector{Int32}()
    for c in text
        push!(tokens, Int32(c))
    end
    return tokens
end

function decode(tok::SimpleTokenizer, tokens::Vector{Int32})::String
    bytes = Vector{UInt8}()
    for t in tokens
        if t == tok.eot_token
            break
        elseif t >= Int32(0) && t <= Int32(255)
            push!(bytes, UInt8(t))
        end
    end
    return String(bytes)
end

# ============================================================================
# Default instance
# ============================================================================

const DEFAULT_CONFIG = IGSDConfig()
const DEFAULT_CORE = IGSDCore(DEFAULT_CONFIG)
const DEFAULT_TOKENIZER = SimpleTokenizer(DEFAULT_CONFIG.vocab_size)

function chat(prompt::String; max_tokens::Int32=Int32(64))::String
    tokens = encode(DEFAULT_TOKENIZER, prompt)
    result = generate(DEFAULT_CORE, tokens; max_new_tokens=max_tokens)
    return decode(DEFAULT_TOKENIZER, result)
end

end # module