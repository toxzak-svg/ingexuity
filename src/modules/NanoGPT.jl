# ============================================================================
# NanoGPT.jl — Scaled GPT for IngExuity
# Based on FluxML/model-zoo/text/nanogpt
# Scaled to ~50M params for local phone deployment
# ============================================================================
module NanoGPT

using Flux
using Statistics

export NanoGPTModel, GPTConfig, generate, train_step!

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef mutable struct GPTConfig
    # Vocabulary & tokens
    vocab_size::Int = 50257           # GPT-2 vocab size (BPE)
    max_seq_len::Int = 1024          # Context length
    
    # Model dimensions
    n_embed::Int = 512               # Model dimension (was 64 in nanoGPT)
    n_heads::Int = 8                 # Number of attention heads
    n_layers::Int = 12               # Number of layers (~50M params total)
    
    # FFN
    qk_dim::Int = 64                 # Query/key dimension (n_embed / n_heads)
    v_dim::Int = 64                  # Value dimension
    n_hidden::Int = 2048              # FFN hidden dim (4 * n_embed)
    
    # Regularization
    dropout::Float32 = 0.1f0
    
    # Generation
    temperature::Float32 = 1.0f0
    top_k::Int = 50
    top_p::Float32 = 0.95f0
end

# ============================================================================
# GPT Block (one transformer layer)
# ============================================================================

struct GPTBlock
    layernorm1::LayerNorm
    mha::MultiHeadAttention
    layernorm2::LayerNorm
    mlp::Chain
    dropout::Dropout
end

Flux.@layer GPTBlock

function GPTBlock(config::GPTConfig)
    GPTBlock(
        LayerNorm(config.n_embed),
        MultiHeadAttention(
            config.n_embed => (config.qk_dim, config.v_dim) => config.n_embed;
            nheads=config.n_heads,
            dropout_prob=config.dropout
        ),
        LayerNorm(config.n_embed),
        Chain(
            Dense(config.n_embed => config.n_hidden, gelu; init=Flux.glorot_uniform(gain=0.02)),
            Dropout(config.dropout),
            Dense(config.n_hidden => config.n_embed; init=Flux.glorot_uniform(gain=0.02)),
            Dropout(config.dropout),
        ),
        Dropout(config.dropout),
    )
end

function (m::GPTBlock)(x)
    # Pre-norm architecture (like GPT-2)
    # Attention block
    x_norm = m.layernorm1(x)
    y, _ = m.mha(x_norm; mask=NNlib.make_causal_mask(x))
    x = x + m.dropout(y)
    
    # FFN block
    x = x + m.mlp(m.layernorm2(x))
    
    return x
end

# ============================================================================
# Full GPT Model
# ============================================================================

struct NanoGPTModel
    config::GPTConfig
    tok_embed::Embedding
    pos_embed::Embedding
    dropout::Dropout
    blocks::Vector{GPTBlock}
    layernorm::LayerNorm
    output_head::Dense
end

Flux.@layer NanoGPTModel

function NanoGPTModel(config::GPTConfig)
    vocab_size = config.vocab_size
    
    model = NanoGPTModel(
        config,
        Embedding(vocab_size => config.n_embed),
        Embedding(config.max_seq_len => config.n_embed),
        Dropout(config.dropout),
        [GPTBlock(config) for _ in 1:config.n_layers],
        LayerNorm(config.n_embed),
        Dense(config.n_embed => vocab_size; bias=false),  # Weight tying
    )
    
    # Initialize weights
    init_weights!(model)
    
    return model
end

function init_weights!(m::NanoGPTModel)
    # Token embedding: small init
    m.tok_embed.weight .*= 0.02
    
    # Output head: same as embedding (weight tying means this IS the embedding
    # but we still init it properly)
    m.output_head.weight .*= 0.02
    
    return m
end

function (m::NanoGPTModel)(tokens::AbstractMatrix{Int})
    T, B = size(tokens)
    
    # Token + positional embeddings
    te = m.tok_embed(tokens)
    pe = m.pos_embed(1:T)
    x = m.dropout(te .+ pe)
    
    # Transformer blocks
    for blk in m.blocks
        x = blk(x)
    end
    
    # Final norm + output
    x = m.layernorm(x)
    x = m.output_head(x)
    
    return x  # (vocab_size, T, B)
end

# For single sequence
(m::NanoGPTModel)(tokens::AbstractVector{Int}) = m(reshape(tokens, length(tokens), 1))

# ============================================================================
# Parameter Count
# ============================================================================

function param_count(m::NanoGPTModel)::Int
    return sum(p -> length(p), Flux.params(m))
end

function Base.show(io::IO, m::NanoGPTModel)
    config = m.config
    params = param_count(m)
    println(io, "NanoGPTModel($(config.n_layers) layers, $(config.n_heads) heads, $(config.n_embed) embed)")
    println(io, "  → ~$(round(params/1e6, digits=1))M params, $(config.vocab_size) vocab, $(config.max_seq_len) ctx")
end

# ============================================================================
# Generation
# ============================================================================

function generate(model::NanoGPTModel, tokenizer, seed::String; 
                  max_new_tokens::Int=100,
                  temperature::Float32=model.config.temperature,
                  top_k::Int=model.config.top_k,
                  top_p::Float32=model.config.top_p,
                  rng=Random.default_rng())::String
    
    config = model.config
    
    # Encode seed
    tokens = encode(tokenizer, seed)
    max_len = min(config.max_seq_len, length(tokens) + max_new_tokens)
    
    while length(tokens) < max_len
        # Get context (last max_seq_len tokens)
        context = tokens[max(1, end-config.max_seq_len+1):end]
        context_matrix = reshape(context, length(context), 1)
        
        # Forward pass
        logits = model(context_matrix)[:, end, 1]  # (vocab_size,)
        
        # Apply temperature
        if temperature != 1.0f0
            logits = logits ./ temperature
        end
        
        # Top-k filtering
        if top_k > 0
            top_k_vals, top_k_idx = topk(logits, top_k)
            mask = ones(length(logits))
            mask[top_k_idx] .= 0
            logits = logits + mask .* (-1e10f0)
        end
        
        # Top-p (nucleus) filtering
        if top_p < 1.0f0
            sorted = sort(logits, rev=true)
            cumsum_probs = cumsum(softmax(sorted))
            cutoff_idx = findfirst(cumsum_probs .> top_p)
            if cutoff_idx !== nothing
                cutoff = sorted[cutoff_idx]
                logits = logits .* (logits .>= cutoff)
            end
        end
        
        # Sample
        probs = softmax(logits)
        next_token = sample(rng, 1:length(probs), Weights(probs))
        
        push!(tokens, next_token)
        
        # Stop on end-of-text token (token ID 50256 for GPT-2)
        if next_token == 50256
            break
        end
    end
    
    return decode(tokenizer, tokens)
end

# For chat: generate response given conversation history
function generate_chat(model::NanoGPTModel, tokenizer, messages::Vector{Dict{String, String}};
                       system_prompt::String="You are IngExuity, a helpful AI assistant.",
                       max_new_tokens::Int=200,
                       temperature::Float32=model.config.temperature,
                       rng=Random.default_rng())::String
    
    # Format as chat string
    full_text = "<|start|>$system_prompt<|endoftext|>"
    
    for msg in messages
        role = msg["role"]
        content = msg["content"]
        full_text *= "<|$role|>$content<|endoftext|>"
    end
    
    full_text *= "<|assistant|>"
    
    # Generate
    return generate(model, tokenizer, full_text; max_new_tokens, temperature, rng)
end

# ============================================================================
# Training
# ============================================================================

function loss_fn(model::NanoGPTModel, tokens::AbstractMatrix{Int})
    # tokens: (seq_len, batch_size)
    # Target: tokens shifted by 1
    T, B = size(tokens)
    
    # Input: all but last token
    input = tokens[1:T-1, :]
    # Target: all but first token
    targets = tokens[2:T, :]
    
    # Forward
    logits = model(input)  # (vocab_size, T-1, B)
    
    # Cross-entropy loss
    # Flux.logitcrossentropy handles this numerically stably
    return Flux.logitcrossentropy(logits, targets)
end

function train_step!(model::NanoGPTModel, opt_state, tokens::AbstractMatrix{Int})
    loss, grads = Flux.withgradient(model) do m
        loss_fn(m, tokens)
    end
    
    Flux.update!(opt_state, model, grads[1])
    
    return loss
end

function train_step!(model::NanoGPTModel, opt_state, tokens::AbstractMatrix{Int}, 
                     lr::Float32)
    # Create new opt state with specific learning rate
    opt = Flux.Adam(lr)
    opt_state = Flux.setup(opt, model)
    train_step!(model, opt_state, tokens)
end

# ============================================================================
# Checkpointing
# ============================================================================

using JLD2

function save_checkpoint(path::String, model::NanoGPTModel, opt_state, config::GPTConfig)
    jldsave(path;
            model_state=Flux.state(model),
            config=config,
            timestamp=time())
end

function load_checkpoint(path::String)::Tuple{NanoGPTModel, Any, GPTConfig}
    data = JLD2.load(path)
    config = data["config"]
    model = NanoGPTModel(config)
    Flux.loadstate!(model, data["model_state"])
    return model, nothing, config  # opt_state reconstruction requires lr, skip for now
end

# ============================================================================
# Convenience
# ============================================================================

# topk implementation for Flux compatibility
function topk(arr, k)
    sorted = sortperm(arr, rev=true)
    return arr[sorted[1:k]], sorted[1:k]
end

# Weights helper
struct Weights{T}
    probs::T
end
Base.length(w::Weights) = length(w.probs)
StatsBase.sample(rng, ::UnitRange{Int}, w::Weights) = sample(rng, eachindex(w.probs), Weights(w.probs))
function StatsBase.sample(rng, indices, w::Weights)
    return sample(rng, indices, Weights(w.probs[indices]))
end
function StatsBase.sample(rng, ::Base.OneTo{Int}, w::Weights)
    return sample(rng, 1:length(w.probs), w)
end

end # module
