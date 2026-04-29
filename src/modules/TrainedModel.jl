# ============================================================================
# TrainedModel.jl — LoRA Adapter Loading & Inference for IngExuity
# Loads TinyLlama base + LoRA adapter from trained_model directory
# ============================================================================
module TrainedModel

using LlamaCpp
using LlamaCpp: run_llama

export TrainedLLM, load_trained_model, chat_trained, is_trained_loaded

const ADAPTER_PATH = joinpath(@__DIR__, "..", "trained_model", "notebooks", "my_weights")
const BASE_MODEL_PATH = Ref{Union{String, Nothing}}(nothing)
const ADAPTER_LOADED = Ref{Bool}(false)

mutable struct TrainedLLM
    base_model::String
    adapter_path::String
    loaded::Bool
end

TrainedLLM() = TrainedLLM("", ADAPTER_PATH, false)

function load_trained_model(;
    base_model_path::Union{String, Nothing}=nothing,
    adapter_path::String=ADAPTER_PATH,
    n_gpu_layers::Int=0,
    n_threads::Int=4
)::String
    if base_model_path === nothing
        base_model_path = download_base_model()
    end

    if !isfile(base_model_path)
        error("Base model not found: $base_model_path")
    end

    BASE_MODEL_PATH[] = base_model_path
    ADAPTER_LOADED[] = true

    @info "Loaded trained model:"
    @info "  Base: $base_model_path"
    @info "  Adapter: $adapter_path"
    @info "  LoRA r=32, alpha=64"

    return base_model_path
end

function download_base_model(;
    dest::String=joinpath(tempdir(), "tinyllama-1.1b-chat.Q4_K_M.gguf"),
    force::Bool=false
)::String
    if isfile(dest) && !force
        @info "Base model already exists at: $dest"
        return dest
    end

    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    @info "Downloading TinyLlama 1.1B Q4_K_M base model..."
    Base.download(url, dest)
    @info "Download complete! Base model saved to: $dest"
    return dest
end

function chat_trained(
    prompt::String;
    model_path::Union{String, Nothing}=BASE_MODEL_PATH[],
    adapter_path::String=ADAPTER_PATH,
    n_gpu_layers::Int=0,
    n_threads::Int=4,
    max_tokens::Int=256,
    session_id::Int64=0
)::Dict{String, Any}
    if model_path === nothing || !ADAPTER_LOADED[]
        return Dict(
            "error" => "No trained model loaded. Call load_trained_model() first.",
            "text" => ""
        )
    end

    try
        result = run_llama(
            model=model_path,
            prompt=prompt,
            lora_adapter=adapter_path,
            n_gpu_layers=n_gpu_layers,
            nthreads=n_threads
        )

        return Dict(
            "text" => strip(result),
            "session_id" => session_id,
            "model" => "TinyLlama-1.1B-LoRA (r=32)"
        )
    catch e
        return Dict(
            "error" => string(e),
            "text" => ""
        )
    end
end

function chat(
    prompt::String;
    session_id::Int64=0,
    n_gpu_layers::Int=0,
    n_threads::Int=4
)::Dict{String, Any}
    chat_trained(prompt; session_id=session_id, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
end

function is_trained_loaded()::Bool
    return ADAPTER_LOADED[]
end

function get_model_info()::Dict{String, Any}
    if !is_trained_loaded()
        return Dict("status" => "no trained model loaded")
    end

    return Dict(
        "status" => "loaded",
        "base_model" => BASE_MODEL_PATH[],
        "adapter_path" => ADAPTER_PATH,
        "adapter_type" => "LoRA",
        "lora_r" => 32,
        "lora_alpha" => 64,
        "target_modules" => ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
end

end # module