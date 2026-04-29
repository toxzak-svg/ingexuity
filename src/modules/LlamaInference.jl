# ============================================================================
# LlamaInference.jl — LlamaCpp wrapper for IngExuity
# Uses llama.cpp for real pretrained model inference
# ============================================================================
module LlamaInference

using LlamaCpp
using LlamaCpp: run_llama

export LlamaModel, load_llama_model, chat_llama

const MODEL_PATH = Ref{Union{String, Nothing}}(nothing)

function download_tinyllama(;
    dest::String=joinpath(tempdir(), "tinyllama-1.1b-chat.Q4_K_M.gguf"),
    force::Bool=false
)::String
    if isfile(dest) && !force
        @info "Model already exists at: $dest"
        return dest
    end

    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    @info "Downloading TinyLlama 1.1B Q4_K_M (~634MB)..."
    @info "This may take a while depending on your connection..."

    Base.download(url, dest)

    @info "Download complete! Model saved to: $dest"
    return dest
end

function load_llama_model(;
    model_path::Union{String, Nothing}=nothing,
    n_gpu_layers::Int=0,
    n_threads::Int=4
)::String
    if model_path === nothing
        model_path = download_tinyllama()
    end

    if !isfile(model_path)
        error("Model file not found: $model_path. Please download a GGUF model first.")
    end

    MODEL_PATH[] = model_path

    @info "Loaded model: $model_path"
    @info "Running on CPU with $n_threads threads"

    return model_path
end

function chat_llama(
    prompt::String;
    model_path::Union{String, Nothing}=MODEL_PATH[],
    n_gpu_layers::Int=0,
    n_threads::Int=4,
    max_tokens::Int=256,
    session_id::Int64=0
)::Dict{String, Any}
    if model_path === nothing
        return Dict(
            "error" => "No model loaded. Call load_llama_model() first.",
            "text" => ""
        )
    end

    try
        result = run_llama(
            model=model_path,
            prompt=prompt,
            n_gpu_layers=n_gpu_layers,
            nthreads=n_threads
        )

        return Dict(
            "text" => strip(result),
            "session_id" => session_id,
            "model" => "TinyLlama-1.1B-Chat (GGUF via llama.cpp)"
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
    chat_llama(prompt; session_id=session_id, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
end

function is_loaded()::Bool
    return MODEL_PATH[] !== nothing
end

function get_model_info()::Dict{String, Any}
    if !is_loaded()
        return Dict("status" => "no model loaded")
    end

    return Dict(
        "status" => "loaded",
        "model_path" => MODEL_PATH[],
        "model_type" => "TinyLlama-1.1B-Chat (Q4_K_M)",
        "size_mb" => round(filesize(MODEL_PATH[]) / 1024 / 1024, digits=2),
        "quantization" => "Q4_K_M (4-bit medium)"
    )
end

end # module