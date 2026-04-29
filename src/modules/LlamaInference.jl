# ============================================================================
# LlamaInference.jl — LlamaCpp wrapper for IngExuity
# Uses llama.cpp for real pretrained model inference
# Supports both base TinyLlama and custom fine-tuned GGUF files
# ============================================================================
module LlamaInference

using LlamaCpp
using LlamaCpp: run_llama

export LlamaModel, load_llama_model, chat_llama, load_finetuned_model

const MODEL_PATH = Ref{Union{String, Nothing}}(nothing)
const MODEL_TYPE = Ref{String}("base")  # "base" or "finetuned"

"""
    download_tinyllama(; dest::String, force::Bool)

Download the base TinyLlama 1.1B Q4_K_M GGUF model (~634MB).
"""
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

"""
    load_llama_model(; model_path, n_gpu_layers, n_threads)

Load the base TinyLlama model. Downloads if not present.
"""
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
    MODEL_TYPE[] = "base"

    @info "Loaded model: $model_path"
    @info "Running on CPU with $n_threads threads"

    return model_path
end

"""
    load_finetuned_model(path; n_gpu_layers, n_threads)

Load a custom fine-tuned GGUF model from a training run.
Pass the path to the GGUF file or directory containing it.
"""
function load_finetuned_model(
    path::String;
    n_gpu_layers::Int=0,
    n_threads::Int=4
)::String
    # If path is a directory, find the GGUF file
    if isdir(path)
        gguf_files = filter(f -> occursin(".gguf", f), readdir(path, join=true))
        if isempty(gguf_files)
            error("No GGUF file found in directory: $path")
        end
        path = first(gguf_files)
    end

    if !isfile(path)
        error("Model file not found: $path")
    end

    MODEL_PATH[] = path
    MODEL_TYPE[] = "finetuned"

    @info "Loaded fine-tuned model: $path"
    @info "File size: $(round(filesize(path) / 1024 / 1024, digits=2)) MB"

    return path
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

    model_type_label = MODEL_TYPE[] == "finetuned" ? "TinyLlama-1.1B-Chat (Fine-tuned)" : "TinyLlama-1.1B-Chat (Q4_K_M)"

    return Dict(
        "status" => "loaded",
        "model_path" => MODEL_PATH[],
        "model_type" => model_type_label,
        "model_category" => MODEL_TYPE[],
        "size_mb" => round(filesize(MODEL_PATH[]) / 1024 / 1024, digits=2),
        "quantization" => "Q4_K_M (4-bit medium)"
    )
end

"""
    get_training_output_path()::Union{String, Nothing}

Get the expected path for fine-tuned model output from training runs.
Checks common locations where Kaggle/Paperspace save training outputs.
"""
function get_training_output_path()::Union{String, Nothing}
    candidates = [
        joinpath("tinyllama_finetuned", "final"),  # Local training output
        joinpath(".", "tinyllama_finetuned", "final"),
        joinpath(pwd(), "tinyllama_finetuned", "final"),
        joinpath(homedir(), "tinyllama_finetuned", "final")  # Home directory
    ]

    for path in candidates
        if isdir(path)
            # Look for GGUF file in the directory
            gguf_files = filter(f -> occursin(".gguf", f), readdir(path, join=true))
            if !isempty(gguf_files)
                return path
            end
        end
    end

    return nothing
end

"""
    load_training_model(; n_gpu_layers, n_threads)

Convenience function to auto-detect and load a fine-tuned model from training output.
"""
function load_training_model(; n_gpu_layers::Int=0, n_threads::Int=4)::String
    path = get_training_output_path()

    if path === nothing
        error("No fine-tuned model found. Train a model first with train_tinyllama.py")
    end

    return load_finetuned_model(path; n_gpu_layers=n_gpu_layers, n_threads=n_threads)
end

end # module