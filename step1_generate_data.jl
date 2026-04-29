using IngExuity
using Random

const LLAMA = IngExuity.LlamaCpp
const IGSD = IngExuity.IGSDCore

function generate_distillation_data(n_samples::Int=1000; max_prompt_len::Int=32, max_new_tokens::Int=32)
    println("Initializing LlamaCpp (TinyLlama)...")
    llama = LLAMA.Llama()

    samples = []
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Hello, how are you?",
        "To infinity and beyond",
        "The quick brown fox",
        "It was a dark and stormy",
        "In the beginning",
        "After a long day",
        "The meaning of life is",
        "Science is",
    ]

    println("Generating $n_samples distillation samples...")
    for i in 1:n_samples
        prompt = rand(prompts)
        if i % 100 == 0
            println("  Sample $i/$n_samples...")
        end

        try
            input_ids = LLAMA.encode(llama, prompt)
            input_len = min(length(input_ids), max_prompt_len)
            input_tokens = input_ids[1:input_len]

            output_ids = LLAMA.generate(llama, input_tokens; max_new_tokens=max_new_tokens)
            push!(samples, (input=Vector{Int32}(input_tokens), output=Vector{Int32}(output_ids)))
        catch e
            println("  Warning: generation failed at sample $i: $e")
        end
    end

    println("Generated $(length(samples)) samples")
    return samples
end

function save_distillation_data(samples, filepath::String)
    open(filepath, "w") do f
        write(f, Int64(length(samples)))
        for (input, output) in samples
            write(f, Int64(length(input)))
            write(f, input)
            write(f, Int64(length(output)))
            write(f, output)
        end
    end
    println("Saved $(length(samples)) samples to $filepath ($(filesize(filepath)/1e6) MB)")
end

function load_distillation_data(filepath::String)
    open(filepath, "r") do f
        n = read(f, Int64)
        samples = []
        for _ in 1:n
            input_len = read(f, Int64)
            input = Vector{Int32}(undef, input_len)
            read!(f, input)
            output_len = read(f, Int64)
            output = Vector{Int32}(undef, output_len)
            read!(f, output)
            push!(samples, (input=input, output=output))
        end
        return samples
    end
end

println("=== Step 1: Generate distillation data from TinyLlama ===")
samples = generate_distillation_data(500)
save_distillation_data(samples, "distillation_data.bin")
println("Done! Distillation data saved.")