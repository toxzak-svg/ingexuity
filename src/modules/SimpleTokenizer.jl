# ============================================================================
# SimpleTokenizer.jl — Simple BPE Tokenizer for IngExuity
# Pure Julia implementation, no external dependencies
# ============================================================================
module SimpleTokenizer

export SimpleTokenizer, encode, decode

using Random

struct SimpleTokenizer
    vocab::Dict{Vector{Int}, Int}
    reverse_vocab::Vector{Vector{Int}}
    eot_token::Int
    eot_byte::Vector{Int}
end

function SimpleTokenizer(vocab_size::Int=5000; rng=Random.Xoshiro(42))
    @assert vocab_size >= 256

    vocab = Dict{Vector{Int}, Int}()
    reverse_vocab = Vector{Vector{Int}}[]

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

end # module