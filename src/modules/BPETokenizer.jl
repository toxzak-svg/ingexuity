# ============================================================================
# BPETokenizer.jl — GPT-2 style Byte-Pair Encoding Tokenizer in Pure Julia
# Based on OpenAI's GPT-2 BPE algorithm
# ============================================================================
module BPETokenizer

export BPETokenizer, encode, decode, load_gpt2_vocab

# ============================================================================
# Core BPE Types
# ============================================================================

struct BPETokenizer
    vocab::Dict{String, Int}
    vocab_scores::Dict{String, Float32}
    merges::Vector{Tuple{String, String}}
    byte_encoder::Dict{UInt8, String}
    byte_decoder::Dict{String, UInt8}
end

function byte_to_string(byte::UInt8)::String
    if byte < 0x80
        return string(Char(byte))
    else
        return string(Char(0xC2 + ((byte >> 6) & 0x01)), Char(0x80 + (byte & 0x3F)))
    end
end

function string_to_byte(s::String)::UInt8
    if length(s) == 1
        return UInt8(s[1])
    else
        b1 = UInt8(s[1]) - 0xC2
        b2 = UInt8(s[2]) - 0x80
        return (b1 << 6) | b2
    end
end

function default_byte_encoder()::Dict{UInt8, String}
    encoder = Dict{UInt8, String}()
    for i in UInt8(0):UInt8(127)
        encoder[i] = string(Char(i))
    end
    for i in UInt8(128):UInt8(255)
        encoder[i] = string(Char(0xC0 + ((i >> 6) & 0x03)), Char(0x80 + (i & 0x3F)))
    end
    encoder
end

function default_byte_decoder()::Dict{String, UInt8}
    decoder = Dict{String, UInt8}()
    for i in UInt8(0):UInt8(127)
        decoder[string(Char(i))] = i
    end
    for i in UInt8(128):UInt8(255)
        s = string(Char(0xC0 + ((i >> 6) & 0x03)), Char(0x80 + (i & 0x3F)))
        decoder[s] = i
    end
    decoder
end

function create_gpt2_vocab()::Tuple{Dict{String, Int}, Dict{String, Float32}}
    vocab = Dict{String, Int}()
    vocab_scores = Dict{String, Float32}()

    for i in 0:255
        s = byte_to_string(UInt8(i))
        vocab[s] = i
        vocab_scores[s] = 0.0f0
    end

    special_tokens = [
        (256, "<|endoftext|>"),
        (257, "<|start|>"),
        (258, "<|endofprompt|>"),
        (259, "<|padding|>"),
        (260, "<|unk|>"),
        (261, "<|system|>"),
        (262, "<|user|>"),
        (263, "<|assistant|>"),
    ]

    for (id, name) in special_tokens
        vocab[name] = id
        vocab_scores[name] = 0.0f0
    end

    return vocab, vocab_scores
end

function get_stats(word::Vector{String})::Dict{Tuple{String, String}, Int}
    pairs = Dict{Tuple{String, String}, Int}()
    for i in 1:length(word)-1
        pair = (word[i], word[i+1])
        pairs[pair] = get(pairs, pair, 0) + 1
    end
    return pairs
end

function merge_pair!(word::Vector{String}, pair::Tuple{String, String})
    new_word = String[]
    i = 1
    while i <= length(word)
        if i < length(word) && word[i] == pair[1] && word[i+1] == pair[2]
            push!(new_word, pair[1] * pair[2])
            i += 2
        else
            push!(new_word, word[i])
            i += 1
        end
    end
    return new_word
end

function encode(tok::BPETokenizer, text::String)::Vector{Int}
    text_bytes = Vector{UInt8}(codeunit(text))

    encoded = String[]
    for b in text_bytes
        push!(encoded, get(tok.byte_encoder, b, byte_to_string(b)))
    end

    changed = true
    while changed
        changed = false
        for (left, right) in tok.merges
            merged = left * right
            new_encoded = String[]
            i = 1
            found = false
            while i <= length(encoded)
                if i < length(encoded) && encoded[i] == left && encoded[i+1] == right
                    push!(new_encoded, merged)
                    i += 2
                    changed = true
                    found = true
                else
                    push!(new_encoded, encoded[i])
                    i += 1
                end
            end
            if found
                encoded = new_encoded
                break
            end
        end
    end

    token_ids = Int[]
    for token in encoded
        id = get(tok.vocab, token, -1)
        if id == -1
            id = get(tok.vocab, byte_to_string(UInt8(first(token))), -1)
            if id == -1
                push!(token_ids, 260)
            else
                push!(token_ids, id)
            end
        else
            push!(token_ids, id)
        end
    end

    return token_ids
end

function decode(tok::BPETokenizer, token_ids::Vector{Int})::String
    bytes = UInt8[]
    for id in token_ids
        token = nothing
        for (t, i) in tok.vocab
            if i == id
                token = t
                break
            end
        end

        if token === nothing
            continue
        end

        if length(token) == 1
            push!(bytes, UInt8(token[1]))
        else
            for c in token
                b = UInt8(c)
                if b >= 0x80
                    push!(bytes, b)
                end
            end
        end
    end

    String(bytes)
end

function load_gpt2_vocab()::BPETokenizer
    vocab, vocab_scores = create_gpt2_vocab()

    merges = Tuple{String, String}[]
    for i in 0:255
        s1 = byte_to_string(UInt8(i))
        for j in 0:255
            s2 = byte_to_string(UInt8(j))
            push!(merges, (s1, s2))
        end
    end

    BPETokenizer(
        vocab,
        vocab_scores,
        merges,
        default_byte_encoder(),
        default_byte_decoder()
    )
end

encode_batch(tok::BPETokenizer, texts::Vector{String}) = [encode(tok, t) for t in texts]
decode_batch(tok::BPETokenizer, batch::Vector{Vector{Int}}) = [decode(tok, ids) for ids in batch]

end # module