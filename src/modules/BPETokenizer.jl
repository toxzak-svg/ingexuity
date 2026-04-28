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
    vocab::Dict{Vector{UInt8}, Int}       # byte sequence → token id
    vocab_scores::Dict{Vector{UInt8}, Float32}  # byte sequence → merge score
    merges::Vector{Tuple{Vector{UInt8}, Vector{UInt8}}}  # sorted merge pairs
    byte_encoder::Dict{UInt8, Vector{UInt8}}   # 0-255 → UTF-8 bytes
    byte_decoder::Dict{Vector{UInt8}, UInt8}   # UTF-8 bytes → 0-255
end

# ============================================================================
# Byte-Level Encoding (GPT-2 uses byte-level BPE)
# ============================================================================

function default_byte_encoder()::Dict{UInt8, Vector{UInt8}}
    # GPT-2 byte encoder: maps bytes 0-255 to visible UTF-8-compatible strings
    encoder = Dict{UInt8, Vector{UInt8}}()
    # 0-127: ASCII printable (same as themselves)
    for i in UInt8(0):UInt8(127)
        encoder[i] = Vector{UInt8}([i])
    end
    # 128-255: control bytes → "Ā" + byte value
    for i in UInt8(128):UInt8(255)
        encoder[i] = Vector{UInt8}('A' + (i - 128))
    end
    return encoder
end

function default_byte_decoder()::Dict{Vector{UInt8}, UInt8}
    decoder = Dict{Vector{UInt8}, UInt8}()
    for (k, v) in default_byte_encoder()
        decoder[v] = k
    end
    decoder[Vector{UInt8}([0xC2, 0x80])] = UInt8(128)
    decoder[Vector{UInt8}([0xC2, 0x81])] = UInt8(129)
    # ... (remaining high bytes)
    for i in UInt8(128):UInt8(191)
        decoder[Vector{UInt8}([0xC2, i])] = i
    end
    for i in UInt8(192):UInt8(191)
        # continuation bytes should not appear alone in decoder
    end
    return decoder
end

# ============================================================================
# GPT-2 Vocabulary (50,257 tokens)
# Simplified version — for production, load actual GPT-2 vocab
# ============================================================================

function create_gpt2_vocab()::Tuple{Dict{Vector{UInt8}, Int}, Dict{Vector{UInt8}, Float32}}
    # This creates a minimal GPT-2-style vocab.
    # For production: use the actual GPT-2 vocab from:
    # https://huggingface.co/gpt2/blob/main/vocab.json
    # and merges from: https://huggingface.co/gpt2/blob/main/merges.txt
    
    vocab = Dict{Vector{UInt8}, Int}()
    vocab_scores = Dict{Vector{UInt8}, Float32}()
    
    # Token IDs 0-255: single bytes (byte-level)
    for i in 0:255
        vocab[Vector{UInt8}([UInt8(i)])] = i
        vocab_scores[Vector{UInt8}([UInt8(i)])] = 0.0f0
    end
    
    # Token IDs 256-266: special tokens (GPT-2 style)
    special_tokens = [
        (256, "<|endoftext|>"),
        (257, "<|start|>"),
        (258, "<|delimiter|>"),
        (259, "<|padding|>"),
        (260, "<|unk|>"),
        (261, "<|system|>"),
        (262, "<|user|>"),
        (263, "<|assistant|>"),
        (264, "<|function|>"),
        (265, "<|tool_call|>"),
        (266, "<|tool_result|>"),
    ]
    for (id, name) in special_tokens
        vocab[Vector{UInt8}(codeunit.(name))] = id
        vocab_scores[Vector{UInt8}(codeunit.(name))] = 0.0f0
    end
    
    return vocab, vocab_scores
end

# ============================================================================
# BPE Merge Algorithm
# ============================================================================

function get_stats(word::Vector{Vector{UInt8}})::Dict{Tuple{Vector{UInt8}, Vector{UInt8}), Int}
    # Count adjacent pairs in a word
    pairs = Dict{Tuple{Vector{UInt8}, Vector{UInt8}), Int}()
    for i in 1:length(word)-1
        pair = (word[i], word[i+1])
        pairs[pair] = get(pairs, pair, 0) + 1
    end
    return pairs
end

function merge_pair!(word::Vector{Vector{UInt8}}, pair::Tuple{Vector{UInt8}, Vector{UInt8}})
    # Replace all occurrences of pair with a single merged token
    new_word = Vector{Vector{UInt8}}()
    i = 1
    while i <= length(word)
        if i < length(word) && word[i] == pair[1] && word[i+1] == pair[2]
            push!(new_word, vcat(pair[1], pair[2]))
            i += 2
        else
            push!(new_word, word[i])
            i += 1
        end
    end
    return new_word
end

# ============================================================================
# Train BPE (for custom corpora)
# ============================================================================

function train_bpe(texts::Vector{String}, vocab_size::Int=50000; num_merges::Int=10000)
    # Collect all byte sequences from texts
    byte_sequences = Vector{Vector{Vector{UInt8}}}()
    
    for text in texts
        bytes = Vector{UInt8}(codeunit(text))
        # Convert to "words" (each byte is initially a separate token)
        word = [[b] for b in bytes]
        push!(byte_sequences, word)
    end
    
    # Get initial pair counts
    pair_counts = Dict{Tuple{Vector{UInt8}, Vector{UInt8}), Int}()
    for word in byte_sequences
        for (pair, count) in get_stats(word)
            pair_counts[pair] = get(pair_counts, pair, 0) + count
        end
    end
    
    # Iteratively merge most common pair
    merges = Tuple{Vector{UInt8}, Vector{UInt8}}[]
    vocab, vocab_scores = create_gpt2_vocab()
    
    for i in 1:num_merges
        if isempty(pair_counts)
            break
        end
        
        # Find most common pair
        most_common = nothing
        max_count = 0
        for (pair, count) in pair_counts
            if count > max_count
                max_count = count
                most_common = pair
            end
        end
        
        if most_common === nothing || max_count < 2
            break
        end
        
        push!(merges, most_common)
        
        # Merge in all words
        new_pair_counts = Dict{Tuple{Vector{UInt8}, Vector{UInt8}), Int}()
        for word in byte_sequences
            new_word = merge_pair!(word, most_common)
            if new_word != word
                word .= new_word  # mutate in place
                for (pair, count) in get_stats(new_word)
                    new_pair_counts[pair] = get(new_pair_counts, pair, 0) + count
                end
            end
        end
        
        # Update pair counts
        for (pair, count) in pair_counts
            if pair != most_common
                # This pair still exists but counts may have changed
                total = count
                for new_pair in keys(new_pair_counts)
                    if new_pair == pair
                        total = new_pair_counts[pair]
                        break
                    end
                end
                if total > 0
                    new_pair_counts[pair] = total
                end
            end
        end
        pair_counts = new_pair_counts
        
        # Add merged token to vocab
        merged_token = vcat(most_common[1], most_common[2])
        next_id = length(vocab)
        vocab[merged_token] = next_id
        vocab_scores[merged_token] = Float32(max_count)
    end
    
    return BPETokenizer(vocab, vocab_scores, merges, default_byte_encoder(), default_byte_decoder())
end

# ============================================================================
# Encode (text → token IDs)
# ============================================================================

function encode(tok::BPETokenizer, text::String)::Vector{Int}
    # 1. Convert text to bytes
    text_bytes = Vector{UInt8}(codeunit(text))
    
    # 2. Encode bytes using byte encoder
    encoded = Vector{UInt8}[]
    for b in text_bytes
        push!(encoded, get(tok.byte_encoder, b, [b]))
    end
    
    # 3. Apply BPE merges (longest match first)
    # Sort merges by length (longer first) for greedy longest-match
    sorted_merges = sort(tok.merges, by=x -> -length(x[1]))
    
    changed = true
    while changed
        changed = false
        for (left, right) in sorted_merges
            merged = vcat(left, right)
            new_encoded = Vector{UInt8}[]
            i = 1
            while i <= length(encoded)
                # Try to find this pair at position i
                if i < length(encoded) && encoded[i] == left && encoded[i+1] == right
                    push!(new_encoded, merged)
                    i += 2
                    changed = true
                else
                    push!(new_encoded, encoded[i])
                    i += 1
                end
            end
            if changed
                encoded = new_encoded
                break
            end
        end
    end
    
    # 4. Convert to token IDs
    token_ids = Int[]
    for token in encoded
        id = get(tok.vocab, token, get(tok.vocab, Vector{UInt8}(token), -1))
        if id == -1
            # Unknown token: use <|unk|> or split into bytes
            push!(token_ids, 260)  # <|unk|>
        else
            push!(token_ids, id)
        end
    end
    
    return token_ids
end

# ============================================================================
# Decode (token IDs → text)
# ============================================================================

function decode(tok::BPETokenizer, token_ids::Vector{Int})::String
    # 1. Convert token IDs back to byte sequences
    bytes = Vector{UInt8}()
    for id in token_ids
        # Find token by ID
        token = nothing
        for (t, i) in tok.vocab
            if i == id
                token = t
                break
            end
        end
        
        if token === nothing
            continue  # skip unknown
        end
        
        # Decode using byte decoder
        decoded = get(tok.byte_decoder, token, token)
        append!(bytes, decoded)
    end
    
    # 2. Convert bytes to string
    return String(bytes)
end

# ============================================================================
# Load Pre-trained GPT-2 Tokenizer
# ============================================================================

function load_gpt2_vocab()::BPETokenizer
    # For a full GPT-2 tokenizer, download from HuggingFace:
    # - vocab.json (the BPE vocabulary)
    # - merges.txt (the merge operations)
    #
    # This function creates a minimal working GPT-2 compatible tokenizer.
    # For production use, implement load_gpt2_from_huggingface() instead.
    
    vocab, vocab_scores = create_gpt2_vocab()
    
    # Add common merge pairs (subset of GPT-2 merges for demo)
    merges = Tuple{Vector{UInt8}, Vector{UInt8}}[]
    
    # Single-byte merges
    for i in 0:255
        push!(merges, (Vector{UInt8}([UInt8(i)]), Vector{UInt8}([UInt8(i)])))
    end
    
    return BPETokenizer(
        vocab, 
        vocab_scores, 
        merges, 
        default_byte_encoder(), 
        default_byte_decoder()
    )
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    encode_batch(tok::BPETokenizer, texts::Vector{String})::Vector{Vector{Int}}

Encode a batch of texts into token IDs.
"""
encode_batch(tok::BPETokenizer, texts::Vector{String}) = [encode(tok, t) for t in texts]

"""
    decode_batch(tok::BPETokenizer, batch::Vector{Vector{Int}})::Vector{String}

Decode a batch of token IDs back to texts.
"""
decode_batch(tok::BPETokenizer, batch::Vector{Vector{Int}}) = [decode(tok, ids) for ids in batch]

"""
    save_tokenizer(tok::BPETokenizer, path::String)

Save tokenizer to a directory.
"""
function save_tokenizer(tok::BPETokenizer, path::String)
    mkpath(path)
    # Save vocab as JSON
    vocab_json = Dict{String, Any}()
    for (k, v) in tok.vocab
        vocab_json[string(k)] = v
    end
    write(joinpath(path, "vocab.json"), JSON.json(vocab_json))
    
    # Save merges
    merges_str = join(["$(join(Char.(m[1])))\n$(join(Char.(m[2])))" for m in tok.merges], "\n")
    write(joinpath(path, "merges.txt"), merges_str)
end

"""
    load_tokenizer(path::String)::BPETokenizer

Load a tokenizer from a directory.
"""
function load_tokenizer(path::String)::BPETokenizer
    # Load vocab JSON
    vocab_json = JSON.parse(read(joinpath(path, "vocab.json"), String))
    vocab = Dict{Vector{UInt8}, Int}()
    for (k, v) in vocab_json
        vocab[Vector{UInt8}(codeunit(k))] = v
    end
    
    # Load merges
    merges_txt = read(joinpath(path, "merges.txt"), String)
    merges = Tuple{Vector{UInt8}, Vector{UInt8}}[]
    for line in split(merges_txt, "\n")
        if isempty(line) || occursin(":")
            continue
        end
        parts = split(line, "\n")
        if length(parts) == 2
            push!(merges, (Vector{UInt8}(codeunit(parts[1])), Vector{UInt8}(codeunit(parts[2]))))
        end
    end
    
    return BPETokenizer(vocab, Dict{Vector{UInt8}, Float32}(), merges, default_byte_encoder(), default_byte_decoder())
end

end # module
