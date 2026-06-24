# ============================================================================
# Research.jl — Investigate and gather information
# ============================================================================
module Research

using ..Types: HumanInput as HumanInputType, WebSearchResult, LiveDataQuery,
              requires_live_data, detect_live_data_query
using Dates, HTTP

export investigate, fill_gaps, fetch_live_data, search_web,
       is_live_data_query, search_cache

# ----------------------------------------------------------------------------
# Live data search cache (in-memory, short TTL)
# ----------------------------------------------------------------------------
const SEARCH_CACHE = Dict{String, Tuple{Vector{WebSearchResult}, Dates.DateTime}}()
const CACHE_TTL_SECONDS = 300  # 5 minutes

function get_cache_ttl()
    Dates.Second(CACHE_TTL_SECONDS)
end

function clean_expired_cache!()
    now = Dates.now()
    for (key, (_, timestamp)) in SEARCH_CACHE
        if now - timestamp > get_cache_ttl()
            delete!(SEARCH_CACHE, key)
        end
    end
end

# ----------------------------------------------------------------------------
# Web search functions (using DuckDuckGo HTML for no-API-key access)
# ----------------------------------------------------------------------------

"""
    search_web(query::String; num_results::Int=5)::Vector{WebSearchResult}

Search the web using DuckDuckGo HTML (no API key required).
Returns vector of WebSearchResult with title, snippet, URL, and relevance.
"""
function search_web(query::String; num_results::Int=5)::Vector{WebSearchResult}
    clean_expired_cache!()
    
    # Check cache first
    cache_key = lowercase(query)
    if haskey(SEARCH_CACHE, cache_key)
        cached_results, cached_time = SEARCH_CACHE[cache_key]
        if Dates.now() - cached_time < get_cache_ttl()
            return cached_results
        end
    end
    
    # Perform fresh search
    results = _duckduckgo_search(query, num_results)
    
    # Cache the results
    SEARCH_CACHE[cache_key] = (results, Dates.now())
    
    return results
end

"""
    _duckduckgo_search(query::String, num_results::Int)::Vector{WebSearchResult}

Low-level DuckDuckGo HTML search. Falls back to alternative sources on failure.
"""
function _duckduckgo_search(query::String, num_results::Int)::Vector{WebSearchResult}
    try
        # DuckDuckGo HTML search endpoint
        encoded_query = HTTP.URIs.escapeuri(query)
        url = "https://html.duckduckgo.com/html/?q=$(encoded_query)&kl=us-en"
        
        response = HTTP.get(url, 
            ["User-Agent" => "Mozilla/5.0 (compatible; IngExuity/1.0)"],
            timeout=10)
        
        html = String(response.body)
        
        # Parse results from HTML
        results = _parse_duckduckgo_html(html, num_results)
        
        if !isempty(results)
            return results
        end
    catch e
        @debug "DuckDuckGo search failed" exception=e
    end
    
    # Fallback: try Bing HTML search
    return _bing_search_fallback(query, num_results)
end

"""
    _parse_duckduckgo_html(html::String, num_results::Int)::Vector{WebSearchResult}

Parse DuckDuckGo HTML results to extract titles, snippets, and URLs.
"""
function _parse_duckduckgo_html(html::String, num_results::Int)::Vector{WebSearchResult}
    results = WebSearchResult[]
    
    # DuckDuckGo HTML format has result classes
    # <a class="result__a" href="...">Title</a>
    # <a class="result__snippet" href="...">Snippet...</a>
    
    # Simple regex-based parsing
    title_pattern = r"<a class=\"result__a\"[^>]*href=\"([^\"]+)\"[^>]*>([^<]+)</a>"
    snippet_pattern = r"<a class=\"result__snippet\"[^>]*>([^<]+)</a>"
    
    title_matches = collect(m.match for m in eachmatch(title_pattern, html))
    snippet_matches = collect(m.match for m in eachmatch(snippet_pattern, html))
    
    # Extract URLs and titles
    url_pattern = r"href=\"([^\"]+)\""
    title_text_pattern = r">([^<]+)</a>"
    
    titles_and_urls = Tuple{String,String}[]
    for match_str in title_matches[1:min(num_results, length(title_matches))]
        url_match = match(url_pattern, match_str)
        title_match = match(title_text_pattern, match_str)
        if url_match !== nothing && title_match !== nothing
            push!(titles_and_urls, (title_match[1], url_match[1]))
        end
    end
    
    snippets = String[]
    for match_str in snippet_matches[1:min(num_results, length(snippet_matches))]
        # Extract text from <a ...>text</a>
        clean = replace(match_str, r"<[^>]+>" => "")
        clean = replace(clean, "&amp;" => "&", "&quot;" => "\"", "&apos;" => "'")
        push!(snippets, strip(clean))
    end
    
    # Build results
    for (i, (title, url)) in enumerate(titles_and_urls)
        snippet = i <= length(snippets) ? snippets[i] : ""
        push!(results, WebSearchResult(title, snippet, url, 1.0 - (i-1) * 0.1))
    end
    
    return results
end

"""
    _bing_search_fallback(query::String, num_results::Int)::Vector{WebSearchResult}

Fallback using Bing HTML search.
"""
function _bing_search_fallback(query::String, num_results::Int)::Vector{WebSearchResult}
    try
        encoded_query = HTTP.URIs.escapeuri(query)
        url = "https://www.bing.com/search?q=$(encoded_query)"
        
        response = HTTP.get(url,
            ["User-Agent" => "Mozilla/5.0 (compatible; IngExuity/1.0)"],
            timeout=10)
        
        html = String(response.body)
        
        # Parse Bing results
        results = _parse_bing_html(html, num_results)
        return results
    catch e
        @debug "Bing fallback failed" exception=e
        return WebSearchResult[]
    end
end

"""
    _parse_bing_html(html::String, num_results::Int)::Vector{WebSearchResult}

Parse Bing HTML results.
"""
function _parse_bing_html(html::String, num_results::Int)::Vector{WebSearchResult}
    results = WebSearchResult[]
    
    # Bing uses h2 titles with class b_topTitle and snippets with class b_paractext
    title_pattern = r"<h2[^>]*><a[^>]*href=\"([^\"]+)\"[^>]*>([^<]+)</a></h2>"
    snippet_pattern = r"<p class=\"b_paractext\"[^>]*>([^<]+)</p>"
    
    title_matches = collect(m.match for m in eachmatch(title_pattern, html))
    snippet_matches = collect(m.match for m in eachmatch(snippet_pattern, html))
    
    titles_and_urls = Tuple{String,String}[]
    url_pattern = r"href=\"([^\"]+)\""
    title_text_pattern = r">([^<]+)</a>"
    
    for match_str in title_matches[1:min(num_results, length(title_matches))]
        url_match = match(url_pattern, match_str)
        title_match = match(title_text_pattern, match_str)
        if url_match !== nothing && title_match !== nothing
            push!(titles_and_urls, (title_match[1], url_match[1]))
        end
    end
    
    snippets = String[]
    for match_str in snippet_matches
        clean = replace(match_str, r"<[^>]+>" => "")
        clean = replace(clean, "&amp;" => "&")
        push!(snippets, strip(clean))
    end
    
    for (i, (title, url)) in enumerate(titles_and_urls)
        snippet = i <= length(snippets) ? snippets[i] : ""
        push!(results, WebSearchResult(title, snippet, url, 1.0 - (i-1) * 0.1))
    end
    
    return results
end

# ----------------------------------------------------------------------------
# Live data detection and fetching
# ----------------------------------------------------------------------------

"""
    is_live_data_query(input::String)::Bool

Returns true if the input likely requires live/current data not in the base model.
"""
is_live_data_query(input::String) = requires_live_data(input)

"""
    fetch_live_data(query::String)::Vector{WebSearchResult}

Fetch live data for a query. Returns search results with snippets.
"""
function fetch_live_data(query::String)::Vector{WebSearchResult}
    search_web(query)
end

"""
    fetch_live_data_for_message(message::String)::Union{Dict{String,Any},Nothing}

Check if a message needs live data and fetch it if so.
Returns a dict with live_data info or nothing if not needed.
"""
function fetch_live_data_for_message(message::String)::Union{Dict{String,Any},Nothing}
    live_query = detect_live_data_query(message)
    
    if live_query === nothing
        return nothing
    end
    
    results = fetch_live_data(live_query.query)
    
    if isempty(results)
        return nothing
    end
    
    # Build context from results
    context_parts = String[]
    for (i, result) in enumerate(results[1:min(3, length(results))])
        push!(context_parts, "$(result.title): $(result.snippet)")
    end
    
    Dict{String,Any}(
        "query_type" => string(live_query.query_type),
        "confidence" => live_query.confidence,
        "results" => [Dict("title" => r.title, "snippet" => r.snippet, "url" => r.url) for r in results],
        "context" => join(context_parts, " | "),
        "needs_verification" => live_query.needs_verification
    )
end

# ----------------------------------------------------------------------------
# Original Research.jl functionality (enhanced)
# ----------------------------------------------------------------------------

function investigate(human_input, comprehension; curiosity=nothing)
    sentiment = haskey(comprehension, :sentiment) ? comprehension[:sentiment] : 0.0
    raw = human_input.raw
    words = split(lowercase(raw))

    depth = length(words) < 15 ? "brief" : length(words) < 40 ? "moderate" : "deep"

    inquiry_types = Symbol[]
    if any(w in ["how", "why", "what", "when", "where", "who"] for w in words)
        push!(inquiry_types, :causal)
    end
    if any(w in ["if", "what if", "suppose", "assume"] for w in words)
        push!(inquiry_types, :hypothetical)
    end
    if any(w in ["explain", "tell me", "describe", "understand"] for w in words)
        push!(inquiry_types, :explanatory)
    end

    gaps_to_fill = curiosity !== nothing ? get(curiosity, :gaps, Symbol[]) : Symbol[]

    # Check for live data needs
    live_data = fetch_live_data_for_message(raw)

    Dict(
        :query => human_input.raw,
        :topic => comprehension[:topic],
        :sentiment => sentiment,
        :depth => depth,
        :inquiry_types => inquiry_types,
        :gaps_to_fill => gaps_to_fill,
        :live_data => live_data,
        :requires_live_data => live_data !== nothing
    )
end

function fill_gaps(investigation::Dict, user_model)::Dict{Symbol,Any}
    filled = Dict{Symbol,Any}()

    gaps = get(investigation, :gaps_to_fill, Symbol[])

    if :topic_depth in gaps
        filled[:topic_suggestion] = "What matters most to you about this?"
    end

    if :stress_patterns in gaps
        filled[:stress_probe] = "Is there anything that's been weighing on you lately?"
    end

    if :temporal_patterns in gaps
        filled[:temporal_probe] = "Has this been a recurring theme for you?"
    end

    if :withdrawal_triggers in gaps
        filled[:withdrawal_probe] = "Take your time — I'm here when you're ready."
    end

    filled
end

end # module