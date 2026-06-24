# ============================================================================
# CreativeIngenuity.jl — Creative generation with fallback paths
# v3: Generates novel solutions, integrates with research and emotional state
# ============================================================================
module CreativeIngenuity

using ..Types: InternalEmotional as InternalEmotionalType,
               UserModel as UserModelType

export generate, generate_with_fallbacks, suggest_alternatives

const DEFAULT_IDEAS = [
    "Try approaching this from a different angle — what would you tell a friend in this situation?",
    "What if the constraint you're feeling isn't actually a constraint?",
    "Sometimes the most honest answer is 'I don't know yet, but let me think about it.'",
    "Consider what you'd do if money weren't a factor — that often reveals the real priority.",
    "The thing you're avoiding might be exactly what needs to happen.",
    "What would 'good enough' look like? Perfection is often the enemy.",
    "Have you considered that the obstacle might actually be information, not a wall?",
    "Sometimes writing down the problem reveals the solution.",
    "What would you do if you weren't worried about what others think?",
    "The tension you're feeling might be a signal that something needs to change."
]

const CURIOSITY_PROMPTS = [
    "What have you tried that hasn't worked?",
    "What would the best possible outcome look like?",
    "What's the root cause you're really addressing?",
    "What would you tell someone you love in this situation?",
    "What are you optimizing for vs what matters?",
    "What would you do differently if you could start over?"
]

const NOVELTY_THRESHOLD = 0.6
const MAX_IDEAS = 3

function generate(research::Dict, internal::InternalEmotionalType;
                  context::Dict=Dict())::Dict{Symbol,Any}
    creative_mode = internal.arousal > 0.6
    emotional_intensity = internal.emotional_charge

    ideas = if creative_mode
        generate_creative_ideas(research, emotional_intensity, context)
    else
        generate_grounded_ideas(research, context)
    end

    tone = if internal.stress_level > 0.5
        "supportive"
    elseif internal.affective_state == "curious"
        "exploratory"
    elseif creative_mode
        "playful"
    else
        "practical"
    end

    selected = select_best_ideas(ideas, emotional_intensity, tone)

    Dict{Symbol,Any}(
        :ideas => selected,
        :creative_mode => creative_mode,
        :tone => tone,
        :confidence => length(selected) > 0 ? 0.7 : 0.3,
        :generation_mode => creative_mode ? "divergent" : "convergent"
    )
end

function generate_with_fallbacks(research::Dict, internal::InternalEmotionalType;
                                  max_attempts::Int=3)::Dict{Symbol,Any}
    attempt = 0
    last_result = Dict{Symbol,Any}()

    while attempt < max_attempts
        result = generate(research, internal)

        if length(result[:ideas]) > 0 && result[:confidence] > 0.5
            return result
        end

        last_result = result
        attempt += 1
    end

    Dict{Symbol,Any}(
        :ideas => DEFAULT_IDEAS[1:min(2, length(DEFAULT_IDEAS))],
        :creative_mode => false,
        :tone => "supportive",
        :confidence => 0.4,
        :generation_mode => "fallback",
        :fallback_used => true
    )
end

function generate_creative_ideas(research::Dict, emotional_intensity::Float64,
                                  context::Dict)::Vector{String}
    ideas = String[]

    topic = get(context, :topic, "general")
    topic_str = string(topic)

    topic_ideas = get_topic_specific_ideas(topic_str)
    append!(ideas, topic_ideas)

    if emotional_intensity > 0.5
        push!(ideas, "This feels heavy right now — what if we broke it into smaller pieces?")
        push!(ideas, "The emotional weight here suggests something deeper is going on.")
    end

    if get(context, :is_stressed, false)
        push!(ideas, "You're in a hard spot. Let's focus on what's actually in your control.")
        push!(ideas, "When everything feels like too much, the first step is just breathing.")
    end

    append!(ideas, sample_creative_prompts(2))

    ideas[1:min(length(ideas), MAX_IDEAS)]
end

function generate_grounded_ideas(research::Dict, context::Dict)::Vector{String}
    ideas = String[]

    topic = get(context, :topic, "general")
    topic_str = string(topic)

    topic_ideas = get_topic_specific_ideas(topic_str)
    append!(ideas, topic_ideas[1:min(2, length(topic_ideas))])

    append!(ideas, sample_practical_prompts(2))

    ideas[1:min(length(ideas), MAX_IDEAS)]
end

function get_topic_specific_ideas(topic::String)::Vector{String}
    topic_lower = lowercase(topic)

    if occursin("work", topic_lower) || occursin("career", topic_lower)
        return [
            "What would you do if you weren't afraid of failing?",
            "The best career move isn't always the most obvious one.",
            "What skills are you not using in your current role?",
            "Sometimes the job description and the actual work are different things."
        ]
    elseif occursin("relationship", topic_lower) || occursin("friend", topic_lower)
        return [
            "What do you need vs what are you asking for?",
            "The hardest conversations are usually the most important ones.",
            "What would a healthy version of this look like?",
            "Boundaries aren't walls — they're guidelines for how to be in relationship."
        ]
    elseif occursin("decision", topic_lower)
        return [
            "What does your gut say after sleeping on it?",
            "What's the cost of not deciding?",
            "You can always adjust later — most decisions aren't permanent.",
            "What's your biggest assumption here?"
        ]
    elseif occursin("stress", topic_lower) || occursin("anxiet", topic_lower)
        return [
            "What's the one thing that's actually bothering you underneath all of this?",
            "When did this start feeling overwhelming?",
            "What have you done in the past that helped?",
            "Stress is often a signal that something needs your attention."
        ]
    else
        return [
            "What's the core issue here?",
            "What would you tell a friend in this situation?",
            "What's the simplest path forward?",
            "What are you assuming that's maybe not true?"
        ]
    end
end

function sample_creative_prompts(n::Int)::Vector{String}
    indices = rand(1:length(CURIOSITY_PROMPTS), min(n, length(CURIOSITY_PROMPTS)))
    [CURIOSITY_PROMPTS[i] for i in indices]
end

function sample_practical_prompts(n::Int)::Vector{String}
    practical = [
        "What's the first step you could take right now?",
        "What's working that you should do more of?",
        "What's not working that you should stop doing?",
        "What information would change your approach?"
    ]
    indices = rand(1:length(practical), min(n, length(practical)))
    [practical[i] for i in indices]
end

function select_best_ideas(ideas::Vector{String}, emotional_intensity::Float64,
                           tone::String)::Vector{String}
    if isempty(ideas)
        return String[]
    end

    priority = if tone == "supportive"
        [i for i in ideas if contains(lowercase(i), "hard") || contains(lowercase(i), "feel")]
    elseif tone == "exploratory"
        [i for i in ideas if contains(lowercase(i), "what") || contains(lowercase(i), "why")]
    elseif tone == "playful"
        [i for i in ideas if contains(lowercase(i), "what if") || contains(lowercase(i), "try")]
    else
        ideas
    end

    selected = if length(priority) > 0
        priority
    else
        ideas[1:min(2, length(ideas))]
    end

    selected[1:min(MAX_IDEAS, length(selected))]
end

function suggest_alternatives(problem_description::String,
                              user_model::UserModelType)::Vector{String}
    style = user_model.communication_style
    style_label = string(style)

    base_ideas = DEFAULT_IDEAS[rand(1:length(DEFAULT_IDEAS), min(3, length(DEFAULT_IDEAS)))]

    if style_label == "direct" || style_label == "direct_comm"
        [replace(idea, r"maybe|perhaps|possibly" => "") for idea in base_ideas]
    else
        base_ideas
    end
end

end # module