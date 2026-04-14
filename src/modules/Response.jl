# ============================================================================
# Response.jl — Shape and formulate the response
# ============================================================================
module Response

using ..Types: Response as ResponseType, ResponseTone as ResponseToneType,
               RESPONSE_TONE_DIRECT, RESPONSE_TONE_WARM,
               RESPONSE_TONE_PLAYFUL, RESPONSE_TONE_CURIOUS,
               RESPONSE_TONE_MINIMAL, RESPONSE_TONE_STAYING_PRESENT

export formulate, adjust_tone

function tone_value(tone)
    if tone === :direct
        RESPONSE_TONE_DIRECT
    elseif tone === :warm
        RESPONSE_TONE_WARM
    elseif tone === :playful
        RESPONSE_TONE_PLAYFUL
    elseif tone === :curious
        RESPONSE_TONE_CURIOUS
    elseif tone === :minimal
        RESPONSE_TONE_MINIMAL
    elseif tone === :staying_present
        RESPONSE_TONE_STAYING_PRESENT
    else
        tone
    end
end

function formulate(predictions, comprehension; tone=:direct)
    topic = comprehension[:topic]
    is_question = comprehension[:is_question]

    content = if is_question
        "That's a good question. Let me think about $(topic) with you."
    elseif topic == "emotional"
        "I hear you. What's on your mind?"
    elseif topic == "work"
        "Tell me more about what you're working on."
    elseif topic == "positive"
        "That's great to hear! What made it so good?"
    else
        "I'm here. What do you want to talk about?"
    end

    ResponseType(content, tone_value(tone), 0.8, 0.7, false)
end

function adjust_tone(response::ResponseType, internal)
    if internal.stress_level > 0.5
        response = ResponseType(response.content, RESPONSE_TONE_DIRECT, 0.9, response.confidence, response.should_retry)
    elseif internal.affective_state == "warm"
        response = ResponseType(response.content, RESPONSE_TONE_WARM, 0.85, response.confidence, response.should_retry)
    end
    response
end

end # module
