# ============================================================================
# Response.jl — Shape and formulate the response
# ============================================================================
module Response

export formulate, adjust_tone

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

    Response(content, ResponseTone(tone), 0.8, 0.7, false)
end

function adjust_tone(response::Response, internal)
    if internal.stress_level > 0.5
        response = Response(response.content, ResponseTone(:direct), 0.9, response.confidence, response.should_retry)
    elseif internal.affective_state == "warm"
        response = Response(response.content, ResponseTone(:warm), 0.85, response.confidence, response.should_retry)
    end
    response
end

end # module
