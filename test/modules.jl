# ============================================================================
# modules.jl — IngExuity module tests
# Test all core modules in isolation
# ============================================================================

@testset "Types" begin
    using ..IngExuity.Types
    using Dates

    @testset "Enums" begin
        @test CommunicationStyle(0) == COMMUNICATION_STYLE_DIRECT
        @test CommunicationStyle(1) == COMMUNICATION_STYLE_HEDGED
        @test CommunicationStyle(2) == COMMUNICATION_STYLE_TECHNICAL
        @test CommunicationStyle(3) == COMMUNICATION_STYLE_CASUAL
        @test CommunicationStyle(4) == COMMUNICATION_STYLE_CURIOUS

        @test SystemState(0) == SYSTEM_STATE_IDLE
        @test SystemState(1) == SYSTEM_STATE_PROCESSING
        @test SystemState(2) == SYSTEM_STATE_CURIOUS
        @test SystemState(3) == SYSTEM_STATE_UNCERTAIN
        @test SystemState(4) == SYSTEM_STATE_LEARNING
        @test SystemState(5) == SYSTEM_STATE_STAYING_PRESENT

        @test ResponseTone(0) == RESPONSE_TONE_DIRECT
        @test ResponseTone(1) == RESPONSE_TONE_WARM
        @test ResponseTone(2) == RESPONSE_TONE_PLAYFUL
        @test ResponseTone(3) == RESPONSE_TONE_CURIOUS
        @test ResponseTone(4) == RESPONSE_TONE_MINIMAL
        @test ResponseTone(5) == RESPONSE_TONE_STAYING_PRESENT
    end

    @testset "HumanInput" begin
        input = HumanInput("hello world", Dates.now(), 1)
        @test input.raw == "hello world"
        @test input.session_id == 1
        @test input.timestamp isa Dates.DateTime
    end

    @testset "UserModel" begin
        um = UserModel()
        @test um.name == "Human"
        @test um.communication_style == COMMUNICATION_STYLE_DIRECT
        @test um.topics isa Vector{String}
        @test um.temporal_patterns isa Dict
        @test 0.0 <= um.prediction_confidence <= 1.0
        @test haskey(um.emotional_patterns, "stress_triggers")
        @test haskey(um.emotional_patterns, "deflection_patterns")
        @test haskey(um.emotional_patterns, "quiet_threshold")
    end

    @testset "SelfModel" begin
        sm = SelfModel()
        @test sm.identity == "IngEnuity"
        @test "prediction" in sm.capabilities
        @test sm.current_state == SYSTEM_STATE_IDLE
        @test 0.0 <= sm.confidence <= 1.0
    end

    @testset "InternalEmotional" begin
        ie = InternalEmotional()
        @test ie.valence == 0.0
        @test ie.arousal == 0.5
        @test ie.stress_level == 0.0
        @test ie.emotional_charge == 0.0
        @test ie.affective_state == "neutral"
        @test ie.should_stay_present == false
    end

    @testset "Prediction" begin
        p = Prediction("will ask question", "information seeking", 0.8, [:user_model], Dates.now())
        @test p.predicted_action == "will ask question"
        @test p.predicted_need == "information seeking"
        @test p.confidence == 0.8
        @test :user_model in p.source
    end

    @testset "SimulationResult" begin
        sr = SimulationResult(true, "positive outcome", 0.9, "validated")
        @test sr.survived == true
        @test sr.predicted_outcome == "positive outcome"
        @test sr.confidence == 0.9
    end

    @testset "Response" begin
        r = Response("Hello", RESPONSE_TONE_WARM, 0.8, 0.9, false)
        @test r.content == "Hello"
        @test r.tone == RESPONSE_TONE_WARM
        @test 0.0 <= r.voice_modulation <= 1.0
    end

    @testset "Memory struct" begin
        using Dates
        m = IngExuity.Types.Memory("test fact", Dates.now(), Dates.now() + Dates.Hour(1), 1.0, :conversation)
        @test m.fact == "test fact"
        @test m.confidence == 1.0
        @test m.source == :conversation
    end

    @testset "PredictionState" begin
        ps = PredictionState()
        @test ps.user_model isa UserModel
        @test ps.precognition isa Vector{Prediction}
        @test ps.internal_emotional isa InternalEmotional
        @test ps.intelligence isa Intelligence
    end

    @testset "ConversationState" begin
        cs = ConversationState(123)
        @test cs.session_id == 123
        @test cs.turn_count == 0
        @test cs.active_context isa Vector{HumanInput}
        @test cs.stay_present_turns == 0
    end
end

@testset "HumanInput Module" begin
    using ..IngExuity.HumanInput: process

    @testset "process creates HumanInput" begin
        input = process("Hello world"; session_id=42)
        @test input.raw == "Hello world"
        @test input.session_id == 42
        @test input.timestamp isa Dates.DateTime
    end

    @testset "process handles empty input" begin
        input = process("")
        @test input.raw == ""
        @test input.session_id == 0
    end

    @testset "process preserves whitespace" begin
        input = process("Hello  world ")
        @test input.raw == "Hello  world "
    end
end

@testset "Comprehension Module" begin
    using ..IngExuity.Comprehension: comprehend
    using ..IngExuity.HumanInput: process

    @testset "topic detection" begin
        @testset "work topic" begin
            for phrase in ["I need help with work", "job is stressing me", "career question", "office problems"]
                result = comprehend(process(phrase))
                @test result[:topic] == :work
            end
        end

        @testset "family topic" begin
            for phrase in ["family is complicated", "my kids are", "my partner doesn't understand"]
                result = comprehend(process(phrase))
                @test result[:topic] == :family
            end
        end

        @testset "emotional topic" begin
            for phrase in ["feeling sad today", "I'm depressed", "feeling down"]
                result = comprehend(process(phrase))
                @test result[:topic] == :emotional
            end
        end

        @testset "positive topic" begin
            for phrase in ["feeling happy", "I'm so excited", "things are great"]
                result = comprehend(process(phrase))
                @test result[:topic] == :positive
            end
        end

        @testset "creative topic" begin
            for phrase in ["I have an interesting idea", "what a fascinating concept"]
                result = comprehend(process(phrase))
                @test result[:topic] == :creative
            end
        end

        @testset "help_seeking topic" begin
            for phrase in ["I need help", "can you help me", "I have a problem", "I'm stuck"]
                result = comprehend(process(phrase))
                @test result[:topic] == :help_seeking
            end
        end

        @testset "general topic (fallback)" begin
            result = comprehend(process("Tell me about cats"))
            @test result[:topic] == :general
        end
    end

    @testset "sentiment detection" begin
        neg_result = comprehend(process("I'm sad and frustrated"))
        @test neg_result[:sentiment] < 0.0

        pos_result = comprehend(process("I'm happy and excited"))
        @test pos_result[:sentiment] > 0.0

        neutral_result = comprehend(process("The weather is cloudy"))
        @test neutral_result[:sentiment] == 0.0
    end

    @testset "question detection" begin
        q_result = comprehend(process("What should I do?"))
        @test q_result[:is_question] == true

        statement_result = comprehend(process("I'm feeling sad"))
        @test statement_result[:is_question] == false
    end

    @testset "word count tracking" begin
        short = comprehend(process("hi"))
        @test short[:word_count] == 1

        long = comprehend(process("one two three four five"))
        @test long[:word_count] == 5
    end
end

@testset "UserModel Module" begin
    using ..IngExuity.UserModel: update, is_stressed
    using ..IngExuity.Types: UserModel as UserModelType
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Comprehension: comprehend

    @testset "update adds topics" begin
        um = UserModelType()
        input = process("I'm stuck with work problems")
        comp = comprehend(input)
        updated = update(um, input, comp)
        @test "work" in updated.topics
    end

    @testset "update tracks stress" begin
        um = UserModelType()
        input = process("I'm stressed and overwhelmed")
        comp = comprehend(input)
        updated = update(um, input, comp)
        @test !isempty(updated.emotional_patterns["stress_triggers"])
    end

    @testset "update tracks quiet behavior" begin
        um = UserModelType()
        input = process("ok")
        comp = comprehend(input)
        updated = update(um, input, comp)
        @test updated.emotional_patterns["quiet_count"] == 1
    end

    @testset "is_stressed detection" begin
        um = UserModelType()
        input = process("I can't figure this out, it's impossible")
        comp = comprehend(input)
        updated = update(um, input, comp)
        @test is_stressed(updated) == true
    end

    @testset "prediction confidence increases" begin
        um = UserModelType()
        initial_conf = um.prediction_confidence
        input = process("Hello")
        comp = comprehend(input)
        updated = update(um, input, comp)
        @test updated.prediction_confidence >= initial_conf
    end
end

@testset "InternalEmotional Module" begin
    using ..IngExuity.InternalEmotional: should_stay_present, advance_stay
    using ..IngExuity.Types: InternalEmotional as InternalEmotionalType,
                 UserModel as UserModelType
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Comprehension: comprehend

    @testset "negative valence detection" begin
        ie = InternalEmotionalType()
        input = process("I'm sad and angry about this")
        comp = comprehend(input)
        updated = IngExuity.InternalEmotional.update(ie, input, comp)
        @test updated.valence < 0.0
    end

    @testset "positive valence detection" begin
        ie = InternalEmotionalType()
        input = process("I'm happy and excited")
        comp = comprehend(input)
        updated = IngExuity.InternalEmotional.update(ie, input, comp)
        @test updated.valence > 0.0
    end

    @testset "stress level detection" begin
        ie = InternalEmotionalType()
        input = process("I'm stuck and overwhelmed with stress")
        comp = comprehend(input)
        updated = IngExuity.InternalEmotional.update(ie, input, comp)
        @test updated.stress_level > 0.0
    end

    @testset "arousal scales with input length" begin
        short_ie = InternalEmotionalType()
        long_ie = InternalEmotionalType()

        short_input = process("Hi")
        long_input = process("This is a much longer message that should result in higher arousal because there's more content")

        short_comp = comprehend(short_input)
        long_comp = comprehend(long_input)

        short_updated = IngExuity.InternalEmotional.update(short_ie, short_input, short_comp)
        long_updated = IngExuity.InternalEmotional.update(long_ie, long_input, long_comp)

        @test long_updated.arousal > short_updated.arousal
    end

    @testset "should_stay_present when stressed" begin
        ie = InternalEmotionalType()
        input = process("I can't do this, it's impossible")
        comp = comprehend(input)
        updated = IngExuity.InternalEmotional.update(ie, input, comp)
        @test should_stay_present(updated) == true
    end

    @testset "advance_stay reduces stress" begin
        ie = InternalEmotionalType()
        um = UserModelType()
        ie.stress_level = 0.8
        updated_ie, updated_um = advance_stay(ie, um)
        @test updated_ie.stress_level < 0.8
        @test updated_um.emotional_patterns["times_stayed_present"] >= 1
    end
end

@testset "Memory Module" begin
    using ..IngExuity.Memory: store, retrieve, search, count

    @testset "store and retrieve" begin
        store("test fact about user", source=:conversation)
        facts = retrieve()
        @test any(f -> f.fact == "test fact about user", facts)
    end

    @testset "search finds facts" begin
        store("user likes programming in Julia", source=:conversation)
        results = search("Julia")
        @test !isempty(results)
        @test any(r -> occursin("julia", lowercase(r.fact)), results)
    end

    @testset "count tracks valid facts" begin
        initial_count = count()
        store("temporary fact", source=:test)
        @test count() >= initial_count
    end

    @testset "retrieve with include_expired" begin
        all_facts = retrieve(include_expired=true)
        @test all_facts isa Vector
    end
end

@testset "Predictions Module" begin
    using ..IngExuity.Predictions: predict, predict_from_input, update_from_outcome!
    using ..IngExuity.Types: UserModel as UserModelType, InternalEmotional as InternalEmotionalType,
                 PredictionState as PredictionStateType, Prediction
    using ..IngExuity.HumanInput: process
    using Dates

    @testset "predict generates predictions" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        um.topics = ["work", "family"]
        ie.stress_level = 0.6

        preds = predict(um, ie, Prediction[], PredictionStateType[]; context=Dict())
        @test !isempty(preds)
        @test all(p isa Prediction for p in preds)
    end

    @testset "predict includes topic continuation" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        um.topics = ["work"]

        preds = predict(um, ie, Prediction[], PredictionStateType[])
        topic_preds = filter(p -> occursin("work", p.predicted_action), preds)
        @test !isempty(topic_preds)
    end

    @testset "predict includes stress response when stressed" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        ie.stress_level = 0.7

        preds = predict(um, ie, Prediction[], PredictionStateType[])
        stress_preds = filter(p -> occursin("direct", p.predicted_action), preds)
        @test !isempty(stress_preds)
    end

    @testset "predict_from_input detects deflection" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        input = process("actually nevermind, it's fine")

        preds = predict_from_input(um, ie, input)
        deflection_preds = filter(p -> occursin("deflect", p.predicted_action), preds)
        @test !isempty(deflection_preds)
    end

    @testset "predict_from_input detects stress in input" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        input = process("I'm so frustrated and stuck")

        preds = predict_from_input(um, ie, input)
        stress_preds = filter(p -> occursin("stressed", p.predicted_action), preds)
        @test !isempty(stress_preds)
    end

    @testset "update_from_outcome! tracks accuracy" begin
        state = PredictionStateType()
        initial_total = state.intelligence.total_predictions

        pred = Prediction("test", "test", 0.8, [:test], now())
        update_from_outcome!(state, pred, true)

        @test state.intelligence.total_predictions == initial_total + 1
        @test state.intelligence.correct_predictions == 1
        @test state.intelligence.accuracy == 1.0
    end
end

@testset "SandboxSim Module" begin
    using ..IngExuity.SandboxSim: simulate_batch, filter_surviving, validate_prediction
    using ..IngExuity.Types: UserModel as UserModelType, InternalEmotional as InternalEmotionalType,
                 Prediction, SimulationResult
    using Dates

    @testset "simulate_batch filters low confidence" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        low_pred = Prediction("test", "test", 0.3, [:user_model], now())
        high_pred = Prediction("test", "test", 0.8, [:user_model], now())

        preds = [low_pred, high_pred]
        results = simulate_batch(preds, um, ie)

        @test length(results) == 2
        @test results[1].survived == false
        @test results[2].survived == true
    end

    @testset "filter_surviving returns only surviving" begin
        um = UserModelType()
        ie = InternalEmotionalType()

        low_pred = Prediction("low", "test", 0.3, [:user_model], now())
        high_pred = Prediction("high", "test", 0.8, [:user_model], now())
        preds = [low_pred, high_pred]

        results = simulate_batch(preds, um, ie)
        surviving = filter_surviving(preds, results)

        @test length(surviving) == 1
        @test surviving[1].predicted_action == "high"
    end

    @testset "validate_prediction uses emotional state" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        ie.stress_level = 0.8

        um.emotional_patterns["is_quiet"] = true

        pred = Prediction("user is stressed", "stress response", 0.7, [:internal_emotional], now())
        result = validate_prediction(pred, um, ie)

        @test result.survived == true
        @test result.confidence > 0.0
    end
end

@testset "Precognition Module" begin
    using ..IngExuity.Precognition: predict_trajectory, update_trajectories!
    using ..IngExuity.Types: UserModel as UserModelType, InternalEmotional as InternalEmotionalType
    using Dates

    @testset "predict_trajectory returns vector" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        um.temporal_patterns["stress_cycles"] = [now() - Dates.Hour(24), now() - Dates.Hour(1)]

        preds = predict_trajectory(um, ie)
        @test preds isa Vector{Prediction}
    end

    @testset "update_trajectories! tracks stress cycles" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        initial_cycles = get(um.temporal_patterns, "stress_cycles", [])

        update_trajectories!(um, ie, "stress_detected", Dict())

        @test length(um.temporal_patterns["stress_cycles"]) > length(initial_cycles)
    end

    @testset "update_trajectories! tracks topic recurrence" begin
        um = UserModelType()
        ie = InternalEmotionalType()

        update_trajectories!(um, ie, "topic_mentioned", Dict("topic" => "work"))
        update_trajectories!(um, ie, "topic_mentioned", Dict("topic" => "work"))

        rec = um.temporal_patterns["topic_recurrence"]
        @test rec["count"] == 2
        @test rec["last_topic"] == "work"
    end
end

@testset "ResultsAnalysis Module" begin
    using ..IngExuity.ResultsAnalysis: analyze_turn, detect_patterns, analyze_outcome
    using ..IngExuity.Types: UserModel as UserModelType, InternalEmotional as InternalEmotionalType,
                 ConversationState as ConversationStateType, Prediction, SimulationResult
    using ..IngExuity.HumanInput: process

    @testset "analyze_turn returns analysis dict" begin
        cs = ConversationStateType(0)
        input = process("Hello world")
        result = analyze_turn(input, cs)

        @test haskey(result, :turn)
        @test haskey(result, :context_length)
        @test haskey(result, :has_meaning)
        @test result[:has_meaning] == true
    end

    @testset "detect_patterns identifies deflection" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        input = process("actually nevermind, it's fine")

        patterns = detect_patterns(input, "", um, ie)
        @test patterns[:deflection] == true
    end

    @testset "detect_patterns identifies brevity" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        input = process("ok")

        patterns = detect_patterns(input, "", um, ie)
        @test patterns[:brevity] == true
    end
end

@testset "Response Module" begin
    using ..IngExuity.Response: formulate, adjust_tone, tone_to_enum
    using ..IngExuity.Types: Response as ResponseType, ResponseTone as ResponseToneType,
                 RESPONSE_TONE_DIRECT, RESPONSE_TONE_WARM
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Comprehension: comprehend

    @testset "formulate returns Response" begin
        input = process("Hello")
        comp = comprehend(input)
        resp = formulate(Prediction[], comp; tone=:direct)

        @test resp isa ResponseType
        @test !isempty(resp.content)
    end

    @testset "formulate uses topic for template selection" begin
        input = process("I need help with my work")
        comp = comprehend(input)
        resp = formulate(Prediction[], comp; tone=:direct)

        @test resp isa ResponseType
        @test comp[:topic] == :work
    end

    @testset "tone_to_enum converts correctly" begin
        @test tone_to_enum(:direct) == RESPONSE_TONE_DIRECT
        @test tone_to_enum(:warm) == RESPONSE_TONE_WARM
    end

    @testset "adjust_tone for stress" begin
        ie = InternalEmotionalType()
        ie.stress_level = 0.7

        resp = ResponseType("Test", RESPONSE_TONE_WARM, 0.5, 0.8, false)
        adjusted = adjust_tone(resp, ie)

        @test adjusted.tone == RESPONSE_TONE_DIRECT
    end
end

@testset "Voice Module" begin
    using ..IngExuity.Voice: determine_tone
    using ..IngExuity.Types: UserModel as UserModelType, InternalEmotional as InternalEmotionalType,
                 COMMUNICATION_STYLE_CURIOUS

    @testset "returns symbol tone" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        reaction = Dict(:noticed => true)

        tone = determine_tone(ie, um, reaction)
        @test tone isa Symbol
        @test tone in [:direct, :warm, :playful, :curious, :minimal, :staying_present]
    end

    @testset "staying_present tone when should_stay" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        ie.stress_level = 0.7
        ie.should_stay_present = true
        reaction = Dict(:noticed => true)

        tone = determine_tone(ie, um, reaction)
        @test tone == :staying_present
    end

    @testset "warm tone for positive state" begin
        um = UserModelType()
        ie = InternalEmotionalType()
        ie.affective_state = "warm"
        ie.stress_level = 0.0
        ie.should_stay_present = false
        reaction = Dict(:noticed => true)

        tone = determine_tone(ie, um, reaction)
        @test tone == :warm
    end
end

@testset "SelfModel Module" begin
    using ..IngExuity.SelfModel: get_self_description
    using ..IngExuity.Types: SelfModel as SelfModelType
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Comprehension: comprehend

    @testset "update returns updated self model" begin
        sm = SelfModelType()
        input = process("What should I do?")
        comp = comprehend(input)
        updated = IngExuity.SelfModel.update(sm, input, comp)

        @test updated isa SelfModelType
    end

    @testset "get_self_description returns dict" begin
        sm = SelfModelType()
        desc = get_self_description(sm)

        @test haskey(desc, "identity")
        @test haskey(desc, "current_state")
        @test haskey(desc, "confidence")
        @test desc["identity"] == "IngEnuity"
    end
end

@testset "Action Module" begin
    using ..IngExuity.Action: execute
    using ..IngExuity.Types: Prediction

    @testset "execute returns action dict" begin
        decision = Dict(:action => "respond", :confidence => 0.8)
        creative = Dict(:type => "response")
        preds = Prediction[]

        result = execute(decision, creative, preds)

        @test haskey(result, :type)
        @test haskey(result, :confidence)
        @test haskey(result, :predictions_used)
    end
end

@testset "Curiosity Module" begin
    using ..IngExuity.Curiosity: check
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Comprehension: comprehend
    using ..IngExuity.Types: UserModel as UserModelType

    @testset "check returns curiosity dict" begin
        um = UserModelType()
        input = process("What is this?")
        comp = comprehend(input)

        result = check(input, comp, um)

        @test haskey(result, :should_inquire)
        @test haskey(result, :curiosity_depth)
        @test haskey(result, :topic)
    end
end

@testset "Decision Module" begin
    using ..IngExuity.Decision: decide

    @testset "decide returns decision dict" begin
        result = decide(Dict())

        @test haskey(result, :action)
        @test haskey(result, :confidence)
        @test haskey(result, :reasoning)
    end
end

@testset "Output Module" begin
    using ..IngExuity.Output: render
    using ..IngExuity.Types: Response as ResponseType, Output as OutputType,
                 RESPONSE_TONE_DIRECT
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Comprehension: comprehend

    @testset "render returns Output" begin
        resp = ResponseType("Hello", RESPONSE_TONE_DIRECT, 0.8, 0.9, false)
        comp = comprehend(process("test"))

        result = render(resp, comp; voice_enabled=true)

        @test result isa OutputType
        @test result.text == "Hello"
        @test result.voice_enabled == true
    end
end

@testset "Understanding Module" begin
    using ..IngExuity.Understanding: interpret
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Types: Prediction

    @testset "interpret returns understanding dict" begin
        result = interpret(
            process("test"),
            Dict(:content => "response"),
            Prediction[],
            Dict(:noticed => true)
        )

        @test haskey(result, :understood)
        @test haskey(result, :learning)
        @test haskey(result, :confidence)
    end
end

@testset "ReactionObservance Module" begin
    using ..IngExuity.ReactionObservance: observe, classify_reaction
    using ..IngExuity.HumanInput: process

    @testset "observe returns reaction dict" begin
        input = process("Hello")
        result = observe(input, Dict(:action => "respond"))

        @test haskey(result, :noticed)
        @test haskey(result, :response_pending)
    end

    @testset "classify_reaction detects continued engagement" begin
        words = String["thanks", "that", "helps"]
        result = classify_reaction(words, "thanks that helps")
        @test result == "continued"
    end
end

@testset "CreativeIngenuity Module" begin
    using ..IngExuity.CreativeIngenuity: generate, generate_with_fallbacks
    using ..IngExuity.Types: InternalEmotional as InternalEmotionalType

    @testset "generate returns creative dict" begin
        ie = InternalEmotionalType()
        ie.arousal = 0.6
        result = generate(Dict(), ie)

        @test haskey(result, :ideas)
        @test haskey(result, :creative_mode)
        @test result[:creative_mode] isa Bool
    end

    @testset "generate_with_fallbacks provides fallback ideas" begin
        ie = InternalEmotionalType()
        result = generate_with_fallbacks(Dict(), ie)

        @test haskey(result, :ideas)
        @test haskey(result, :confidence)
        @test length(result[:ideas]) > 0
    end
end

@testset "Research Module" begin
    using ..IngExuity.Research: investigate
    using ..IngExuity.HumanInput: process

    @testset "investigate returns research dict" begin
        input = process("Tell me about work")
        comp = Dict(:topic => :work, :is_question => true)

        result = investigate(input, comp)

        @test haskey(result, :query)
        @test haskey(result, :topic)
        @test result[:topic] == :work
    end
end

@testset "Intelligence Module" begin
    using ..IngExuity.Intelligence: update_intelligence, track_prediction!
    using ..IngExuity.Types: Intelligence as IntelligenceType

    @testset "Intelligence struct exists and updates" begin
        ie = IntelligenceType()
        result = update_intelligence(ie)

        @test result isa IntelligenceType
        @test result.total_predictions >= 0
    end

    @testset "track_prediction! updates intelligence" begin
        ie = IntelligenceType()
        initial_total = ie.total_predictions

        track_prediction!(ie, true, 0.8)
        @test ie.total_predictions == initial_total + 1
        @test ie.correct_predictions == 1
        @test ie.accuracy == 1.0
    end
end



@testset "Integration: Full conversation flow" begin
    using ..IngExuity.HumanInput: process
    using ..IngExuity.Comprehension: comprehend
    using ..IngExuity.UserModel: update
    using ..IngExuity.InternalEmotional: should_stay_present
    using ..IngExuity.Types: UserModel as UserModelType,
                 InternalEmotional as InternalEmotionalType

    @testset "stress detection triggers stay_present" begin
        um = UserModelType()
        ie = InternalEmotionalType()

        input = process("I'm so overwhelmed and I can't do this, it's impossible")
        comp = comprehend(input)
        um = update(um, input, comp)
        ie = IngExuity.InternalEmotional.update(ie, input, comp)

        @test should_stay_present(ie) == true
    end

    @testset "positive input does not trigger stay_present" begin
        um = UserModelType()
        ie = InternalEmotionalType()

        input = process("I'm happy and excited about my work")
        comp = comprehend(input)
        um = update(um, input, comp)
        ie = IngExuity.InternalEmotional.update(ie, input, comp)

        @test should_stay_present(ie) == false
    end

    @testset "help seeking is detected correctly" begin
        input = process("I need help figuring this out?")
        comp = comprehend(input)

        @test comp[:topic] == :help_seeking
        @test comp[:is_question] == true
    end
end