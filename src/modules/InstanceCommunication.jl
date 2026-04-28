# ============================================================================
# InstanceCommunication.jl — Multi-instance message passing
# v2.0: Instance discovery, message passing, shared state sync
# Each instance is a distinct entity unless connected by user
# ============================================================================
module InstanceCommunication

using ..Types: HumanInput as HumanInputType, UserModel as UserModelType,
               InternalEmotional as InternalEmotionalType, SelfModel as SelfModelType
using Dates

export InstanceMessage, InstanceRegistry, register_instance, unregister_instance,
       send_message, broadcast_to_all, sync_state, get_active_instances,
       get_messages_for_instance, acknowledge_messages!, create_sync_payload

const MESSAGE_TYPES = [:ping, :pong, :state_sync, :prediction_share, :memory_share,
                       :emotional_contagion, :shutdown, :identity_transfer]

struct InstanceMessage
    from_instance_id::String
    to_instance_id::Union{String,Nothing}
    message_type::Symbol
    payload::Dict{String,Any}
    timestamp::String
    priority::Int64
    acknowledged::Bool
end

InstanceMessage(from::String, to::Union{String,Nothing}, msg_type::Symbol, payload::Dict{String,Any}) =
    InstanceMessage(from, to, msg_type, payload, string(now()), 0, false)

struct InstanceRegistry
    instance_id::String
    instances::Dict{String,Dict{String,Any}}
    message_queue::Vector{InstanceMessage}
    is_leader::Bool
    connected_instances::Set{String}
end

InstanceRegistry(id::String, inst::Dict, queue::Vector, leader::Bool) =
    InstanceRegistry(id, inst, queue, leader, Set{String}())

function uuid4()::String
    bytes = rand(UInt8, 16)
    bytes[7] = (bytes[7] & 0x0f) | 0x40
    bytes[9] = (bytes[9] & 0x3f) | 0x80
    hex_chars = Vector{Char}(undef, 32)
    for i in 1:16
        hex_chars[(i-1)*2+1] = "0123456789abcdef"[bytes[i]>>4+1]
        hex_chars[(i-1)*2+2] = "0123456789abcdef"[bytes[i]&0x0f+1]
    end
    String(hex_chars)
end

const GLOBAL_REGISTRY = InstanceRegistry(
    uuid4(),
    Dict{String,Dict{String,Any}}(),
    InstanceMessage[],
    false,
    Set{String}()
)

function register_instance(
    registry::InstanceRegistry,
    instance_id::String;
    metadata::Dict{String,Any}=Dict{String,Any}()
)::Bool
    if haskey(registry.instances, instance_id)
        registry.instances[instance_id]["last_seen"] = string(now())
        return true
    end

    registry.instances[instance_id] = Dict{String,Any}(
        "instance_id" => instance_id,
        "registered_at" => string(now()),
        "last_seen" => string(now()),
        "is_leader" => false,
        "metadata" => metadata,
        "connected_to" => String[]
    )

    if isempty(registry.instances) || registry.is_leader
        registry.is_leader = instance_id == registry.instance_id
    end

    true
end

function unregister_instance(registry::InstanceRegistry, instance_id::String)::Bool
    if !haskey(registry.instances, instance_id)
        return false
    end

    delete!(registry.instances, instance_id)
    delete!(registry.connected_instances, instance_id)
    true
end

function connect_instances!(registry::InstanceRegistry, instance_id_1::String, instance_id_2::String)::Bool
    if !haskey(registry.instances, instance_id_1) || !haskey(registry.instances, instance_id_2)
        return false
    end

    push!(registry.connected_instances, instance_id_1)
    push!(registry.connected_instances, instance_id_2)

    registry.instances[instance_id_1]["connected_to"] = push!(
        get(registry.instances[instance_id_1], "connected_to", String[]),
        instance_id_2
    )
    registry.instances[instance_id_2]["connected_to"] = push!(
        get(registry.instances[instance_id_2], "connected_to", String[]),
        instance_id_1
    )

    true
end

function disconnect_instances!(registry::InstanceRegistry, instance_id_1::String, instance_id_2::String)::Bool
    if !haskey(registry.instances, instance_id_1) || !haskey(registry.instances, instance_id_2)
        return false
    end

    delete!(registry.connected_instances, instance_id_1)
    delete!(registry.connected_instances, instance_id_2)

    registry.instances[instance_id_1]["connected_to"] = filter!(
        x -> x != instance_id_2,
        get(registry.instances[instance_id_1], "connected_to", String[])
    )
    registry.instances[instance_id_2]["connected_to"] = filter!(
        x -> x != instance_id_1,
        get(registry.instances[instance_id_2], "connected_to", String[])
    )

    true
end

function send_message(
    registry::InstanceRegistry,
    message::InstanceMessage
)::Bool
    if message.to_instance_id !== nothing &&
       !haskey(registry.instances, message.to_instance_id)
        return false
    end

    push!(registry.message_queue, message)
    true
end

function broadcast_to_all(
    registry::InstanceRegistry,
    from_instance_id::String,
    message_type::Symbol,
    payload::Dict{String,Any}
)::Int64
    count = 0
    for instance_id in keys(registry.instances)
        if instance_id != from_instance_id
            msg = InstanceMessage(from_instance_id, instance_id, message_type, payload)
            if send_message(registry, msg)
                count += 1
            end
        end
    end
    count
end

function broadcast_to_connected(
    registry::InstanceRegistry,
    from_instance_id::String,
    message_type::Symbol,
    payload::Dict{String,Any}
)::Int64
    count = 0
    connected = get(registry.instances[from_instance_id], "connected_to", String[])

    for target_id in connected
        if haskey(registry.instances, target_id)
            msg = InstanceMessage(from_instance_id, target_id, message_type, payload)
            if send_message(registry, msg)
                count += 1
            end
        end
    end
    count
end

function sync_state(
    registry::InstanceRegistry,
    instance_id::String,
    state::Dict{String,Any}
)::Bool
    if !haskey(registry.instances, instance_id)
        return false
    end

    registry.instances[instance_id]["state"] = state
    registry.instances[instance_id]["last_seen"] = string(now())
    true
end

function get_active_instances(registry::InstanceRegistry)::Vector{Dict{String,Any}}
    instances = []
    for (id, info) in registry.instances
        push!(instances, Dict(
            "instance_id" => id,
            "registered_at" => info["registered_at"],
            "last_seen" => info["last_seen"],
            "is_leader" => info["is_leader"],
            "connected_to" => get(info, "connected_to", String[]),
            "is_self" => id == registry.instance_id
        ))
    end
    instances
end

function get_instance(registry::InstanceRegistry, instance_id::String)::Union{Dict{String,Any},Nothing}
    get(registry.instances, instance_id, nothing)
end

function get_messages_for_instance(
    registry::InstanceRegistry,
    instance_id::String;
    include_acknowledged::Bool=false
)::Vector{InstanceMessage}
    msgs = []
    for msg in registry.message_queue
        if msg.to_instance_id == instance_id
            if include_acknowledged || !msg.acknowledged
                push!(msgs, msg)
            end
        end
    end
    msgs
end

function acknowledge_messages!(
    registry::InstanceRegistry,
    instance_id::String,
    message_ids::Vector{Int64}
)::Int64
    count = 0
    for msg in registry.message_queue
        if msg.to_instance_id == instance_id && msg.timestamp in message_ids
            msg.acknowledged = true
            count += 1
        end
    end
    count
end

function create_sync_payload(
    self_model,
    internal,
    intelligence,
    user_model,
    memory_count::Int64
)::Dict{String,Any}
    Dict{String,Any}(
        "self_model" => Dict(
            "identity" => self_model.identity,
            "confidence" => self_model.confidence,
            "current_state" => string(self_model.current_state)
        ),
        "internal" => Dict(
            "valence" => internal.valence,
            "arousal" => internal.arousal,
            "stress_level" => internal.stress_level,
            "affective_state" => internal.affective_state
        ),
        "intelligence" => Dict(
            "accuracy" => intelligence.accuracy,
            "total_predictions" => intelligence.total_predictions
        ),
        "user_model" => Dict(
            "topics" => user_model.topics[max(1, end-5):end],
            "prediction_confidence" => user_model.prediction_confidence
        ),
        "memory_count" => memory_count,
        "timestamp" => string(now())
    )
end

function elect_leader(registry::InstanceRegistry)::String
    isempty(registry.instances) && return registry.instance_id

    candidates = [id for (id, info) in registry.instances if info["is_leader"]]
    if !isempty(candidates)
        leader_id = candidates[1]
        registry.is_leader = true
        return leader_id
    end

    registered_times = [(id, info["registered_at"]) for (id, info) in registry.instances]
    sort!(registered_times, by=x->x[2])
    leader_id = registered_times[1][1]

    for (id, info) in registry.instances
        info["is_leader"] = (id == leader_id)
    end
    registry.is_leader = (leader_id == registry.instance_id)

    leader_id
end

function relay_message(registry::InstanceRegistry, target_instance_id::String)::Bool
    toRelay = [msg for msg in registry.message_queue if msg.to_instance_id == target_instance_id]

    for msg in toRelay
        push!(registry.message_queue, msg)
    end

    filter!(msg -> msg.to_instance_id != target_instance_id, registry.message_queue)
    !isempty(toRelay)
end

function clear_message_queue!(registry::InstanceRegistry)::Int64
    count = length(registry.message_queue)
    registry.message_queue = InstanceMessage[]
    count
end

function cleanup_stale_instances!(registry::InstanceRegistry; stale_minutes::Int=30)::Int64
    cutoff = now() - Dates.Minute(stale_minutes)
    stale_ids = String[]

    for (id, info) in registry.instances
        try
            last_seen = parse(DateTime, info["last_seen"])
        catch
            last_seen = now()
        end
        if last_seen < cutoff
            push!(stale_ids, id)
        end
    end

    for id in stale_ids
        unregister_instance(registry, id)
    end

    length(stale_ids)
end

end # module InstanceCommunication