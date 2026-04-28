# ============================================================================
# InstanceCommunication.jl — Multi-instance message passing
# v1.0: Instance discovery, message passing, shared state sync
# ============================================================================
module InstanceCommunication

using ..Types: HumanInput as HumanInputType, UserModel as UserModelType,
               InternalEmotional as InternalEmotionalType, SelfModel as SelfModelType
using Dates

export InstanceMessage, InstanceRegistry, register_instance, unregister_instance,
       send_message, broadcast_to_all, sync_state, get_active_instances

const MESSAGE_TYPES = [:ping, :pong, :state_sync, :prediction_share, :memory_share,
                       :emotional_contagion, :shutdown]

struct InstanceMessage
    from_instance_id::String
    to_instance_id::Union{String,Nothing}
    message_type::Symbol
    payload::Dict{String,Any}
    timestamp::String
    priority::Int64
end

InstanceMessage(from::String, to::Union{String,Nothing}, msg_type::Symbol, payload::Dict{String,Any}) =
    InstanceMessage(from, to, msg_type, payload, string(now()), 0)

struct InstanceRegistry
    instance_id::String
    instances::Dict{String,Dict{String,Any}}
    message_queue::Vector{InstanceMessage}
    is_leader::Bool
end

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
    false
)

function register_instance(
    registry::InstanceRegistry,
    instance_id::String;
    metadata::Dict{String,Any}=Dict{String,Any}()
)::Bool
    if haskey(registry.instances, instance_id)
        return false
    end

    registry.instances[instance_id] = Dict{String,Any}(
        "instance_id" => instance_id,
        "registered_at" => string(now()),
        "last_seen" => string(now()),
        "is_leader" => false,
        "metadata" => metadata
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
    collect(values(registry.instances))
end

function get_instance(registry::InstanceRegistry, instance_id::String)::Union{Dict{String,Any},Nothing}
    get(registry.instances, instance_id, nothing)
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

end # module InstanceCommunication