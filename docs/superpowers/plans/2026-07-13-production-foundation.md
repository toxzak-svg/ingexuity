# IngExuity Production Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a tested production foundation for hardware-aware local inference, relationship modes, reversible adaptive growth, permission tiers, and a status API without disrupting the existing 187-test conversation pipeline.

**Architecture:** Add four focused Julia modules behind stable interfaces: hardware profiling, companion state, version history, and action permissions. Wire a single foundation state into the existing package entry point and expose it read-only through `/api/foundation`. Keep model research and persistent encryption out of this slice; later plans will implement the model router, durable vault, desktop shell, and tools against these interfaces.

**Tech Stack:** Julia 1.12, Julia standard library (`Dates`, `Test`), existing `HTTP.jl` and `JSON.jl`, current hand-written HTTP server.

---

## File Structure

- Modify `Project.toml` — correct the `Test` standard-library UUID so `Pkg.test()` works.
- Create `src/foundation/RuntimeProfiles.jl` — classify hardware and describe model residency policy.
- Create `src/foundation/VersionHistory.jl` — generic in-memory checkpoints and rollback with defensive copies.
- Create `src/foundation/CompanionState.jl` — relationship modes, adaptive-growth toggle, and versioned behavior state.
- Create `src/foundation/Permissions.jl` — permission tiers, action risk, and deterministic authorization decisions.
- Modify `src/IngExuity.jl` — include/export foundations, initialize state, and add `/api/foundation`.
- Create `test/foundation/runtime_profiles.jl` — deterministic hardware classification tests.
- Create `test/foundation/version_history.jl` — checkpoint and rollback tests.
- Create `test/foundation/companion_state.jl` — mode and adaptive-growth tests.
- Create `test/foundation/permissions.jl` — authorization matrix tests.
- Create `test/foundation/integration.jl` — package exports and endpoint contract.
- Modify `test/runtests.jl` — include focused foundation test files.
- Modify `README.md` — replace the stale phase claim with the executable foundation status.
- Modify `plans/INGEXUITY_PHASED_BUILD_PLAN.md` — point future work at the approved production architecture and research promotion gate.

### Task 1: Restore the canonical package test path

**Files:**
- Modify: `Project.toml`

- [ ] **Step 1: Confirm the metadata failure**

Run:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

Expected: FAIL before test execution with `expected package Test [8f399da3] to be registered`.

- [ ] **Step 2: Correct the Test UUID**

Replace the current `[extras]` entry with:

```toml
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
```

- [ ] **Step 3: Verify both test entry points**

Run:

```powershell
julia --project=. test/runtests.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

Expected: both commands run the suite successfully with at least `187/187` existing tests passing.

- [ ] **Step 4: Commit**

```powershell
git add Project.toml
git commit -m "fix: restore Julia package test target"
```

### Task 2: Add deterministic hardware profiles

**Files:**
- Create: `src/foundation/RuntimeProfiles.jl`
- Create: `test/foundation/runtime_profiles.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing hardware-profile tests**

Create `test/foundation/runtime_profiles.jl`:

```julia
using .IngExuity.RuntimeProfiles

@testset "RuntimeProfiles" begin
    gib = 1024^3

    constrained = classify_hardware(8gib; cpu_threads=4)
    mainstream = classify_hardware(16gib; cpu_threads=8)
    enthusiast = classify_hardware(32gib; cpu_threads=12, gpu_backend=:cuda)
    workstation = classify_hardware(64gib; cpu_threads=24, gpu_backend=:cuda)

    @test constrained.tier == HARDWARE_CONSTRAINED
    @test mainstream.tier == HARDWARE_MAINSTREAM
    @test enthusiast.tier == HARDWARE_ENTHUSIAST
    @test workstation.tier == HARDWARE_WORKSTATION

    policy = model_policy(mainstream)
    @test policy["fast_model_resident"] == true
    @test policy["deliberate_model"] == "on_demand"
    @test policy["specialists"] == "sequential"

    @test_throws ArgumentError classify_hardware(0)
    @test_throws ArgumentError classify_hardware(16gib; gpu_backend=:unknown)
end
```

Add after the existing module tests in `test/runtests.jl`:

```julia
include(joinpath(@__DIR__, "foundation", "runtime_profiles.jl"))
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
julia --project=. test/runtests.jl
```

Expected: FAIL because `IngExuity.RuntimeProfiles` does not exist.

- [ ] **Step 3: Implement hardware classification and policy**

Create `src/foundation/RuntimeProfiles.jl`:

```julia
module RuntimeProfiles

export HardwareTier, RuntimeProfile,
       HARDWARE_CONSTRAINED, HARDWARE_MAINSTREAM,
       HARDWARE_ENTHUSIAST, HARDWARE_WORKSTATION,
       classify_hardware, detect_hardware, model_policy

@enum HardwareTier begin
    HARDWARE_CONSTRAINED
    HARDWARE_MAINSTREAM
    HARDWARE_ENTHUSIAST
    HARDWARE_WORKSTATION
end

const SUPPORTED_GPU_BACKENDS = Set([:cpu, :integrated, :cuda, :metal, :vulkan, :rocm])

struct RuntimeProfile
    memory_bytes::Int
    cpu_threads::Int
    gpu_backend::Symbol
    tier::HardwareTier
end

function classify_hardware(
    memory_bytes::Integer;
    cpu_threads::Integer=Sys.CPU_THREADS,
    gpu_backend::Symbol=:cpu,
)::RuntimeProfile
    memory_bytes > 0 || throw(ArgumentError("memory_bytes must be positive"))
    cpu_threads > 0 || throw(ArgumentError("cpu_threads must be positive"))
    gpu_backend in SUPPORTED_GPU_BACKENDS || throw(ArgumentError("unsupported gpu_backend: $gpu_backend"))

    gib = memory_bytes / 1024^3
    tier = if gib < 12
        HARDWARE_CONSTRAINED
    elseif gib < 24
        HARDWARE_MAINSTREAM
    elseif gib < 56
        HARDWARE_ENTHUSIAST
    else
        HARDWARE_WORKSTATION
    end

    RuntimeProfile(Int(memory_bytes), Int(cpu_threads), gpu_backend, tier)
end

detect_hardware(; gpu_backend::Symbol=:cpu) = classify_hardware(
    Sys.total_memory();
    cpu_threads=Sys.CPU_THREADS,
    gpu_backend=gpu_backend,
)

function model_policy(profile::RuntimeProfile)::Dict{String,Any}
    if profile.tier == HARDWARE_CONSTRAINED
        return Dict(
            "fast_model_resident" => true,
            "deliberate_model" => "disabled",
            "specialists" => "one_at_a_time",
        )
    elseif profile.tier == HARDWARE_MAINSTREAM
        return Dict(
            "fast_model_resident" => true,
            "deliberate_model" => "on_demand",
            "specialists" => "sequential",
        )
    elseif profile.tier == HARDWARE_ENTHUSIAST
        return Dict(
            "fast_model_resident" => true,
            "deliberate_model" => "resident_if_accelerated",
            "specialists" => "parallel_limited",
        )
    end

    Dict(
        "fast_model_resident" => true,
        "deliberate_model" => "resident",
        "specialists" => "parallel",
    )
end

end
```

Include the module in `src/IngExuity.jl` immediately after `Types.jl`:

```julia
include("foundation/RuntimeProfiles.jl")
```

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
julia --project=. test/runtests.jl
```

Expected: all previous tests plus `RuntimeProfiles` pass.

- [ ] **Step 5: Commit**

```powershell
git add src/foundation/RuntimeProfiles.jl test/foundation/runtime_profiles.jl test/runtests.jl src/IngExuity.jl
git commit -m "feat: add hardware-aware runtime profiles"
```

### Task 3: Add generic checkpoints and rollback

**Files:**
- Create: `src/foundation/VersionHistory.jl`
- Create: `test/foundation/version_history.jl`
- Modify: `src/IngExuity.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing rollback tests**

Create `test/foundation/version_history.jl`:

```julia
using .IngExuity.VersionHistory

@testset "VersionHistory" begin
    state = Dict("tone" => "warm")
    history = VersionedState(state)

    first_id = checkpoint!(history, "baseline")
    history.current["tone"] = "direct"
    second_id = checkpoint!(history, "direct")
    history.current["tone"] = "playful"

    restored = rollback!(history, first_id)
    @test restored["tone"] == "warm"
    @test history.current["tone"] == "warm"
    @test length(list_snapshots(history)) == 2
    @test second_id > first_id

    restored["tone"] = "mutated copy"
    @test history.current["tone"] == "warm"
    @test_throws KeyError rollback!(history, 999)
end
```

Add to `test/runtests.jl`:

```julia
include(joinpath(@__DIR__, "foundation", "version_history.jl"))
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run `julia --project=. test/runtests.jl`.

Expected: FAIL because `VersionHistory` does not exist.

- [ ] **Step 3: Implement defensive-copy version history**

Create `src/foundation/VersionHistory.jl`:

```julia
module VersionHistory

using Dates

export StateSnapshot, VersionedState, checkpoint!, rollback!, list_snapshots

struct StateSnapshot{T}
    id::Int
    label::String
    created_at::DateTime
    state::T
end

mutable struct VersionedState{T}
    current::T
    snapshots::Vector{StateSnapshot{T}}
    next_id::Int
end

VersionedState(initial::T) where {T} = VersionedState{T}(deepcopy(initial), StateSnapshot{T}[], 1)

function checkpoint!(history::VersionedState{T}, label::AbstractString)::Int where {T}
    isempty(strip(label)) && throw(ArgumentError("snapshot label cannot be empty"))
    id = history.next_id
    push!(history.snapshots, StateSnapshot{T}(id, String(label), now(), deepcopy(history.current)))
    history.next_id += 1
    id
end

function rollback!(history::VersionedState, id::Integer)
    snapshot = findfirst(s -> s.id == id, history.snapshots)
    snapshot === nothing && throw(KeyError(id))
    history.current = deepcopy(history.snapshots[snapshot].state)
    deepcopy(history.current)
end

list_snapshots(history::VersionedState) = [
    Dict("id" => s.id, "label" => s.label, "created_at" => string(s.created_at))
    for s in history.snapshots
]

end
```

Include after `RuntimeProfiles.jl`:

```julia
include("foundation/VersionHistory.jl")
```

- [ ] **Step 4: Run tests and commit**

Run `julia --project=. test/runtests.jl` and expect PASS.

```powershell
git add src/foundation/VersionHistory.jl test/foundation/version_history.jl test/runtests.jl src/IngExuity.jl
git commit -m "feat: add reversible state checkpoints"
```

### Task 4: Add relationship modes and reversible adaptive growth

**Files:**
- Create: `src/foundation/CompanionState.jl`
- Create: `test/foundation/companion_state.jl`
- Modify: `src/IngExuity.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing companion-state tests**

Create `test/foundation/companion_state.jl`:

```julia
using .IngExuity.CompanionState

@testset "CompanionState" begin
    state = create_companion_state()
    @test state.mode == MODE_EVERYDAY
    @test state.behavior.current.adaptive_growth == true
    @test state.behavior.current.strict_harm_prevention == true

    baseline = checkpoint_behavior!(state, "baseline")
    set_mode!(state, MODE_PRODUCTIVITY)
    set_growth_enabled!(state, false)
    record_adaptation!(state, "tone", "concise")

    @test state.mode == MODE_PRODUCTIVITY
    @test state.behavior.current.adaptive_growth == false
    @test !haskey(state.behavior.current.traits, "tone")

    set_growth_enabled!(state, true)
    record_adaptation!(state, "tone", "concise")
    @test state.behavior.current.traits["tone"] == "concise"

    rollback_behavior!(state, baseline)
    @test state.behavior.current.adaptive_growth == true
    @test isempty(state.behavior.current.traits)
    @test state.mode == MODE_PRODUCTIVITY
end
```

- [ ] **Step 2: Run and verify failure**

Expected: missing `CompanionState` module.

- [ ] **Step 3: Implement companion state**

Create `src/foundation/CompanionState.jl`:

```julia
module CompanionState

using Dates
using ..VersionHistory

export RelationshipMode, MODE_EVERYDAY, MODE_EMOTIONAL_SUPPORT, MODE_PRODUCTIVITY,
       BehaviorState, CompanionProfile, create_companion_state,
       set_mode!, set_growth_enabled!, record_adaptation!,
       checkpoint_behavior!, rollback_behavior!, companion_summary

@enum RelationshipMode begin
    MODE_EVERYDAY
    MODE_EMOTIONAL_SUPPORT
    MODE_PRODUCTIVITY
end

mutable struct BehaviorState
    adaptive_growth::Bool
    strict_harm_prevention::Bool
    traits::Dict{String,Any}
    revision::Int
end

mutable struct CompanionProfile
    mode::RelationshipMode
    behavior::VersionedState{BehaviorState}
    updated_at::DateTime
end

create_companion_state() = CompanionProfile(
    MODE_EVERYDAY,
    VersionedState(BehaviorState(true, true, Dict{String,Any}(), 0)),
    now(),
)

function set_mode!(profile::CompanionProfile, mode::RelationshipMode)
    profile.mode = mode
    profile.updated_at = now()
    profile
end

function set_growth_enabled!(profile::CompanionProfile, enabled::Bool)
    profile.behavior.current.adaptive_growth = enabled
    profile.behavior.current.revision += 1
    profile.updated_at = now()
    profile
end

function record_adaptation!(profile::CompanionProfile, key::AbstractString, value)
    behavior = profile.behavior.current
    behavior.adaptive_growth || return false
    behavior.traits[String(key)] = deepcopy(value)
    behavior.revision += 1
    profile.updated_at = now()
    true
end

checkpoint_behavior!(profile::CompanionProfile, label::AbstractString) = checkpoint!(profile.behavior, label)

function rollback_behavior!(profile::CompanionProfile, id::Integer)
    restored = rollback!(profile.behavior, id)
    profile.updated_at = now()
    restored
end

function companion_summary(profile::CompanionProfile)::Dict{String,Any}
    behavior = profile.behavior.current
    Dict(
        "mode" => string(profile.mode),
        "adaptive_growth" => behavior.adaptive_growth,
        "strict_harm_prevention" => behavior.strict_harm_prevention,
        "behavior_revision" => behavior.revision,
        "snapshot_count" => length(profile.behavior.snapshots),
    )
end

end
```

Include it after `VersionHistory.jl`, then add the test include to `test/runtests.jl`.

- [ ] **Step 4: Run tests and commit**

Run `julia --project=. test/runtests.jl` and expect PASS.

```powershell
git add src/foundation/CompanionState.jl test/foundation/companion_state.jl test/runtests.jl src/IngExuity.jl
git commit -m "feat: add reversible companion growth state"
```

### Task 5: Add permission tiers and deterministic authorization

**Files:**
- Create: `src/foundation/Permissions.jl`
- Create: `test/foundation/permissions.jl`
- Modify: `src/IngExuity.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the authorization matrix tests**

Create `test/foundation/permissions.jl`:

```julia
using .IngExuity.Permissions

@testset "Permissions" begin
    request = ActionRequest(:files, RISK_LOW, true, false)

    suggest = PermissionPolicy(tier=PERMISSION_SUGGEST_ONLY, grants=Set([:files]))
    confirm = PermissionPolicy(tier=PERMISSION_CONFIRM_EACH, grants=Set([:files]))
    tiered = PermissionPolicy(tier=PERMISSION_TIERED, grants=Set([:files]))
    broad = PermissionPolicy(tier=PERMISSION_BROAD_AUTONOMY, grants=Set([:files]))

    @test authorize(suggest, request).decision == ACTION_SUGGEST
    @test authorize(confirm, request).decision == ACTION_CONFIRM
    @test authorize(tiered, request).decision == ACTION_ALLOW
    @test authorize(broad, request).decision == ACTION_ALLOW

    @test authorize(tiered, ActionRequest(:messages, RISK_LOW, false, false)).decision == ACTION_DENY
    @test authorize(tiered, ActionRequest(:files, RISK_HIGH, false, false)).decision == ACTION_CONFIRM
    @test authorize(broad, ActionRequest(:files, RISK_LOW, false, true)).decision == ACTION_CONFIRM
end
```

- [ ] **Step 2: Run and verify failure**

Expected: missing `Permissions` module.

- [ ] **Step 3: Implement permission policy**

Create `src/foundation/Permissions.jl`:

```julia
module Permissions

export PermissionTier, RiskLevel, ActionDecision,
       PERMISSION_SUGGEST_ONLY, PERMISSION_CONFIRM_EACH,
       PERMISSION_TIERED, PERMISSION_BROAD_AUTONOMY,
       RISK_LOW, RISK_MEDIUM, RISK_HIGH,
       ACTION_DENY, ACTION_SUGGEST, ACTION_CONFIRM, ACTION_ALLOW,
       ActionRequest, PermissionPolicy, Authorization, authorize, permission_summary

@enum PermissionTier begin
    PERMISSION_SUGGEST_ONLY
    PERMISSION_CONFIRM_EACH
    PERMISSION_TIERED
    PERMISSION_BROAD_AUTONOMY
end

@enum RiskLevel begin
    RISK_LOW
    RISK_MEDIUM
    RISK_HIGH
end

@enum ActionDecision begin
    ACTION_DENY
    ACTION_SUGGEST
    ACTION_CONFIRM
    ACTION_ALLOW
end

struct ActionRequest
    category::Symbol
    risk::RiskLevel
    reversible::Bool
    protected_resource::Bool
end

Base.@kwdef mutable struct PermissionPolicy
    tier::PermissionTier = PERMISSION_TIERED
    grants::Set{Symbol} = Set{Symbol}()
end

struct Authorization
    decision::ActionDecision
    reason::String
end

function authorize(policy::PermissionPolicy, request::ActionRequest)::Authorization
    request.category in policy.grants || return Authorization(ACTION_DENY, "category_not_granted")
    policy.tier == PERMISSION_SUGGEST_ONLY && return Authorization(ACTION_SUGGEST, "suggest_only")
    policy.tier == PERMISSION_CONFIRM_EACH && return Authorization(ACTION_CONFIRM, "confirm_each")
    request.protected_resource && return Authorization(ACTION_CONFIRM, "protected_resource")
    request.risk == RISK_HIGH && return Authorization(ACTION_CONFIRM, "high_risk")
    policy.tier == PERMISSION_TIERED && request.risk == RISK_MEDIUM &&
        return Authorization(ACTION_CONFIRM, "tiered_medium_risk")
    Authorization(ACTION_ALLOW, "granted")
end

permission_summary(policy::PermissionPolicy) = Dict(
    "tier" => string(policy.tier),
    "grants" => sort!(string.(collect(policy.grants))),
)

end
```

Include it after `CompanionState.jl`, then include the test in `test/runtests.jl`.

- [ ] **Step 4: Run tests and commit**

Run `julia --project=. test/runtests.jl` and expect PASS.

```powershell
git add src/foundation/Permissions.jl test/foundation/permissions.jl test/runtests.jl src/IngExuity.jl
git commit -m "feat: add permission tiers and authorization"
```

### Task 6: Wire the foundation state and status API

**Files:**
- Modify: `src/IngExuity.jl`
- Create: `test/foundation/integration.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing integration tests**

Create `test/foundation/integration.jl`:

```julia
using HTTP
using JSON

@testset "Foundation integration" begin
    summary = IngExuity.get_foundation_status()
    @test summary["companion"]["mode"] == "MODE_EVERYDAY"
    @test summary["permissions"]["tier"] == "PERMISSION_TIERED"
    @test haskey(summary["runtime"], "tier")

    response = IngExuity.handle_request(HTTP.Request("GET", "/api/foundation"))
    @test response.status == 200
    payload = JSON.parse(String(response.body))
    @test payload["companion"]["adaptive_growth"] == true
    @test payload["runtime"]["policy"]["fast_model_resident"] == true
end
```

- [ ] **Step 2: Run and verify missing API failure**

Run `julia --project=. test/runtests.jl`.

Expected: FAIL because `get_foundation_status` is undefined.

- [ ] **Step 3: Add foundation globals and API**

In `src/IngExuity.jl`, export and initialize:

```julia
export get_foundation_status

const RUNTIME_PROFILE = RuntimeProfiles.detect_hardware()
const COMPANION_PROFILE = CompanionState.create_companion_state()
const PERMISSION_POLICY = Permissions.PermissionPolicy(
    tier=Permissions.PERMISSION_TIERED,
    grants=Set([:files, :browser, :reminders, :notifications, :apps]),
)

function get_foundation_status()::Dict{String,Any}
    runtime = Dict(
        "tier" => string(RUNTIME_PROFILE.tier),
        "memory_bytes" => RUNTIME_PROFILE.memory_bytes,
        "cpu_threads" => RUNTIME_PROFILE.cpu_threads,
        "gpu_backend" => string(RUNTIME_PROFILE.gpu_backend),
        "policy" => RuntimeProfiles.model_policy(RUNTIME_PROFILE),
    )
    Dict(
        "runtime" => runtime,
        "companion" => CompanionState.companion_summary(COMPANION_PROFILE),
        "permissions" => Permissions.permission_summary(PERMISSION_POLICY),
    )
end
```

Add to `handle_request` before the local-model endpoints:

```julia
elseif target == "/api/foundation" && HTTP.method(req) == "GET"
    return HTTP.Response(
        200,
        ["Content-Type" => "application/json"],
        body=to_json(get_foundation_status()),
    )
```

Include the integration test in `test/runtests.jl`.

- [ ] **Step 4: Run tests and commit**

Run both:

```powershell
julia --project=. test/runtests.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

Expected: all tests pass through both entry points.

```powershell
git add src/IngExuity.jl test/foundation/integration.jl test/runtests.jl
git commit -m "feat: expose production foundation status"
```

### Task 7: Reconcile public roadmap and verify the slice

**Files:**
- Modify: `README.md`
- Modify: `plans/INGEXUITY_PHASED_BUILD_PLAN.md`

- [ ] **Step 1: Update status wording**

Replace claims that Phase 1 is merely starting with evidence-bound wording:

```markdown
## Status

The existing conversation pipeline is covered by the Julia test suite. The production-foundation slice adds hardware profiles, relationship modes, reversible adaptive-growth state, permission tiers, and `/api/foundation`.

The shipping inference backend remains the validated GGUF path. Julia-native model work is an isolated research track and must pass quality, latency, memory, and reliability promotion gates before replacing production inference.
```

At the top of `plans/INGEXUITY_PHASED_BUILD_PLAN.md`, add:

```markdown
> Superseded for production sequencing by `docs/superpowers/specs/2026-07-13-edge-pal-product-design.md`. This file remains historical context. Experimental model work cannot block the desktop product and requires benchmark promotion gates.
```

- [ ] **Step 2: Run final verification**

Run:

```powershell
julia --project=. test/runtests.jl
julia --project=. -e "using Pkg; Pkg.test()"
git diff --check
git status --short
```

Expected:

- Direct and package test paths both pass.
- No whitespace errors.
- Only intended documentation changes remain unstaged.

- [ ] **Step 3: Commit documentation**

```powershell
git add README.md plans/INGEXUITY_PHASED_BUILD_PLAN.md
git commit -m "docs: align roadmap with production foundation"
```

- [ ] **Step 4: Record the next implementation plans**

Create follow-up plans, in order, for:

1. durable encrypted vault and independent memory/behavior restore;
2. local model ladder and compound orchestrator;
3. desktop shell and mode/growth controls;
4. permissioned tools, receipts, and deterministic verification;
5. Broad Autonomy hardening;
6. Android identity-bundle pairing;
7. Julia research benchmarks and production promotion.

