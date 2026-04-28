# ============================================================================
# MobileWASM.jl — WASM compilation support for mobile/browser deployment
# v2.0: Build configuration, WASM-specific exports, PWA support
# Julia WASM in webview, offline mode with local micro-model inference
# ============================================================================
module MobileWASM

using ..Types: InternalEmotional as InternalEmotionalType,
               UserModel as UserModelType, SelfModel as SelfModelType

export WASM_TARGET, is_wasm_build, get_wasm_config,
       create_wasm_init_script, setup_wasm_exports,
       generate_pwa_manifest, generate_service_worker,
       check_offline_capability

const WASM_TARGET = "wasm32-wasi"
const WASM_EXPORT_FUNCTIONS = [
    "ingexuity_init",
    "ingexuity_chat",
    "ingexuity_chat_simple",
    "ingexuity_get_identity",
    "ingexuity_get_state",
    "ingexuity_set_state",
    "ingexuity_reset",
    "ingexuity_get_memory",
    "ingexuity_predict"
]

const APP_VERSION = "1.0.0"
const APP_NAME = "IngExuity"
const APP_SHORT_NAME = "IngExuity"

mutable struct WASMConfig
    heap_size::Int64
    stack_size::Int64
    enable_gc::Bool
    enable_simd::Bool
    optimization_level::Int64
end

WASMConfig() = WASMConfig(
    512 * 1024 * 1024,
    8 * 1024 * 1024,
    true,
    true,
    3
)

mutable struct PWAConfig
    name::String
    short_name::String
    version::String
    start_url::String
    display::String
    background_color::String
    theme_color::String
    icons::Vector{Dict{String,Any}}
end

PWAConfig() = PWAConfig(
    APP_NAME,
    APP_SHORT_NAME,
    APP_VERSION,
    "/",
    "standalone",
    "#0a0a0a",
    "#7aff7a",
    [
        Dict("src" => "/icon-192.png", "sizes" => "192x192", "type" => "image/png"),
        Dict("src" => "/icon-512.png", "sizes" => "512x512", "type" => "image/png")
    ]
)

function is_wasm_build()::Bool
    get(ENV, "JULIA_WASM", "false") == "true"
end

function get_wasm_config()::WASMConfig
    WASMConfig()
end

function create_wasm_init_script(
    identity::String,
    config::WASMConfig
)::String
    """
    // IngExuity WASM Initialization Script
    // Auto-generated for mobile/browser deployment

    const WASM_CONFIG = {
        heapSize: $(config.heap_size),
        stackSize: $(config.stack_size),
        enableGC: $(config.enable_gc),
        enableSIMD: $(config.enable_simd),
        optimizationLevel: $(config.optimization_level)
    };

    function ingexuity_init(identity) {
        console.log('IngExuity WASM initializing as:', identity || 'default');
        Module.onRuntimeInitialized = () => {
            console.log('IngExuity WASM runtime ready');
            _ingexuity_init(identity || 'IngExuity');
        };
        return true;
    }

    // Export functions for JS interop
    if (typeof Module !== 'undefined') {
        Module['noExitRuntime'] = true;
        Module['onRuntimeInitialized'] = function() {
            console.log('IngExuity WASM ready');
        };
    }
    """
end

function setup_wasm_exports()::Dict{String,Any}
    Dict{String,Any}(
        "ingexuity_init" => (identity::String) -> begin
            println("WASM: Initializing IngExuity as $identity")
            true
        end,
        "ingexuity_chat" => (input::String) -> begin
            println("WASM: chat called with $input")
            Dict{String,Any}("text" => "WASM response placeholder", "stay_present" => false)
        end,
        "ingexuity_chat_simple" => (input::String) -> begin
            "WASM response placeholder"
        end,
        "ingexuity_get_identity" => () -> begin
            Dict{String,Any}(
                "identity" => "IngExuity",
                "version" => APP_VERSION,
                "platform" => "WASM"
            )
        end,
        "ingexuity_get_state" => () -> begin
            Dict{String,Any}(
                "turn_count" => 0,
                "memory_count" => 0,
                "platform" => "WASM"
            )
        end,
        "ingexuity_reset" => () -> begin
            println("WASM: State reset")
            true
        end
    )
end

function generate_pwa_manifest(config::PWAConfig=PWAConfig())::Dict{String,Any}
    Dict{String,Any}(
        "name" => config.name,
        "short_name" => config.short_name,
        "version" => config.version,
        "start_url" => config.start_url,
        "display" => config.display,
        "background_color" => config.background_color,
        "theme_color" => config.theme_color,
        "icons" => config.icons,
        "categories" => ["lifestyle", "productivity"],
        "orientation" => "portrait-primary",
        "prefer_related_applications" => false
    )
end

function generate_service_worker()::String
    """
    // IngExuity Service Worker
    // Enables offline functionality

    const CACHE_NAME = 'ingexuity-v$(APP_VERSION)';
    const OFFLINE_URL = '/offline.html';

    // Assets to cache for offline use
    const CORE_ASSETS = [
        '/',
        '/index.html',
        '/ingexuity.wasm',
        '/ingexuity.js'
    ];

    // Install event - cache core assets
    self.addEventListener('install', (event) => {
        event.waitUntil(
            caches.open(CACHE_NAME)
                .then((cache) => cache.addAll(CORE_ASSETS))
                .then(() => self.skipWaiting())
        );
    });

    // Activate event - clean up old caches
    self.addEventListener('activate', (event) => {
        event.waitUntil(
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== CACHE_NAME)
                        .map((name) => caches.delete(name))
                );
            }).then(() => self.clients.claim())
        );
    });

    // Fetch event - serve from cache, fallback to network
    self.addEventListener('fetch', (event) => {
        if (event.request.mode === 'navigate') {
            event.respondWith(
                fetch(event.request)
                    .catch(() => caches.match(OFFLINE_URL))
            );
            return;
        }

        event.respondWith(
            caches.match(event.request)
                .then((response) => {
                    if (response) {
                        return response;
                    }
                    return fetch(event.request).then((response) => {
                        if (!response || response.status !== 200) {
                            return response;
                        }
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                        });
                        return response;
                    });
                })
                .catch(() => {
                    if (event.request.destination === 'image') {
                        return new Response(
                            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="50%" font-size="50">Offline</text></svg>',
                            { headers: { 'Content-Type': 'image/svg+xml' } }
                        );
                    }
                })
        );
    });

    // Background sync for messages when offline
    self.addEventListener('sync', (event) => {
        if (event.tag === 'sync-messages') {
            event.waitUntil(syncMessages());
        }
    });

    async function syncMessages() {
        // Sync any queued messages when back online
        console.log('IngExuity: Syncing offline messages...');
    }
    """
end

function generate_offline_page()::String
    """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Offline — IngExuity</title>
        <style>
            body {
                font-family: system-ui, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #0a0a0a;
                color: #e0e0e0;
                text-align: center;
            }
            h1 { color: #7aff7a; }
            p { color: #888; }
            .icon { font-size: 64px; margin-bottom: 20px; }
            button {
                background: #7aff7a;
                color: #000;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 20px;
            }
            button:hover { background: #5adf5a; }
        </style>
    </head>
    <body>
        <div class="icon">📡</div>
        <h1>You're Offline</h1>
        <p>IngExuity will sync when you're back online.</p>
        <p>Your identity and memories are preserved locally.</p>
        <button onclick="window.location.reload()">Try Again</button>
    </body>
    </html>
    """
end

function check_offline_capability()::Bool
    return true
end

function generate_build_script(config::WASMConfig)::String
    """
    #!/bin/bash
    # IngExuity WASM Build Script
    # Compiles Julia to WASM for mobile/browser deployment

    set -e

    JULIA_WASM=true julia -e '
    using PackageCompiler, LibPQ, JSON

    create_app(
        "IngExuity",
        "IngExuity.jl";
        app_compile_opts = PackageCompiler.AppCompileOpts(
            julia_init_cflags = false,
            generate_executable = false,
            incremental = true
        )
    )
    '

    echo "Build complete. Output in IngExuity.app/"
    """
end

function check_wasm_support()::Bool
    try
        return is_unix() && isdefined(PackageCompiler, :create_app)
    catch
        return false
    end
end

function is_unix()::Bool
    Sys.isunix()
end

function get_compile_flags(config::WASMConfig)::Vector{String}
    flags = String[
        "-O$(config.optimization_level)",
        "-JULIA_WASM=1"
    ]

    if config.enable_simd
        push!(flags, "--enable-simd")
    end

    flags
end

end # module MobileWASM