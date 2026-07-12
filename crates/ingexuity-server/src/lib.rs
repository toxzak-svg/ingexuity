#![forbid(unsafe_code)]

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Path, State},
    http::{HeaderName, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ingexuity_core::{
    process_turn, BackendHealth, ChatOutcome, ConversationState, CoreError, HeuristicBackend,
    InferenceBackend, SessionId, MAX_MESSAGE_BYTES,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock, Semaphore};
use tower_http::{
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::TraceLayer,
};
use uuid::Uuid;

pub const MAX_REQUEST_BODY_BYTES: usize = MAX_MESSAGE_BYTES + 1024;
pub const DEFAULT_SESSION_TTL: Duration = Duration::from_secs(30 * 60);
pub const DEFAULT_MAX_INFERENCE_CONCURRENCY: usize = 4;

pub trait Clock: Send + Sync {
    fn now_millis(&self) -> u64;
}

#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_millis(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

struct SessionEntry {
    state: Mutex<ConversationState>,
    created_at_ms: u64,
    last_access_ms: AtomicU64,
}

impl SessionEntry {
    fn new(state: ConversationState, now_ms: u64) -> Self {
        Self {
            state: Mutex::new(state),
            created_at_ms: now_ms,
            last_access_ms: AtomicU64::new(now_ms),
        }
    }
}

#[derive(Clone)]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, Arc<SessionEntry>>>>,
    clock: Arc<dyn Clock>,
    ttl_ms: u64,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(DEFAULT_SESSION_TTL)
    }
}

impl SessionManager {
    #[must_use]
    pub fn new(ttl: Duration) -> Self {
        Self::with_clock(ttl, Arc::new(SystemClock))
    }

    #[must_use]
    pub fn with_clock(ttl: Duration, clock: Arc<dyn Clock>) -> Self {
        let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX).max(1);
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            clock,
            ttl_ms,
        }
    }

    pub async fn create(&self) -> SessionId {
        let id = Uuid::new_v4();
        let now_ms = self.clock.now_millis();
        let entry = Arc::new(SessionEntry::new(ConversationState::new(id), now_ms));
        self.sessions.write().await.insert(id, entry);
        id
    }

    async fn get(&self, id: SessionId) -> Option<Arc<SessionEntry>> {
        let now_ms = self.clock.now_millis();
        let entry = self.sessions.read().await.get(&id).cloned()?;

        if self.is_expired(&entry, now_ms) {
            let mut sessions = self.sessions.write().await;
            let should_remove = sessions
                .get(&id)
                .map(|current| Arc::ptr_eq(current, &entry) && self.is_expired(current, now_ms))
                .unwrap_or(false);
            if should_remove {
                sessions.remove(&id);
            }
            return None;
        }

        entry.last_access_ms.store(now_ms, Ordering::Release);
        Some(entry)
    }

    #[must_use]
    pub fn ttl_seconds(&self) -> u64 {
        self.ttl_ms.div_ceil(1000)
    }

    pub async fn count(&self) -> usize {
        self.sweep_expired().await;
        self.sessions.read().await.len()
    }

    pub async fn snapshot(&self, id: SessionId) -> Option<ConversationState> {
        let entry = self.get(id).await?;
        let snapshot = entry.state.lock().await.clone();
        Some(snapshot)
    }

    pub async fn metadata(&self, id: SessionId) -> Option<SessionMetadataResponse> {
        let entry = self.get(id).await?;
        let state = entry.state.lock().await;
        Some(self.metadata_from(&entry, &state))
    }

    pub async fn reset(&self, id: SessionId) -> Option<SessionMetadataResponse> {
        let entry = self.get(id).await?;
        let mut state = entry.state.lock().await;
        *state = ConversationState::new(id);
        Some(self.metadata_from(&entry, &state))
    }

    pub async fn delete(&self, id: SessionId) -> bool {
        self.sessions.write().await.remove(&id).is_some()
    }

    pub async fn sweep_expired(&self) -> usize {
        let now_ms = self.clock.now_millis();
        let mut sessions = self.sessions.write().await;
        let before = sessions.len();
        sessions.retain(|_, entry| !self.is_expired(entry, now_ms));
        before - sessions.len()
    }

    fn is_expired(&self, entry: &SessionEntry, now_ms: u64) -> bool {
        let last_access_ms = entry.last_access_ms.load(Ordering::Acquire);
        now_ms.saturating_sub(last_access_ms) > self.ttl_ms
    }

    fn metadata_from(
        &self,
        entry: &SessionEntry,
        state: &ConversationState,
    ) -> SessionMetadataResponse {
        let last_access_ms = entry.last_access_ms.load(Ordering::Acquire);
        SessionMetadataResponse {
            session_id: state.session_id,
            turn_count: state.turn_count,
            state_version: state.version,
            created_at_ms: entry.created_at_ms,
            last_access_ms,
            expires_at_ms: last_access_ms.saturating_add(self.ttl_ms),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub session_ttl: Duration,
    pub max_inference_concurrency: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            session_ttl: DEFAULT_SESSION_TTL,
            max_inference_concurrency: DEFAULT_MAX_INFERENCE_CONCURRENCY,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    sessions: SessionManager,
    backend: Arc<dyn InferenceBackend>,
    inference_slots: Arc<Semaphore>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(Arc::new(HeuristicBackend), AppConfig::default())
    }
}

impl AppState {
    #[must_use]
    pub fn new(backend: Arc<dyn InferenceBackend>, config: AppConfig) -> Self {
        Self::with_components(
            backend,
            SessionManager::new(config.session_ttl),
            config.max_inference_concurrency,
        )
    }

    #[must_use]
    pub fn with_components(
        backend: Arc<dyn InferenceBackend>,
        sessions: SessionManager,
        max_inference_concurrency: usize,
    ) -> Self {
        Self {
            sessions,
            backend,
            inference_slots: Arc::new(Semaphore::new(max_inference_concurrency.max(1))),
        }
    }

    #[must_use]
    pub fn sessions(&self) -> &SessionManager {
        &self.sessions
    }
}

pub fn app(state: AppState) -> Router {
    let request_id_header = HeaderName::from_static("x-request-id");

    Router::new()
        .route("/health", get(liveness))
        .route("/health/live", get(liveness))
        .route("/health/ready", get(readiness))
        .route("/api/v1/sessions", post(create_session))
        .route(
            "/api/v1/sessions/{session_id}",
            get(get_session).delete(delete_session),
        )
        .route("/api/v1/sessions/{session_id}/reset", post(reset_session))
        .route("/api/v1/chat", post(chat))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(PropagateRequestIdLayer::new(request_id_header.clone()))
        .layer(TraceLayer::new_for_http())
        .layer(SetRequestIdLayer::new(request_id_header, MakeRequestUuid))
        .with_state(state)
}

#[derive(Debug, Serialize)]
struct LivenessResponse {
    status: &'static str,
    runtime: &'static str,
}

async fn liveness() -> Json<LivenessResponse> {
    Json(LivenessResponse {
        status: "ok",
        runtime: "rust",
    })
}

#[derive(Debug, Serialize)]
struct ReadinessResponse {
    status: &'static str,
    runtime: &'static str,
    backend: BackendHealth,
    active_sessions: usize,
    available_inference_slots: usize,
}

async fn readiness(State(state): State<AppState>) -> (StatusCode, Json<ReadinessResponse>) {
    let backend = state.backend.health();
    let ready = !matches!(backend, BackendHealth::Unavailable);
    let status = if ready { "ready" } else { "not_ready" };
    let status_code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status_code,
        Json(ReadinessResponse {
            status,
            runtime: "rust",
            backend,
            active_sessions: state.sessions.count().await,
            available_inference_slots: state.inference_slots.available_permits(),
        }),
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session_id: SessionId,
    pub expires_in_seconds: u64,
}

async fn create_session(
    State(state): State<AppState>,
) -> (StatusCode, Json<CreateSessionResponse>) {
    let session_id = state.sessions.create().await;
    (
        StatusCode::CREATED,
        Json(CreateSessionResponse {
            session_id,
            expires_in_seconds: state.sessions.ttl_seconds(),
        }),
    )
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionMetadataResponse {
    pub session_id: SessionId,
    pub turn_count: u64,
    pub state_version: u64,
    pub created_at_ms: u64,
    pub last_access_ms: u64,
    pub expires_at_ms: u64,
}

async fn get_session(
    Path(session_id): Path<SessionId>,
    State(state): State<AppState>,
) -> Result<Json<SessionMetadataResponse>, AppError> {
    let metadata = state
        .sessions
        .metadata(session_id)
        .await
        .ok_or(AppError::UnknownSession)?;
    Ok(Json(metadata))
}

async fn reset_session(
    Path(session_id): Path<SessionId>,
    State(state): State<AppState>,
) -> Result<Json<SessionMetadataResponse>, AppError> {
    let metadata = state
        .sessions
        .reset(session_id)
        .await
        .ok_or(AppError::UnknownSession)?;
    Ok(Json(metadata))
}

async fn delete_session(
    Path(session_id): Path<SessionId>,
    State(state): State<AppState>,
) -> Result<StatusCode, AppError> {
    if state.sessions.delete(session_id).await {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::UnknownSession)
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub session_id: SessionId,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub session_id: SessionId,
    pub text: String,
    pub backend_id: String,
    pub turn_count: u64,
    pub state_version: u64,
}

async fn chat(
    State(state): State<AppState>,
    payload: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Json<ChatResponse>, AppError> {
    let Json(request) = payload.map_err(AppError::from_json_rejection)?;
    let session = state
        .sessions
        .get(request.session_id)
        .await
        .ok_or(AppError::UnknownSession)?;
    let _permit = state
        .inference_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| AppError::ServerBusy)?;

    let mut session = session.state.lock().await;
    let ChatOutcome {
        text,
        backend,
        turn_count,
        state_version,
    } = process_turn(&mut session, &request.message, state.backend.as_ref())?;

    Ok(Json(ChatResponse {
        session_id: request.session_id,
        text,
        backend_id: backend.id,
        turn_count,
        state_version,
    }))
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error("request body must be valid JSON")]
    InvalidJson,
    #[error("content-type must be application/json")]
    UnsupportedMediaType,
    #[error("request body is too large")]
    RequestTooLarge,
    #[error("session does not exist or has expired")]
    UnknownSession,
    #[error("inference capacity is currently full")]
    ServerBusy,
    #[error(transparent)]
    Core(#[from] CoreError),
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

impl AppError {
    fn from_json_rejection(rejection: JsonRejection) -> Self {
        match rejection.status() {
            StatusCode::PAYLOAD_TOO_LARGE => Self::RequestTooLarge,
            StatusCode::UNSUPPORTED_MEDIA_TYPE => Self::UnsupportedMediaType,
            _ => Self::InvalidJson,
        }
    }

    fn status_and_code(&self) -> (StatusCode, &'static str) {
        match self {
            Self::InvalidJson => (StatusCode::BAD_REQUEST, "invalid_json"),
            Self::UnsupportedMediaType => {
                (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type")
            }
            Self::RequestTooLarge => (StatusCode::PAYLOAD_TOO_LARGE, "request_too_large"),
            Self::UnknownSession => (StatusCode::NOT_FOUND, "unknown_session"),
            Self::ServerBusy => (StatusCode::TOO_MANY_REQUESTS, "server_busy"),
            Self::Core(CoreError::EmptyMessage) => (StatusCode::BAD_REQUEST, "empty_message"),
            Self::Core(CoreError::MessageTooLarge) => {
                (StatusCode::PAYLOAD_TOO_LARGE, "message_too_large")
            }
            Self::Core(CoreError::Inference(_)) => {
                (StatusCode::SERVICE_UNAVAILABLE, "inference_unavailable")
            }
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code) = self.status_and_code();
        let body = ErrorEnvelope {
            error: ErrorBody {
                code,
                message: self.to_string(),
            },
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{header::CONTENT_TYPE, Request},
    };
    use http_body_util::BodyExt;
    use ingexuity_core::{BackendMetadata, Generation, GenerationRequest, InferenceError};
    use serde_json::{json, Value};
    use std::sync::atomic::AtomicBool;
    use tower::ServiceExt;

    async fn json_body(response: axum::response::Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn chat_request(session_id: SessionId, message: &str) -> Request<Body> {
        Request::post("/api/v1/chat")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({"session_id": session_id, "message": message}).to_string(),
            ))
            .unwrap()
    }

    #[tokio::test]
    async fn liveness_and_readiness_report_real_state() {
        let router = app(AppState::default());

        let live = router
            .clone()
            .oneshot(Request::get("/health/live").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(live.status(), StatusCode::OK);
        let live_body = json_body(live).await;
        assert_eq!(live_body["runtime"], "rust");
        assert_eq!(live_body["status"], "ok");

        let ready = router
            .oneshot(Request::get("/health/ready").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::OK);
        let ready_body = json_body(ready).await;
        assert_eq!(ready_body["status"], "ready");
        assert_eq!(ready_body["backend"], "ready");
    }

    #[tokio::test]
    async fn every_response_has_a_request_id() {
        let response = app(AppState::default())
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert!(response.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn malformed_json_is_machine_readable() {
        let response = app(AppState::default())
            .oneshot(
                Request::post("/api/v1/chat")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_json");
    }

    #[tokio::test]
    async fn missing_content_type_is_rejected() {
        let response = app(AppState::default())
            .oneshot(
                Request::post("/api/v1/chat")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "unsupported_media_type");
    }

    #[tokio::test]
    async fn oversized_request_is_rejected() {
        let message = "x".repeat(MAX_REQUEST_BODY_BYTES + 1);
        let response = app(AppState::default())
            .oneshot(
                Request::post("/api/v1/chat")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"session_id": Uuid::new_v4(), "message": message}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "request_too_large");
    }

    #[tokio::test]
    async fn sessions_are_isolated() {
        let state = AppState::default();
        let session_a = state.sessions.create().await;
        let session_b = state.sessions.create().await;
        let router = app(state.clone());

        for (session_id, message) in [
            (session_a, "My work and Rust code are the priority"),
            (session_b, "My family is the priority"),
        ] {
            let response = router
                .clone()
                .oneshot(chat_request(session_id, message))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let a = state.sessions.snapshot(session_a).await.unwrap();
        let b = state.sessions.snapshot(session_b).await.unwrap();

        assert!(a.user_model.topics.contains("work"));
        assert!(a.user_model.topics.contains("software"));
        assert!(!a.user_model.topics.contains("family"));
        assert!(b.user_model.topics.contains("family"));
        assert!(!b.user_model.topics.contains("work"));
    }

    #[tokio::test]
    async fn reset_and_delete_control_session_lifecycle() {
        let state = AppState::default();
        let session_id = state.sessions.create().await;
        let router = app(state.clone());

        let chat = router
            .clone()
            .oneshot(chat_request(session_id, "Rust work"))
            .await
            .unwrap();
        assert_eq!(chat.status(), StatusCode::OK);

        let reset = router
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/sessions/{session_id}/reset"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reset.status(), StatusCode::OK);
        let reset_body = json_body(reset).await;
        assert_eq!(reset_body["turn_count"], 0);
        assert_eq!(reset_body["state_version"], 0);

        let delete = router
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/sessions/{session_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delete.status(), StatusCode::NO_CONTENT);

        let missing = router
            .oneshot(chat_request(session_id, "hello"))
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[derive(Default)]
    struct ManualClock {
        now_ms: AtomicU64,
    }

    impl ManualClock {
        fn advance(&self, millis: u64) {
            self.now_ms.fetch_add(millis, Ordering::AcqRel);
        }
    }

    impl Clock for ManualClock {
        fn now_millis(&self) -> u64 {
            self.now_ms.load(Ordering::Acquire)
        }
    }

    #[tokio::test]
    async fn expired_sessions_are_removed_deterministically() {
        let clock = Arc::new(ManualClock::default());
        let sessions = SessionManager::with_clock(Duration::from_millis(100), clock.clone());
        let session_id = sessions.create().await;

        clock.advance(101);

        assert!(sessions.snapshot(session_id).await.is_none());
        assert_eq!(sessions.count().await, 0);
    }

    struct SlowBackend {
        started: Arc<AtomicBool>,
    }

    impl InferenceBackend for SlowBackend {
        fn metadata(&self) -> BackendMetadata {
            BackendMetadata {
                id: "slow-test".to_owned(),
                kind: "test".to_owned(),
                model: None,
                local: true,
            }
        }

        fn health(&self) -> BackendHealth {
            BackendHealth::Ready
        }

        fn generate(&self, _: &GenerationRequest) -> Result<Generation, InferenceError> {
            self.started.store(true, Ordering::Release);
            std::thread::sleep(Duration::from_millis(150));
            Ok(Generation {
                text: "done".to_owned(),
            })
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inference_concurrency_is_bounded() {
        let started = Arc::new(AtomicBool::new(false));
        let state = AppState::with_components(
            Arc::new(SlowBackend {
                started: started.clone(),
            }),
            SessionManager::default(),
            1,
        );
        let session_a = state.sessions.create().await;
        let session_b = state.sessions.create().await;
        let router = app(state);

        let first_router = router.clone();
        let first = tokio::spawn(async move {
            first_router
                .oneshot(chat_request(session_a, "first"))
                .await
                .unwrap()
        });

        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }

        let second = router
            .oneshot(chat_request(session_b, "second"))
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = json_body(second).await;
        assert_eq!(body["error"]["code"], "server_busy");

        assert_eq!(first.await.unwrap().status(), StatusCode::OK);
    }
}
