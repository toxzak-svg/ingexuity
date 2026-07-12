#![forbid(unsafe_code)]

mod persistence;

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
use ingexuity_store::{NewEvent, SqliteStore, StoreError, StoredSession};
use persistence::{Persistence, PersistenceError};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
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
use tracing::error;
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
    deleted: AtomicBool,
}

impl SessionEntry {
    fn new(state: ConversationState, created_at_ms: u64, last_access_ms: u64) -> Self {
        Self {
            state: Mutex::new(state),
            created_at_ms,
            last_access_ms: AtomicU64::new(last_access_ms),
            deleted: AtomicBool::new(false),
        }
    }
}

#[derive(Clone)]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, Arc<SessionEntry>>>>,
    expired_ids: Arc<Mutex<Vec<SessionId>>>,
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
            expired_ids: Arc::new(Mutex::new(Vec::new())),
            clock,
            ttl_ms,
        }
    }

    pub async fn create(&self) -> SessionId {
        let id = Uuid::new_v4();
        let now_ms = self.clock.now_millis();
        let entry = Arc::new(SessionEntry::new(
            ConversationState::new(id),
            now_ms,
            now_ms,
        ));
        self.sessions.write().await.insert(id, entry);
        id
    }

    pub async fn restore(&self, stored: StoredSession) -> bool {
        let now_ms = self.clock.now_millis();
        if now_ms.saturating_sub(stored.updated_at_ms) > self.ttl_ms {
            self.expired_ids.lock().await.push(stored.session_id);
            return false;
        }

        let entry = Arc::new(SessionEntry::new(
            stored.snapshot,
            stored.created_at_ms,
            stored.updated_at_ms,
        ));
        self.sessions.write().await.insert(stored.session_id, entry);
        true
    }

    async fn get(&self, id: SessionId) -> Option<Arc<SessionEntry>> {
        let now_ms = self.clock.now_millis();
        let entry = self.sessions.read().await.get(&id).cloned()?;

        if entry.deleted.load(Ordering::Acquire) || self.is_expired(&entry, now_ms) {
            let mut sessions = self.sessions.write().await;
            let should_remove = sessions
                .get(&id)
                .map(|current| Arc::ptr_eq(current, &entry))
                .unwrap_or(false);
            if should_remove {
                entry.deleted.store(true, Ordering::Release);
                sessions.remove(&id);
                self.expired_ids.lock().await.push(id);
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

    pub fn now_millis(&self) -> u64 {
        self.clock.now_millis()
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

    async fn remove(&self, id: SessionId) -> bool {
        let entry = self.sessions.write().await.remove(&id);
        if let Some(entry) = entry {
            entry.deleted.store(true, Ordering::Release);
            true
        } else {
            false
        }
    }

    async fn sweep_expired(&self) -> usize {
        let now_ms = self.clock.now_millis();
        let mut expired = Vec::new();
        let mut sessions = self.sessions.write().await;
        sessions.retain(|id, entry| {
            let keep = !entry.deleted.load(Ordering::Acquire) && !self.is_expired(entry, now_ms);
            if !keep {
                entry.deleted.store(true, Ordering::Release);
                expired.push(*id);
            }
            keep
        });
        let count = expired.len();
        drop(sessions);
        self.expired_ids.lock().await.extend(expired);
        count
    }

    async fn drain_expired(&self) -> Vec<SessionId> {
        std::mem::take(&mut *self.expired_ids.lock().await)
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

#[derive(Debug, Error)]
pub enum AppInitError {
    #[error(transparent)]
    Persistence(#[from] PersistenceError),
}

#[derive(Clone)]
pub struct AppState {
    sessions: SessionManager,
    backend: Arc<dyn InferenceBackend>,
    inference_slots: Arc<Semaphore>,
    persistence: Persistence,
}

impl Default for AppState {
    fn default() -> Self {
        let store = Arc::new(
            SqliteStore::open_in_memory().expect("in-memory SQLite store must initialize"),
        );
        Self::with_store(
            Arc::new(HeuristicBackend),
            SessionManager::default(),
            DEFAULT_MAX_INFERENCE_CONCURRENCY,
            store,
        )
    }
}

impl AppState {
    #[must_use]
    pub fn new(backend: Arc<dyn InferenceBackend>, config: AppConfig) -> Self {
        let store = Arc::new(
            SqliteStore::open_in_memory().expect("in-memory SQLite store must initialize"),
        );
        Self::with_store(
            backend,
            SessionManager::new(config.session_ttl),
            config.max_inference_concurrency,
            store,
        )
    }

    #[must_use]
    pub fn with_components(
        backend: Arc<dyn InferenceBackend>,
        sessions: SessionManager,
        max_inference_concurrency: usize,
    ) -> Self {
        let store = Arc::new(
            SqliteStore::open_in_memory().expect("in-memory SQLite store must initialize"),
        );
        Self::with_store(backend, sessions, max_inference_concurrency, store)
    }

    #[must_use]
    pub fn with_store(
        backend: Arc<dyn InferenceBackend>,
        sessions: SessionManager,
        max_inference_concurrency: usize,
        store: Arc<SqliteStore>,
    ) -> Self {
        Self {
            sessions,
            backend,
            inference_slots: Arc::new(Semaphore::new(max_inference_concurrency.max(1))),
            persistence: Persistence::new(store),
        }
    }

    pub async fn restore(
        backend: Arc<dyn InferenceBackend>,
        config: AppConfig,
        store: Arc<SqliteStore>,
    ) -> Result<Self, AppInitError> {
        let state = Self::with_store(
            backend,
            SessionManager::new(config.session_ttl),
            config.max_inference_concurrency,
            store,
        );
        state.persistence.quick_check().await?;
        for stored in state.persistence.list_sessions().await? {
            state.sessions.restore(stored).await;
        }
        state.purge_expired().await.map_err(|error| match error {
            AppError::Persistence(source) => AppInitError::Persistence(source),
            _ => unreachable!("startup expiration purge only returns persistence errors"),
        })?;
        Ok(state)
    }

    #[must_use]
    pub fn sessions(&self) -> &SessionManager {
        &self.sessions
    }

    async fn lookup_session(&self, id: SessionId) -> Result<Arc<SessionEntry>, AppError> {
        let entry = self.sessions.get(id).await;
        self.purge_expired().await?;
        entry.ok_or(AppError::UnknownSession)
    }

    async fn purge_expired(&self) -> Result<(), AppError> {
        for id in self.sessions.drain_expired().await {
            self.persistence
                .delete_session(id)
                .await
                .map_err(AppError::from_persistence)?;
        }
        Ok(())
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
    storage: &'static str,
    active_sessions: usize,
    available_inference_slots: usize,
}

async fn readiness(State(state): State<AppState>) -> (StatusCode, Json<ReadinessResponse>) {
    let backend = state.backend.health();
    let storage_ready = state.persistence.quick_check().await.is_ok();
    let active_sessions = state.sessions.count().await;
    let purge_ready = state.purge_expired().await.is_ok();
    let ready = !matches!(backend, BackendHealth::Unavailable) && storage_ready && purge_ready;

    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(ReadinessResponse {
            status: if ready { "ready" } else { "not_ready" },
            runtime: "rust",
            backend,
            storage: if storage_ready && purge_ready {
                "ready"
            } else {
                "unavailable"
            },
            active_sessions,
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
) -> Result<(StatusCode, Json<CreateSessionResponse>), AppError> {
    let session_id = state.sessions.create().await;
    let snapshot = state
        .sessions
        .snapshot(session_id)
        .await
        .ok_or(AppError::UnknownSession)?;
    let metadata = state
        .sessions
        .metadata(session_id)
        .await
        .ok_or(AppError::UnknownSession)?;

    if let Err(error) = state
        .persistence
        .create_session(snapshot, metadata.created_at_ms)
        .await
    {
        state.sessions.remove(session_id).await;
        return Err(AppError::from_persistence(error));
    }

    Ok((
        StatusCode::CREATED,
        Json(CreateSessionResponse {
            session_id,
            expires_in_seconds: state.sessions.ttl_seconds(),
        }),
    ))
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
    state.lookup_session(session_id).await?;
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
    let entry = state.lookup_session(session_id).await?;
    let mut guard = entry.state.lock().await;
    if entry.deleted.load(Ordering::Acquire) {
        return Err(AppError::UnknownSession);
    }

    let previous = guard.clone();
    guard.reset();
    let next = guard.clone();
    let now_ms = state.sessions.now_millis();
    let event = NewEvent::new(
        "session.reset",
        json!({"previous_version": previous.version}),
        now_ms,
    );

    if let Err(error) = state
        .persistence
        .append_events(session_id, previous.version, vec![event], next, now_ms)
        .await
    {
        *guard = previous;
        return Err(AppError::from_persistence(error));
    }

    Ok(Json(state.sessions.metadata_from(&entry, &guard)))
}

async fn delete_session(
    Path(session_id): Path<SessionId>,
    State(state): State<AppState>,
) -> Result<StatusCode, AppError> {
    let entry = state.lookup_session(session_id).await?;
    let _guard = entry.state.lock().await;
    if entry.deleted.load(Ordering::Acquire) {
        return Err(AppError::UnknownSession);
    }

    let deleted = state
        .persistence
        .delete_session(session_id)
        .await
        .map_err(AppError::from_persistence)?;
    if !deleted {
        return Err(AppError::UnknownSession);
    }
    state.sessions.remove(session_id).await;
    Ok(StatusCode::NO_CONTENT)
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
    let entry = state.lookup_session(request.session_id).await?;
    let _permit = state
        .inference_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| AppError::ServerBusy)?;

    let mut guard = entry.state.lock().await;
    if entry.deleted.load(Ordering::Acquire) {
        return Err(AppError::UnknownSession);
    }

    let previous = guard.clone();
    let ChatOutcome {
        text,
        backend,
        turn_count,
        state_version,
    } = process_turn(&mut guard, &request.message, state.backend.as_ref())?;
    let next = guard.clone();
    let now_ms = state.sessions.now_millis();
    let events = vec![
        NewEvent::new(
            "turn.user_recorded",
            json!({
                "turn_count": turn_count,
                "content_bytes": request.message.len()
            }),
            now_ms,
        ),
        NewEvent::new(
            "turn.assistant_recorded",
            json!({
                "backend_id": backend.id,
                "state_version": state_version
            }),
            now_ms,
        ),
    ];

    if let Err(error) = state
        .persistence
        .append_events(
            request.session_id,
            previous.version,
            events,
            next,
            now_ms,
        )
        .await
    {
        *guard = previous;
        return Err(AppError::from_persistence(error));
    }

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
    #[error("session state changed concurrently; retry the request")]
    StateConflict,
    #[error("durable storage is unavailable")]
    StorageUnavailable,
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Persistence(PersistenceError),
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

    fn from_persistence(error_value: PersistenceError) -> Self {
        match &error_value {
            PersistenceError::Store(StoreError::VersionConflict { .. }) => Self::StateConflict,
            PersistenceError::Store(StoreError::SessionNotFound(_)) => Self::UnknownSession,
            _ => {
                error!(error = %error_value, "durable storage operation failed");
                Self::Persistence(error_value)
            }
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
            Self::StateConflict => (StatusCode::CONFLICT, "state_conflict"),
            Self::StorageUnavailable | Self::Persistence(_) => {
                (StatusCode::SERVICE_UNAVAILABLE, "storage_unavailable")
            }
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
        let message = match self {
            Self::Persistence(_) => "durable storage is unavailable".to_owned(),
            other => other.to_string(),
        };
        let body = ErrorEnvelope {
            error: ErrorBody { code, message },
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
    use serde_json::Value;
    use std::sync::atomic::AtomicBool;
    use tempfile::tempdir;
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

    async fn create_via_api(router: &Router) -> SessionId {
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/v1/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        let body = json_body(response).await;
        serde_json::from_value(body["session_id"].clone()).unwrap()
    }

    #[tokio::test]
    async fn liveness_and_readiness_report_storage() {
        let router = app(AppState::default());
        let ready = router
            .oneshot(Request::get("/health/ready").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::OK);
        let body = json_body(ready).await;
        assert_eq!(body["runtime"], "rust");
        assert_eq!(body["backend"], "ready");
        assert_eq!(body["storage"], "ready");
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
    async fn malformed_json_and_media_type_are_machine_readable() {
        let router = app(AppState::default());
        let malformed = router
            .clone()
            .oneshot(
                Request::post("/api/v1/chat")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(malformed).await["error"]["code"], "invalid_json");

        let media = router
            .oneshot(
                Request::post("/api/v1/chat")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(media.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn sessions_are_isolated_and_persisted() {
        let state = AppState::default();
        let router = app(state.clone());
        let session_a = create_via_api(&router).await;
        let session_b = create_via_api(&router).await;

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
        assert!(a.user_model.topics.contains("software"));
        assert!(!a.user_model.topics.contains("family"));
        assert!(b.user_model.topics.contains("family"));
        assert!(!b.user_model.topics.contains("work"));

        assert_eq!(
            state
                .persistence
                .load_session(session_a)
                .await
                .unwrap()
                .unwrap()
                .snapshot,
            a
        );
    }

    #[tokio::test]
    async fn reset_and_delete_are_durable() {
        let state = AppState::default();
        let router = app(state.clone());
        let session_id = create_via_api(&router).await;
        assert_eq!(
            router
                .clone()
                .oneshot(chat_request(session_id, "Rust work"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

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
        assert!(reset_body["state_version"].as_u64().unwrap() > 0);

        let stored = state
            .persistence
            .load_session(session_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.snapshot.turn_count, 0);

        let deleted = router
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/sessions/{session_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
        assert!(state
            .persistence
            .load_session(session_id)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn runtime_recovers_sessions_after_restart() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("runtime.sqlite3");
        let store = Arc::new(SqliteStore::open(&path).unwrap());
        let first = AppState::restore(
            Arc::new(HeuristicBackend),
            AppConfig::default(),
            store.clone(),
        )
        .await
        .unwrap();
        let first_router = app(first);
        let session_id = create_via_api(&first_router).await;
        assert_eq!(
            first_router
                .clone()
                .oneshot(chat_request(session_id, "Rust work"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        drop(first_router);
        drop(store);

        let reopened = Arc::new(SqliteStore::open(&path).unwrap());
        let second = AppState::restore(
            Arc::new(HeuristicBackend),
            AppConfig::default(),
            reopened,
        )
        .await
        .unwrap();
        let recovered = second.sessions.snapshot(session_id).await.unwrap();
        assert_eq!(recovered.turn_count, 1);
        assert!(recovered.user_model.topics.contains("software"));
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
    async fn expiration_removes_durable_session() {
        let clock = Arc::new(ManualClock::default());
        let sessions = SessionManager::with_clock(Duration::from_millis(100), clock.clone());
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let state = AppState::with_store(
            Arc::new(HeuristicBackend),
            sessions,
            1,
            store.clone(),
        );
        let router = app(state.clone());
        let session_id = create_via_api(&router).await;
        clock.advance(101);

        let response = router
            .oneshot(chat_request(session_id, "hello"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(store.load_session(session_id).unwrap().is_none());
    }

    #[tokio::test]
    async fn optimistic_conflict_rolls_back_memory() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let state = AppState::with_store(
            Arc::new(HeuristicBackend),
            SessionManager::default(),
            1,
            store.clone(),
        );
        let router = app(state.clone());
        let session_id = create_via_api(&router).await;
        let before = state.sessions.snapshot(session_id).await.unwrap();

        let mut external = before.clone();
        process_turn(&mut external, "external update", &HeuristicBackend).unwrap();
        store
            .append_events(
                session_id,
                before.version,
                &[NewEvent::new("external.update", json!({}), 1)],
                &external,
                1,
            )
            .unwrap();

        let response = router
            .oneshot(chat_request(session_id, "local update"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(response).await["error"]["code"], "state_conflict");
        assert_eq!(state.sessions.snapshot(session_id).await.unwrap(), before);
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
    async fn inference_concurrency_remains_bounded() {
        let started = Arc::new(AtomicBool::new(false));
        let state = AppState::with_components(
            Arc::new(SlowBackend {
                started: started.clone(),
            }),
            SessionManager::default(),
            1,
        );
        let router = app(state);
        let session_a = create_via_api(&router).await;
        let session_b = create_via_api(&router).await;

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
        assert_eq!(first.await.unwrap().status(), StatusCode::OK);
    }
}
