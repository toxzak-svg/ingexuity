#![forbid(unsafe_code)]

use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ingexuity_core::{
    process_turn, BackendHealth, ChatOutcome, ConversationState, CoreError, HeuristicBackend,
    InferenceBackend, SessionId,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

#[derive(Clone, Default)]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, Arc<Mutex<ConversationState>>>>>,
}

impl SessionManager {
    pub async fn create(&self) -> SessionId {
        let id = Uuid::new_v4();
        let session = Arc::new(Mutex::new(ConversationState::new(id)));
        self.sessions.write().await.insert(id, session);
        id
    }

    pub async fn get(&self, id: SessionId) -> Option<Arc<Mutex<ConversationState>>> {
        self.sessions.read().await.get(&id).cloned()
    }

    pub async fn count(&self) -> usize {
        self.sessions.read().await.len()
    }

    pub async fn snapshot(&self, id: SessionId) -> Option<ConversationState> {
        let session = self.get(id).await?;
        Some(session.lock().await.clone())
    }
}

#[derive(Clone)]
pub struct AppState {
    pub sessions: SessionManager,
    pub backend: Arc<dyn InferenceBackend>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            sessions: SessionManager::default(),
            backend: Arc::new(HeuristicBackend),
        }
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/sessions", post(create_session))
        .route("/api/v1/chat", post(chat))
        .with_state(state)
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    runtime: &'static str,
    backend: BackendHealth,
    sessions: usize,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        runtime: "rust",
        backend: state.backend.health(),
        sessions: state.sessions.count().await,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session_id: SessionId,
}

async fn create_session(
    State(state): State<AppState>,
) -> (StatusCode, Json<CreateSessionResponse>) {
    let session_id = state.sessions.create().await;
    (
        StatusCode::CREATED,
        Json(CreateSessionResponse { session_id }),
    )
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
    let Json(request) = payload.map_err(|_| AppError::InvalidJson)?;
    let session = state
        .sessions
        .get(request.session_id)
        .await
        .ok_or(AppError::UnknownSession)?;

    let mut session = session.lock().await;
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
    #[error("session does not exist")]
    UnknownSession,
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
    fn status_and_code(&self) -> (StatusCode, &'static str) {
        match self {
            Self::InvalidJson => (StatusCode::BAD_REQUEST, "invalid_json"),
            Self::UnknownSession => (StatusCode::NOT_FOUND, "unknown_session"),
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
    use serde_json::{json, Value};
    use tower::ServiceExt;

    async fn json_body(response: axum::response::Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_reports_rust_runtime() {
        let response = app(AppState::default())
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["runtime"], "rust");
        assert_eq!(body["status"], "ok");
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
    async fn unknown_session_returns_not_found() {
        let response = app(AppState::default())
            .oneshot(
                Request::post("/api/v1/chat")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"session_id": Uuid::new_v4(), "message": "hello"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "unknown_session");
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
                .oneshot(
                    Request::post("/api/v1/chat")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            json!({"session_id": session_id, "message": message}).to_string(),
                        ))
                        .unwrap(),
                )
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
        assert!(!b.user_model.topics.contains("software"));
    }

    #[tokio::test]
    async fn empty_message_does_not_advance_state() {
        let state = AppState::default();
        let session_id = state.sessions.create().await;
        let before = state.sessions.snapshot(session_id).await.unwrap();

        let response = app(state.clone())
            .oneshot(
                Request::post("/api/v1/chat")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"session_id": session_id, "message": "   "}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let after = state.sessions.snapshot(session_id).await.unwrap();
        assert_eq!(before, after);
    }
}
