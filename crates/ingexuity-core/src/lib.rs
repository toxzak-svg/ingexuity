#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;
use uuid::Uuid;

pub type SessionId = Uuid;

pub const MAX_MESSAGE_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Turn {
    pub sequence: u64,
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserModel {
    pub topics: BTreeSet<String>,
    pub explicit_preferences: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PredictionStatus {
    Pending,
    Resolved { outcome: String },
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    pub id: Uuid,
    pub target: String,
    pub confidence: f64,
    pub issued_before_turn: u64,
    pub status: PredictionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversationState {
    pub session_id: SessionId,
    pub version: u64,
    pub turn_count: u64,
    pub turns: Vec<Turn>,
    pub user_model: UserModel,
    pub predictions: Vec<Prediction>,
}

impl ConversationState {
    #[must_use]
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            version: 0,
            turn_count: 0,
            turns: Vec::new(),
            user_model: UserModel::default(),
            predictions: Vec::new(),
        }
    }

    pub fn reset(&mut self) -> Result<(), CoreError> {
        let next_version = self.version.checked_add(1).ok_or_else(|| {
            CoreError::Inference(InferenceError::Failed(
                "conversation state version overflowed".to_owned(),
            ))
        })?;
        *self = Self::new(self.session_id);
        self.version = next_version;
        Ok(())
    }

    fn record_user_message(&mut self, message: &str) {
        self.turn_count += 1;
        self.version += 1;
        self.turns.push(Turn {
            sequence: self.turns.len() as u64 + 1,
            role: Role::User,
            content: message.to_owned(),
        });
        self.user_model.topics.extend(classify_topics(message));
    }

    fn record_assistant_message(&mut self, message: &str) {
        self.version += 1;
        self.turns.push(Turn {
            sequence: self.turns.len() as u64 + 1,
            role: Role::Assistant,
            content: message.to_owned(),
        });
    }

    fn issue_next_turn_prediction(&mut self) {
        self.predictions.push(Prediction {
            id: Uuid::new_v4(),
            target: "next_turn.intent".to_owned(),
            confidence: 0.5,
            issued_before_turn: self.turn_count + 1,
            status: PredictionStatus::Pending,
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendMetadata {
    pub id: String,
    pub kind: String,
    pub model: Option<String>,
    pub local: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendHealth {
    Ready,
    Degraded,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerationRequest {
    pub session_id: SessionId,
    pub turn_count: u64,
    pub message: String,
    pub known_topics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Generation {
    pub text: String,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InferenceError {
    #[error("inference backend unavailable")]
    Unavailable,
    #[error("inference timed out")]
    Timeout,
    #[error("inference failed: {0}")]
    Failed(String),
}

pub trait InferenceBackend: Send + Sync {
    fn metadata(&self) -> BackendMetadata;
    fn health(&self) -> BackendHealth;
    fn generate(&self, request: &GenerationRequest) -> Result<Generation, InferenceError>;
}

#[derive(Debug, Default)]
pub struct HeuristicBackend;

impl InferenceBackend for HeuristicBackend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            id: "heuristic-v1".to_owned(),
            kind: "deterministic_fallback".to_owned(),
            model: None,
            local: true,
        }
    }

    fn health(&self) -> BackendHealth {
        BackendHealth::Ready
    }

    fn generate(&self, request: &GenerationRequest) -> Result<Generation, InferenceError> {
        let normalized = request.message.trim().to_lowercase();
        let text = if normalized.contains("overwhelmed") || normalized.contains("stressed") {
            "I hear that this is a lot. I can stay with it and still help with the next concrete step."
        } else if normalized.ends_with('?') {
            "I do not have a language model loaded yet. The Rust runtime is healthy, and this request reached the deterministic fallback correctly."
        } else {
            "The Rust runtime received your message. Model-backed generation is not enabled yet, so I am responding through the deterministic fallback."
        };

        Ok(Generation {
            text: text.to_owned(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatOutcome {
    pub text: String,
    pub backend: BackendMetadata,
    pub turn_count: u64,
    pub state_version: u64,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoreError {
    #[error("message cannot be empty")]
    EmptyMessage,
    #[error("message exceeds {MAX_MESSAGE_BYTES} bytes")]
    MessageTooLarge,
    #[error(transparent)]
    Inference(#[from] InferenceError),
}

pub fn process_turn(
    state: &mut ConversationState,
    message: &str,
    backend: &dyn InferenceBackend,
) -> Result<ChatOutcome, CoreError> {
    let message = message.trim();
    if message.is_empty() {
        return Err(CoreError::EmptyMessage);
    }
    if message.len() > MAX_MESSAGE_BYTES {
        return Err(CoreError::MessageTooLarge);
    }

    // Work on a clone so backend failure cannot partially mutate committed state.
    let mut next = state.clone();
    next.record_user_message(message);
    next.issue_next_turn_prediction();

    let request = GenerationRequest {
        session_id: next.session_id,
        turn_count: next.turn_count,
        message: message.to_owned(),
        known_topics: next.user_model.topics.iter().cloned().collect(),
    };
    let generation = backend.generate(&request)?;
    next.record_assistant_message(&generation.text);

    let outcome = ChatOutcome {
        text: generation.text,
        backend: backend.metadata(),
        turn_count: next.turn_count,
        state_version: next.version,
    };
    *state = next;
    Ok(outcome)
}

fn classify_topics(message: &str) -> BTreeSet<String> {
    let words: BTreeSet<String> = message
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect();

    let mut topics = BTreeSet::new();
    if words
        .iter()
        .any(|word| ["work", "job", "career"].contains(&word.as_str()))
    {
        topics.insert("work".to_owned());
    }
    if words
        .iter()
        .any(|word| ["family", "partner", "kids"].contains(&word.as_str()))
    {
        topics.insert("family".to_owned());
    }
    if words
        .iter()
        .any(|word| ["code", "rust", "software", "programming"].contains(&word.as_str()))
    {
        topics.insert("software".to_owned());
    }
    if words
        .iter()
        .any(|word| ["stressed", "overwhelmed", "sad", "angry"].contains(&word.as_str()))
    {
        topics.insert("emotional_context".to_owned());
    }
    topics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_messages_without_mutating_state() {
        let id = Uuid::new_v4();
        let mut state = ConversationState::new(id);
        let original = state.clone();

        let error = process_turn(&mut state, "   ", &HeuristicBackend).unwrap_err();

        assert_eq!(error, CoreError::EmptyMessage);
        assert_eq!(state, original);
    }

    #[test]
    fn backend_failure_is_transactional() {
        struct FailingBackend;
        impl InferenceBackend for FailingBackend {
            fn metadata(&self) -> BackendMetadata {
                BackendMetadata {
                    id: "failing".to_owned(),
                    kind: "test".to_owned(),
                    model: None,
                    local: true,
                }
            }
            fn health(&self) -> BackendHealth {
                BackendHealth::Unavailable
            }
            fn generate(&self, _: &GenerationRequest) -> Result<Generation, InferenceError> {
                Err(InferenceError::Unavailable)
            }
        }

        let mut state = ConversationState::new(Uuid::new_v4());
        let original = state.clone();
        let error = process_turn(&mut state, "hello", &FailingBackend).unwrap_err();

        assert_eq!(error, CoreError::Inference(InferenceError::Unavailable));
        assert_eq!(state, original);
    }

    #[test]
    fn predictions_are_issued_but_not_self_resolved() {
        let mut state = ConversationState::new(Uuid::new_v4());
        let outcome = process_turn(
            &mut state,
            "Can you help with Rust code?",
            &HeuristicBackend,
        )
        .unwrap();

        assert_eq!(outcome.turn_count, 1);
        assert!(state.user_model.topics.contains("software"));
        assert_eq!(state.predictions.len(), 1);
        assert_eq!(state.predictions[0].status, PredictionStatus::Pending);
        assert_eq!(state.turns.len(), 2);
    }

    #[test]
    fn reset_clears_personal_state_and_advances_version() {
        let mut state = ConversationState::new(Uuid::new_v4());
        process_turn(&mut state, "Rust work", &HeuristicBackend).unwrap();
        let prior_version = state.version;

        state.reset().unwrap();

        assert_eq!(state.turn_count, 0);
        assert!(state.turns.is_empty());
        assert!(state.predictions.is_empty());
        assert!(state.user_model.topics.is_empty());
        assert_eq!(state.version, prior_version + 1);
    }

    #[test]
    fn deterministic_backend_is_repeatable() {
        let backend = HeuristicBackend;
        let request = GenerationRequest {
            session_id: Uuid::nil(),
            turn_count: 1,
            message: "What is happening?".to_owned(),
            known_topics: Vec::new(),
        };

        assert_eq!(backend.generate(&request), backend.generate(&request));
    }
}
