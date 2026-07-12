use ingexuity_core::{ConversationState, SessionId};
use ingexuity_store::{NewEvent, SqliteStore, StoreError, StoredSession};
use std::sync::Arc;
use tokio::task::JoinError;

#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("persistence worker failed: {0}")]
    Worker(#[from] JoinError),
}

#[derive(Clone)]
pub struct Persistence {
    store: Arc<SqliteStore>,
}

impl Persistence {
    #[must_use]
    pub fn new(store: Arc<SqliteStore>) -> Self {
        Self { store }
    }

    pub async fn quick_check(&self) -> Result<(), PersistenceError> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.quick_check()).await??;
        Ok(())
    }

    pub async fn list_sessions(&self) -> Result<Vec<StoredSession>, PersistenceError> {
        let store = self.store.clone();
        Ok(tokio::task::spawn_blocking(move || store.list_sessions()).await??)
    }

    pub async fn create_session(
        &self,
        snapshot: ConversationState,
        created_at_ms: u64,
    ) -> Result<(), PersistenceError> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.create_session(&snapshot, created_at_ms))
            .await??;
        Ok(())
    }

    pub async fn append_events(
        &self,
        session_id: SessionId,
        expected_version: u64,
        events: Vec<NewEvent>,
        snapshot: ConversationState,
        updated_at_ms: u64,
    ) -> Result<u64, PersistenceError> {
        let store = self.store.clone();
        Ok(tokio::task::spawn_blocking(move || {
            store.append_events(
                session_id,
                expected_version,
                &events,
                &snapshot,
                updated_at_ms,
            )
        })
        .await??)
    }

    pub async fn delete_session(&self, session_id: SessionId) -> Result<bool, PersistenceError> {
        let store = self.store.clone();
        Ok(tokio::task::spawn_blocking(move || store.delete_session(session_id)).await??)
    }

    #[cfg(test)]
    pub async fn load_session(
        &self,
        session_id: SessionId,
    ) -> Result<Option<StoredSession>, PersistenceError> {
        let store = self.store.clone();
        Ok(tokio::task::spawn_blocking(move || store.load_session(session_id)).await??)
    }
}
