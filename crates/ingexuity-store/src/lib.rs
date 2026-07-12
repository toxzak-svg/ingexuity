#![forbid(unsafe_code)]

use ingexuity_core::{ConversationState, SessionId};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    path::Path,
    sync::{Mutex, MutexGuard},
};
use thiserror::Error;
use uuid::Uuid;

const MIGRATIONS: &[(u32, &str)] = &[(1, include_str!("../../../migrations/0001_event_store.sql"))];

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("SQLite operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("state serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("stored UUID is invalid: {0}")]
    InvalidUuid(#[from] uuid::Error),
    #[error("store mutex was poisoned")]
    Poisoned,
    #[error("session {0} does not exist")]
    SessionNotFound(SessionId),
    #[error("session {session_id} version conflict: expected {expected}, actual {actual}")]
    VersionConflict {
        session_id: SessionId,
        expected: u64,
        actual: u64,
    },
    #[error("snapshot session {snapshot_id} does not match requested session {requested_id}")]
    SnapshotSessionMismatch {
        requested_id: SessionId,
        snapshot_id: SessionId,
    },
    #[error(
        "stored snapshot version {snapshot_version} does not match row version {stored_version}"
    )]
    SnapshotVersionMismatch {
        stored_version: u64,
        snapshot_version: u64,
    },
    #[error("snapshot version must advance beyond {expected}, got {actual}")]
    NonAdvancingVersion { expected: u64, actual: u64 },
    #[error("at least one event is required for an append")]
    EmptyEventBatch,
    #[error("event type cannot be empty")]
    EmptyEventType,
    #[error("numeric value cannot be represented by SQLite INTEGER")]
    NumericOverflow,
    #[error("stored numeric value was negative")]
    NegativeStoredValue,
    #[error("database integrity check failed: {0}")]
    IntegrityCheckFailed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NewEvent {
    pub event_id: Uuid,
    pub event_type: String,
    pub payload: Value,
    pub occurred_at_ms: u64,
}

impl NewEvent {
    #[must_use]
    pub fn new(event_type: impl Into<String>, payload: Value, occurred_at_ms: u64) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            event_type: event_type.into(),
            payload,
            occurred_at_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredEvent {
    pub session_id: SessionId,
    pub sequence: u64,
    pub event_id: Uuid,
    pub event_type: String,
    pub payload: Value,
    pub occurred_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredSession {
    pub session_id: SessionId,
    pub version: u64,
    pub snapshot: ConversationState,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

pub struct SqliteStore {
    connection: Mutex<Connection>,
}

impl SqliteStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let connection = Connection::open(path)?;
        Self::from_connection(connection)
    }

    pub fn open_in_memory() -> Result<Self, StoreError> {
        let connection = Connection::open_in_memory()?;
        Self::from_connection(connection)
    }

    fn from_connection(connection: Connection) -> Result<Self, StoreError> {
        configure_connection(&connection)?;
        apply_migrations(&connection)?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    pub fn schema_version(&self) -> Result<u32, StoreError> {
        let connection = self.connection()?;
        let version = connection.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        u32::try_from(version).map_err(|_| StoreError::NumericOverflow)
    }

    pub fn quick_check(&self) -> Result<(), StoreError> {
        let connection = self.connection()?;
        let result =
            connection.query_row("PRAGMA quick_check", [], |row| row.get::<_, String>(0))?;
        if result == "ok" {
            Ok(())
        } else {
            Err(StoreError::IntegrityCheckFailed(result))
        }
    }

    pub fn create_session(
        &self,
        snapshot: &ConversationState,
        created_at_ms: u64,
    ) -> Result<(), StoreError> {
        let snapshot_json = serde_json::to_string(snapshot)?;
        let connection = self.connection()?;
        connection.execute(
            "INSERT INTO sessions (
                session_id, version, snapshot_json, created_at_ms, updated_at_ms
             ) VALUES (?1, ?2, ?3, ?4, ?4)",
            params![
                snapshot.session_id.to_string(),
                to_i64(snapshot.version)?,
                snapshot_json,
                to_i64(created_at_ms)?,
            ],
        )?;
        Ok(())
    }

    pub fn load_session(&self, session_id: SessionId) -> Result<Option<StoredSession>, StoreError> {
        let connection = self.connection()?;
        let row = connection
            .query_row(
                "SELECT version, snapshot_json, created_at_ms, updated_at_ms
                 FROM sessions WHERE session_id = ?1",
                [session_id.to_string()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;

        row.map(|(version, snapshot_json, created_at_ms, updated_at_ms)| {
            let snapshot: ConversationState = serde_json::from_str(&snapshot_json)?;
            if snapshot.session_id != session_id {
                return Err(StoreError::SnapshotSessionMismatch {
                    requested_id: session_id,
                    snapshot_id: snapshot.session_id,
                });
            }
            let stored_version = from_i64(version)?;
            if snapshot.version != stored_version {
                return Err(StoreError::SnapshotVersionMismatch {
                    stored_version,
                    snapshot_version: snapshot.version,
                });
            }
            Ok(StoredSession {
                session_id,
                version: stored_version,
                snapshot,
                created_at_ms: from_i64(created_at_ms)?,
                updated_at_ms: from_i64(updated_at_ms)?,
            })
        })
        .transpose()
    }

    pub fn append_events(
        &self,
        session_id: SessionId,
        expected_version: u64,
        events: &[NewEvent],
        new_snapshot: &ConversationState,
        updated_at_ms: u64,
    ) -> Result<u64, StoreError> {
        if events.is_empty() {
            return Err(StoreError::EmptyEventBatch);
        }
        if new_snapshot.session_id != session_id {
            return Err(StoreError::SnapshotSessionMismatch {
                requested_id: session_id,
                snapshot_id: new_snapshot.session_id,
            });
        }
        if new_snapshot.version <= expected_version {
            return Err(StoreError::NonAdvancingVersion {
                expected: expected_version,
                actual: new_snapshot.version,
            });
        }
        if events
            .iter()
            .any(|event| event.event_type.trim().is_empty())
        {
            return Err(StoreError::EmptyEventType);
        }

        let snapshot_json = serde_json::to_string(new_snapshot)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let actual_version = transaction
            .query_row(
                "SELECT version FROM sessions WHERE session_id = ?1",
                [session_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .ok_or(StoreError::SessionNotFound(session_id))?;
        let actual_version = from_i64(actual_version)?;

        if actual_version != expected_version {
            return Err(StoreError::VersionConflict {
                session_id,
                expected: expected_version,
                actual: actual_version,
            });
        }

        let current_sequence = transaction.query_row(
            "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE session_id = ?1",
            [session_id.to_string()],
            |row| row.get::<_, i64>(0),
        )?;
        let current_sequence = from_i64(current_sequence)?;

        for (index, event) in events.iter().enumerate() {
            let sequence = current_sequence
                .checked_add(u64::try_from(index).map_err(|_| StoreError::NumericOverflow)?)
                .and_then(|value| value.checked_add(1))
                .ok_or(StoreError::NumericOverflow)?;
            transaction.execute(
                "INSERT INTO events (
                    session_id, sequence, event_id, event_type, payload_json, occurred_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    session_id.to_string(),
                    to_i64(sequence)?,
                    event.event_id.to_string(),
                    event.event_type.trim(),
                    serde_json::to_string(&event.payload)?,
                    to_i64(event.occurred_at_ms)?,
                ],
            )?;
        }

        let updated = transaction.execute(
            "UPDATE sessions
             SET version = ?1, snapshot_json = ?2, updated_at_ms = ?3
             WHERE session_id = ?4 AND version = ?5",
            params![
                to_i64(new_snapshot.version)?,
                snapshot_json,
                to_i64(updated_at_ms)?,
                session_id.to_string(),
                to_i64(expected_version)?,
            ],
        )?;

        if updated != 1 {
            return Err(StoreError::VersionConflict {
                session_id,
                expected: expected_version,
                actual: actual_version,
            });
        }

        transaction.commit()?;
        Ok(new_snapshot.version)
    }

    pub fn list_events(&self, session_id: SessionId) -> Result<Vec<StoredEvent>, StoreError> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT sequence, event_id, event_type, payload_json, occurred_at_ms
             FROM events WHERE session_id = ?1 ORDER BY sequence ASC",
        )?;
        let rows = statement.query_map([session_id.to_string()], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })?;

        let mut events = Vec::new();
        for row in rows {
            let (sequence, event_id, event_type, payload_json, occurred_at_ms) = row?;
            events.push(StoredEvent {
                session_id,
                sequence: from_i64(sequence)?,
                event_id: Uuid::parse_str(&event_id)?,
                event_type,
                payload: serde_json::from_str(&payload_json)?,
                occurred_at_ms: from_i64(occurred_at_ms)?,
            });
        }
        Ok(events)
    }

    pub fn delete_session(&self, session_id: SessionId) -> Result<bool, StoreError> {
        let connection = self.connection()?;
        let deleted = connection.execute(
            "DELETE FROM sessions WHERE session_id = ?1",
            [session_id.to_string()],
        )?;
        Ok(deleted == 1)
    }

    pub fn session_count(&self) -> Result<u64, StoreError> {
        let connection = self.connection()?;
        let count = connection.query_row("SELECT COUNT(*) FROM sessions", [], |row| {
            row.get::<_, i64>(0)
        })?;
        from_i64(count)
    }

    fn connection(&self) -> Result<MutexGuard<'_, Connection>, StoreError> {
        self.connection.lock().map_err(|_| StoreError::Poisoned)
    }
}

fn configure_connection(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;
         PRAGMA synchronous = FULL;",
    )?;
    let _journal_mode: String =
        connection.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))?;
    Ok(())
}

fn apply_migrations(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version         INTEGER PRIMARY KEY NOT NULL,
            applied_at_ms   INTEGER NOT NULL DEFAULT (
                CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)
            )
        );",
    )?;

    let current = connection.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
        [],
        |row| row.get::<_, i64>(0),
    )?;

    for (version, sql) in MIGRATIONS {
        if i64::from(*version) <= current {
            continue;
        }
        let transaction = connection.unchecked_transaction()?;
        transaction.execute_batch(sql)?;
        transaction.execute(
            "INSERT INTO schema_migrations (version) VALUES (?1)",
            [i64::from(*version)],
        )?;
        transaction.commit()?;
    }
    Ok(())
}

fn to_i64(value: u64) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| StoreError::NumericOverflow)
}

fn from_i64(value: i64) -> Result<u64, StoreError> {
    u64::try_from(value).map_err(|_| StoreError::NegativeStoredValue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingexuity_core::{process_turn, HeuristicBackend};
    use serde_json::json;
    use tempfile::tempdir;

    fn advanced_state(session_id: SessionId) -> ConversationState {
        let mut state = ConversationState::new(session_id);
        process_turn(&mut state, "Help with Rust work?", &HeuristicBackend).unwrap();
        state
    }

    #[test]
    fn migrations_are_idempotent_and_database_is_sound() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert_eq!(store.schema_version().unwrap(), 1);
        store.quick_check().unwrap();
    }

    #[test]
    fn session_survives_store_reopen() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("identity.sqlite3");
        let session_id = Uuid::new_v4();
        let state = ConversationState::new(session_id);

        {
            let store = SqliteStore::open(&path).unwrap();
            store.create_session(&state, 1000).unwrap();
        }

        let reopened = SqliteStore::open(&path).unwrap();
        let loaded = reopened.load_session(session_id).unwrap().unwrap();
        assert_eq!(loaded.snapshot, state);
        assert_eq!(loaded.version, 0);
        assert_eq!(loaded.created_at_ms, 1000);
        reopened.quick_check().unwrap();
    }

    #[test]
    fn append_is_atomic_and_ordered() {
        let store = SqliteStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        let initial = ConversationState::new(session_id);
        store.create_session(&initial, 100).unwrap();
        let next = advanced_state(session_id);
        let events = vec![
            NewEvent::new("turn.user", json!({"text": "redacted"}), 110),
            NewEvent::new("turn.assistant", json!({"backend": "heuristic-v1"}), 111),
        ];

        let version = store
            .append_events(session_id, 0, &events, &next, 112)
            .unwrap();
        assert_eq!(version, next.version);

        let loaded = store.load_session(session_id).unwrap().unwrap();
        assert_eq!(loaded.snapshot, next);
        assert_eq!(loaded.updated_at_ms, 112);

        let stored_events = store.list_events(session_id).unwrap();
        assert_eq!(stored_events.len(), 2);
        assert_eq!(stored_events[0].sequence, 1);
        assert_eq!(stored_events[1].sequence, 2);
        assert_eq!(stored_events[0].event_type, "turn.user");
    }

    #[test]
    fn version_conflict_writes_nothing() {
        let store = SqliteStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        let initial = ConversationState::new(session_id);
        store.create_session(&initial, 100).unwrap();
        let next = advanced_state(session_id);
        let event = NewEvent::new("turn.completed", json!({}), 110);

        let error = store
            .append_events(session_id, 1, &[event], &next, 111)
            .unwrap_err();
        assert!(matches!(error, StoreError::VersionConflict { .. }));
        assert!(store.list_events(session_id).unwrap().is_empty());
        assert_eq!(store.load_session(session_id).unwrap().unwrap().version, 0);
    }

    #[test]
    fn delete_cascades_event_history() {
        let store = SqliteStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        let initial = ConversationState::new(session_id);
        store.create_session(&initial, 100).unwrap();
        let next = advanced_state(session_id);
        store
            .append_events(
                session_id,
                0,
                &[NewEvent::new("turn.completed", json!({}), 110)],
                &next,
                111,
            )
            .unwrap();

        assert!(store.delete_session(session_id).unwrap());
        assert!(store.load_session(session_id).unwrap().is_none());
        assert!(store.list_events(session_id).unwrap().is_empty());
        assert_eq!(store.session_count().unwrap(), 0);
    }

    #[test]
    fn snapshot_identity_and_version_must_be_valid() {
        let store = SqliteStore::open_in_memory().unwrap();
        let session_id = Uuid::new_v4();
        store
            .create_session(&ConversationState::new(session_id), 100)
            .unwrap();

        let wrong = advanced_state(Uuid::new_v4());
        let mismatch = store
            .append_events(
                session_id,
                0,
                &[NewEvent::new("turn.completed", json!({}), 110)],
                &wrong,
                111,
            )
            .unwrap_err();
        assert!(matches!(
            mismatch,
            StoreError::SnapshotSessionMismatch { .. }
        ));

        let unchanged = ConversationState::new(session_id);
        let non_advancing = store
            .append_events(
                session_id,
                0,
                &[NewEvent::new("turn.completed", json!({}), 110)],
                &unchanged,
                111,
            )
            .unwrap_err();
        assert!(matches!(
            non_advancing,
            StoreError::NonAdvancingVersion { .. }
        ));
    }
}
