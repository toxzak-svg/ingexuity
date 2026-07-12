CREATE TABLE sessions (
    session_id      TEXT PRIMARY KEY NOT NULL,
    version         INTEGER NOT NULL CHECK (version >= 0),
    snapshot_json   TEXT NOT NULL,
    created_at_ms   INTEGER NOT NULL CHECK (created_at_ms >= 0),
    updated_at_ms   INTEGER NOT NULL CHECK (updated_at_ms >= created_at_ms)
);

CREATE TABLE events (
    session_id      TEXT NOT NULL,
    sequence        INTEGER NOT NULL CHECK (sequence > 0),
    event_id        TEXT NOT NULL UNIQUE,
    event_type      TEXT NOT NULL CHECK (length(event_type) > 0),
    payload_json    TEXT NOT NULL,
    occurred_at_ms  INTEGER NOT NULL CHECK (occurred_at_ms >= 0),
    PRIMARY KEY (session_id, sequence),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX events_by_session_time
    ON events(session_id, occurred_at_ms, sequence);
