#![forbid(unsafe_code)]

use ingexuity_core::HeuristicBackend;
use ingexuity_server::{app, AppConfig, AppState};
use ingexuity_store::SqliteStore;
use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let bind = env::var("INGEXUITY_BIND").unwrap_or_else(|_| "0.0.0.0:8000".to_owned());
    let database_path = PathBuf::from(
        env::var("INGEXUITY_DB_PATH").unwrap_or_else(|_| "data/ingexuity.sqlite3".to_owned()),
    );
    create_parent_directory(&database_path)?;

    let store = Arc::new(SqliteStore::open(&database_path)?);
    let state = AppState::restore(
        Arc::new(HeuristicBackend),
        AppConfig::default(),
        store,
    )
    .await?;

    let address: SocketAddr = bind.parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;

    info!(%address, database = %database_path.display(), "starting IngExuity Rust server");
    axum::serve(listener, app(state)).await?;
    Ok(())
}

fn create_parent_directory(path: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
