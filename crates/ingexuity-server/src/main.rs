#![forbid(unsafe_code)]

use ingexuity_server::{app, AppState};
use std::{env, net::SocketAddr};
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
    let address: SocketAddr = bind.parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;

    info!(%address, "starting IngExuity Rust server");
    axum::serve(listener, app(AppState::default())).await?;
    Ok(())
}
