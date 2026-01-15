use execution_core::{Config, ExecutionCoreImpl, ExecutionCore};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "execution_core=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Execution Core");

    // Load configuration
    let config = Config::load()?;
    tracing::info!("Configuration loaded: {:?}", config);

    // Initialize execution core
    let mut execution_core = ExecutionCoreImpl::new(&config)?;
    tracing::info!("Execution core initialized");

    // Start health check server (simple HTTP server for Docker health checks)
    let health_server = tokio::spawn(async {
        use std::convert::Infallible;
        use std::net::SocketAddr;
        use hyper::{Body, Request, Response, Server};
        use hyper::service::{make_service_fn, service_fn};

        async fn health_check(_req: Request<Body>) -> Result<Response<Body>, Infallible> {
            Ok(Response::new(Body::from("OK")))
        }

        let make_svc = make_service_fn(|_conn| async {
            Ok::<_, Infallible>(service_fn(health_check))
        });

        let addr = SocketAddr::from(([0, 0, 0, 0], 8001));
        let server = Server::bind(&addr).serve(make_svc);

        tracing::info!("Health check server listening on {}", addr);

        if let Err(e) = server.await {
            tracing::error!("Health check server error: {}", e);
        }
    });

    // Main execution loop (placeholder for now)
    let main_loop = tokio::spawn(async move {
        loop {
            // Process events, check risk limits, etc.
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            
            // Example: Check risk limits periodically
            let risk_status = execution_core.check_risk_limits();
            if !risk_status.can_trade() {
                tracing::warn!("Risk limits breached: {:?}", risk_status);
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = health_server => {
            tracing::error!("Health server terminated unexpectedly");
        }
        _ = main_loop => {
            tracing::error!("Main loop terminated unexpectedly");
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received shutdown signal");
        }
    }

    tracing::info!("Execution Core shutting down");
    Ok(())
}