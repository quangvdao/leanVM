use tracing_forest::{ForestLayer, util::LevelFilter};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use tracing_chrome::ChromeLayerBuilder;

/// Initialize tracing for the project.
///
/// - Always installs the `tracing-forest` text logger.
/// - If the `TRACE_CHROME` env var is set, also installs a `tracing-chrome`
///   layer and returns a guard that must be kept alive until tracing is done.
///   Dropping the guard flushes and finalizes the Chrome trace file
///   (Perfetto-compatible).
pub fn init_tracing() -> Option<impl Drop> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    // If TRACE_CHROME is set, also emit a Chrome trace file that can be
    // viewed in Perfetto (trace-<pid>.json by default).
    if std::env::var_os("TRACE_CHROME").is_some() {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        let _ = Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .with(chrome_layer)
            .try_init();
        Some(guard)
    } else {
        let _ = Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .try_init();
        None
    }
}
