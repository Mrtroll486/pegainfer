//! DeepSeek V4 Flash model-line bring-up.
//!
//! G0 is deliberately CPU-only: it validates the checkpoint's two config
//! sources and every safetensors header before later gates allocate GPU state.

pub mod config;
pub mod manifest;

#[cfg(feature = "server")]
pub mod model_line;

pub use config::Dsv4Config;
pub use manifest::Dsv4Manifest;
pub use manifest::G0Report;
