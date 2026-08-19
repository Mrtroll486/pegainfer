//! DSV4F model-line detection and the intentionally non-serving G0 launch.

use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

use crate::Dsv4Config;
use crate::Dsv4Manifest;

pub static MODEL_LINE: Dsv4fLine = Dsv4fLine;

pub struct Dsv4fLine;

impl ModelLine for Dsv4fLine {
    fn name(&self) -> &'static str {
        "DeepSeek-V4-Flash"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        crate::config::probe_root_config(config).map_err(|error| error.to_string())
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let config = Dsv4Config::load(ctx.model_path)?;
        let report = Dsv4Manifest::inspect(ctx.model_path, &config)?;
        anyhow::bail!(
            "DSV4F G0 validation passed ({} target tensors, {} DSpark tensors skipped, {} bytes); execution runtime is not implemented until G1-G4",
            report.target_tensor_count,
            report.dspark_skip_tensor_count,
            report.source_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_claims_only_the_pinned_v4_flash_shape() {
        let good: serde_json::Value =
            serde_json::from_str(include_str!("../test_data/config.json")).unwrap();
        MODEL_LINE.probe(&good).unwrap();

        let mut wrong = good;
        wrong["hidden_size"] = serde_json::Value::from(8192);
        let error = MODEL_LINE.probe(&wrong).unwrap_err();
        assert!(error.contains("hidden_size"), "{error}");
    }
}
