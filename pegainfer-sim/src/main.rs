use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use clap::Parser;
use pegainfer_sim::SimulatedEngineConfig;
use pegainfer_sim::profile::ENGINE_PROFILE_SCHEMA_VERSION;
use pegainfer_sim::profile::EngineProfile;
use pegainfer_sim::profile::OutOfDomainPolicy;
use pegainfer_sim::profile::ParametricFallback;
use pegainfer_sim::profile::PrefillPolicy;
use pegainfer_sim::profile::ProfileProvenance;
use pegainfer_sim::profile::SchedulerPolicy;
use pegainfer_sim::profile::SchedulerProfile;
use pegainfer_sim::profile::StepTimingProfile;
use pegainfer_sim::profile::TimingGrid;
use pegainfer_sim::start_engine;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-0.6B";
const DEFAULT_MAX_MODEL_LEN: u32 = 8192;
const DEFAULT_BASE_TTFT_MS: f64 = 5.0;
const DEFAULT_PREFILL_TOKENS_PER_MS: f64 = 100.0;
const DEFAULT_TPOT_MS: f64 = 12.0;
const DEFAULT_FALLBACK_TOKEN_ID: u32 = 0;
const LEGACY_MAX_NUM_SEQS: u32 = 1024;

#[derive(Parser, Debug)]
#[command(
    name = "pegainfer-sim",
    about = "CPU-only simulated inference server for OpenAI/vLLM serving benchmarks"
)]
struct Args {
    /// Model identity. In legacy mode it also remains the metadata path; with
    /// --profile it must match the profile's target model id.
    #[arg(long)]
    model_id: Option<String>,

    /// Local tokenizer/model metadata directory used by the vLLM frontend.
    /// With --profile, this can differ from the profile's target model id.
    #[arg(long, value_name = "PATH")]
    model_path: Option<PathBuf>,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Max context length reported to the vLLM frontend.
    #[arg(long)]
    max_model_len: Option<u32>,

    /// Fixed TTFT floor before the first fake token.
    #[arg(long)]
    base_ttft_ms: Option<f64>,

    /// Simulated prefill throughput used as prompt_len / throughput.
    #[arg(long)]
    prefill_tokens_per_ms: Option<f64>,

    /// Fixed delay between generated fake tokens.
    #[arg(long)]
    tpot_ms: Option<f64>,

    /// Token id used when a request has an empty prompt-token list.
    #[arg(long, default_value_t = DEFAULT_FALLBACK_TOKEN_ID)]
    fallback_token_id: u32,

    /// Versioned timing and scheduler profile generated from a target engine.
    #[arg(long, value_name = "FILE")]
    profile: Option<PathBuf>,

    /// Reject step shapes outside the profile timing grid instead of using
    /// the profile's parametric fallback.
    #[arg(long)]
    strict: bool,
}

#[derive(Debug)]
struct RuntimeConfig {
    engine: SimulatedEngineConfig,
    model_path: PathBuf,
    served_model_name: Vec<String>,
    max_model_len: u32,
    profile: EngineProfile,
    out_of_domain: OutOfDomainPolicy,
}

fn build_runtime(args: &Args) -> Result<RuntimeConfig> {
    if let Some(path) = &args.profile {
        ensure_legacy_timing_flags_are_absent(args)?;
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read engine profile {}", path.display()))?;
        let profile = EngineProfile::from_json_slice(&bytes)
            .with_context(|| format!("failed to load engine profile {}", path.display()))?;
        let model_id = profile.provenance.model_id.clone();
        if let Some(requested) = &args.model_id {
            ensure!(
                requested == &profile.provenance.model_id,
                "--model-id '{}' conflicts with profile model_id '{}'",
                requested,
                profile.provenance.model_id
            );
        }
        if let Some(requested) = args.max_model_len {
            ensure!(
                requested == profile.scheduler.max_model_len,
                "--max-model-len {} conflicts with profile max_model_len {}",
                requested,
                profile.scheduler.max_model_len
            );
        }
        let out_of_domain = if args.strict {
            OutOfDomainPolicy::Strict
        } else {
            OutOfDomainPolicy::WarnAndFallback
        };
        let engine = SimulatedEngineConfig::default()
            .with_fallback_token_id(args.fallback_token_id)
            .with_engine_profile(profile.clone(), out_of_domain)?;
        let model_path = args.model_path.clone().unwrap_or_else(|| {
            args.model_id
                .as_deref()
                .map_or_else(|| PathBuf::from(&model_id), PathBuf::from)
        });
        return Ok(RuntimeConfig {
            engine,
            model_path,
            served_model_name: vec![model_id],
            max_model_len: profile.scheduler.max_model_len,
            profile,
            out_of_domain,
        });
    }

    ensure!(
        !args.strict,
        "--strict requires --profile; legacy timing has no profile domain to validate"
    );
    let model_id = args
        .model_id
        .clone()
        .or_else(|| {
            args.model_path
                .as_deref()
                .map(|path| path.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
    let model_path = args
        .model_path
        .clone()
        .unwrap_or_else(|| PathBuf::from(&model_id));
    let max_model_len = args.max_model_len.unwrap_or(DEFAULT_MAX_MODEL_LEN);
    let base_ttft_ms = args.base_ttft_ms.unwrap_or(DEFAULT_BASE_TTFT_MS);
    let prefill_tokens_per_ms = args
        .prefill_tokens_per_ms
        .unwrap_or(DEFAULT_PREFILL_TOKENS_PER_MS);
    let tpot_ms = args.tpot_ms.unwrap_or(DEFAULT_TPOT_MS);
    let engine = SimulatedEngineConfig::new(
        base_ttft_ms,
        prefill_tokens_per_ms,
        tpot_ms,
        args.fallback_token_id,
    )?;
    let profile = legacy_profile(
        &model_id,
        max_model_len,
        base_ttft_ms,
        prefill_tokens_per_ms,
        tpot_ms,
    )?;
    let out_of_domain = OutOfDomainPolicy::WarnAndFallback;
    let engine = engine.with_engine_profile(profile.clone(), out_of_domain)?;
    Ok(RuntimeConfig {
        engine,
        model_path,
        served_model_name: if args.model_path.is_some() && args.model_id.is_some() {
            vec![model_id]
        } else {
            Vec::new()
        },
        max_model_len,
        profile,
        out_of_domain,
    })
}

fn ensure_legacy_timing_flags_are_absent(args: &Args) -> Result<()> {
    let provided = [
        ("--base-ttft-ms", args.base_ttft_ms.is_some()),
        (
            "--prefill-tokens-per-ms",
            args.prefill_tokens_per_ms.is_some(),
        ),
        ("--tpot-ms", args.tpot_ms.is_some()),
    ];
    if let Some((name, true)) = provided.into_iter().find(|(_, present)| *present) {
        bail!("{name} cannot be combined with --profile; timing comes from the profile");
    }
    Ok(())
}

fn legacy_profile(
    model_id: &str,
    max_model_len: u32,
    base_ttft_ms: f64,
    prefill_tokens_per_ms: f64,
    tpot_ms: f64,
) -> Result<EngineProfile> {
    ensure!(max_model_len > 0, "max_model_len must be positive");
    let max_num_seqs = LEGACY_MAX_NUM_SEQS;
    let max_num_batched_tokens = max_num_seqs.max(max_model_len);
    let max_context = u64::from(max_num_seqs)
        .checked_mul(u64::from(max_model_len))
        .context("legacy profile context domain overflow")?;
    let base_us = milliseconds_to_micros(base_ttft_ms)?;
    let prefill_token_us = 1_000.0 / prefill_tokens_per_ms;
    let decode_request_us = milliseconds_to_micros(tpot_ms)? as f64;
    let decode_reqs = vec![0, 1, max_num_seqs];
    let sum_decode_ctx_tokens = vec![0, 1, max_context];
    let prefill_tokens_in_step = vec![0, 1, max_num_batched_tokens];
    let mut step_duration_us = Vec::with_capacity(
        decode_reqs.len() * sum_decode_ctx_tokens.len() * prefill_tokens_in_step.len(),
    );
    for &decode in &decode_reqs {
        for &context in &sum_decode_ctx_tokens {
            for &prefill in &prefill_tokens_in_step {
                step_duration_us.push(legacy_step_duration_us(
                    base_us,
                    prefill_token_us,
                    decode_request_us,
                    decode,
                    context,
                    prefill,
                )?);
            }
        }
    }
    Ok(EngineProfile {
        schema_version: ENGINE_PROFILE_SCHEMA_VERSION,
        profile_id: "legacy-cli".to_string(),
        provenance: ProfileProvenance {
            target_engine: "pegainfer-sim".to_string(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            model_id: model_id.to_string(),
            model_revision: "legacy-cli".to_string(),
            model_config_sha256: "00".repeat(32),
            gpu: "cpu".to_string(),
            server_flags: vec!["legacy-timing-options".to_string()],
        },
        scheduler: SchedulerProfile {
            policy: SchedulerPolicy::VllmV1,
            max_num_seqs,
            max_num_batched_tokens,
            max_model_len,
            prefill: PrefillPolicy::Chunked {
                max_chunk_tokens: max_num_batched_tokens,
            },
        },
        timing: StepTimingProfile {
            grid: TimingGrid {
                decode_reqs,
                sum_decode_ctx_tokens,
                prefill_tokens_in_step,
                step_duration_us,
            },
            fallback: ParametricFallback {
                t0_us: base_us as f64,
                prefill_token_us,
                decode_request_us,
                decode_context_token_us: 0.0,
            },
        },
    })
}

fn legacy_step_duration_us(
    base_us: u64,
    prefill_token_us: f64,
    decode_request_us: f64,
    decode_reqs: u32,
    sum_decode_ctx_tokens: u64,
    prefill_tokens: u32,
) -> Result<u64> {
    let mut duration_us = if decode_reqs > 0 && prefill_tokens == 0 && sum_decode_ctx_tokens == 0 {
        base_us as f64
    } else {
        decode_request_us * f64::from(decode_reqs)
    };
    if prefill_tokens > 0 {
        duration_us += base_us as f64 + prefill_token_us * f64::from(prefill_tokens);
    }
    ensure!(
        duration_us.is_finite() && duration_us >= 0.0 && duration_us < u64::MAX as f64,
        "legacy profile timing overflow"
    );
    Ok(duration_us.round() as u64)
}

fn milliseconds_to_micros(milliseconds: f64) -> Result<u64> {
    ensure!(
        milliseconds.is_finite() && milliseconds >= 0.0,
        "timing values must be finite and non-negative"
    );
    let micros = milliseconds * 1_000.0;
    ensure!(
        micros < u64::MAX as f64,
        "timing value is too large to represent in microseconds"
    );
    Ok(micros.round() as u64)
}

fn report_profile(runtime: &RuntimeConfig) {
    let scheduler = &runtime.profile.scheduler;
    eprintln!(
        "active engine profile: id={} target={} version={} model={} revision={} gpu={} scheduler={:?} max_num_seqs={} max_num_batched_tokens={} max_model_len={} timing_domain={:?} out_of_domain={:?}",
        runtime.profile.profile_id,
        runtime.profile.provenance.target_engine,
        runtime.profile.provenance.engine_version,
        runtime.profile.provenance.model_id,
        runtime.profile.provenance.model_revision,
        runtime.profile.provenance.gpu,
        scheduler.policy,
        scheduler.max_num_seqs,
        scheduler.max_num_batched_tokens,
        scheduler.max_model_len,
        runtime.profile.timing.grid.domain(),
        runtime.out_of_domain,
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let runtime = build_runtime(&args)?;
    report_profile(&runtime);
    let engine = start_engine(&runtime.engine);

    pegainfer_frontend::vllm::serve(
        std::future::ready(Ok(engine.into())),
        &runtime.model_path,
        runtime.served_model_name,
        args.port,
        Some(runtime.max_model_len),
        pegainfer_frontend::vllm::shutdown_token_from_ctrl_c(),
    )
    .await
}
