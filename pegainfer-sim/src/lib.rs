use std::collections::HashMap;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::SpecDecodeCounters;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::spawn_scheduler;

pub mod profile;
pub mod worker;

use profile::EngineProfile;
use profile::OutOfDomainPolicy;
use worker::CancelResult;
use worker::GeneratedToken;
use worker::RequestRejection;
use worker::StepId;
use worker::StepOutcome;
use worker::SubmissionResult;
use worker::WorkerRequest;
use worker::WorkerState;

/// Cap on how long `step` parks while waiting for the next due token. New
/// submissions only drain between steps, so a full TTFT/TPOT sleep would
/// stall admission; 1ms keeps the CPU-only sim from spinning a core.
const WAIT_SLICE: Duration = Duration::from_millis(1);

#[derive(Clone, Debug)]
pub struct SimulatedEngineConfig {
    base_ttft_ms: f64,
    prefill_tokens_per_ms: f64,
    tpot_ms: f64,
    fallback_token_id: u32,
    /// Explicit completion token-id sequence to replay verbatim. Empty (the
    /// default) keeps the legacy behaviour of cycling the prompt tokens.
    scripted_completion: Vec<u32>,
    /// A pretend drafter: `(K, accepted per verify step)`; `None` is no drafter.
    spec_decode: Option<(usize, usize)>,
    profile: Option<ProfileConfig>,
}

#[derive(Clone, Debug)]
struct ProfileConfig {
    profile: EngineProfile,
    out_of_domain: OutOfDomainPolicy,
}

impl SimulatedEngineConfig {
    pub fn new(
        base_ttft_ms: f64,
        prefill_tokens_per_ms: f64,
        tpot_ms: f64,
        fallback_token_id: u32,
    ) -> Result<Self> {
        ensure!(
            base_ttft_ms.is_finite() && base_ttft_ms >= 0.0,
            "base TTFT must be finite and non-negative"
        );
        ensure!(
            prefill_tokens_per_ms.is_finite() && prefill_tokens_per_ms > 0.0,
            "prefill throughput must be finite and positive"
        );
        ensure!(
            tpot_ms.is_finite() && tpot_ms >= 0.0,
            "TPOT must be finite and non-negative"
        );

        Ok(Self {
            base_ttft_ms,
            prefill_tokens_per_ms,
            tpot_ms,
            fallback_token_id,
            scripted_completion: Vec::new(),
            spec_decode: None,
            profile: None,
        })
    }

    /// Make every decode step a verify step, so the metrics path has a drafter
    /// without a GPU or a checkpoint. Panics on a `K` past `MAX_SPEC_TOKENS`.
    #[must_use]
    pub fn with_speculative_decoding(
        mut self,
        num_spec_tokens: usize,
        num_accepted: usize,
    ) -> Self {
        assert!(
            num_accepted <= num_spec_tokens,
            "a verify step cannot accept more draft tokens than it proposed"
        );
        SpecDecodeCounters::new(num_spec_tokens).expect("K within MAX_SPEC_TOKENS");
        self.spec_decode = Some((num_spec_tokens, num_accepted));
        self
    }

    /// Replay `ids` verbatim as the completion for every request
    #[must_use]
    pub fn with_scripted_completion(mut self, ids: Vec<u32>) -> Self {
        self.scripted_completion = ids;
        self
    }

    /// Use a validated engine profile for online step timing and scheduling.
    /// CLI loading belongs to the following commit; this entry point keeps
    /// the profiled path injectable for the scheduler and focused tests.
    pub fn with_engine_profile(
        mut self,
        profile: EngineProfile,
        out_of_domain: OutOfDomainPolicy,
    ) -> Result<Self> {
        profile.validate()?;
        self.profile = Some(ProfileConfig {
            profile,
            out_of_domain,
        });
        Ok(self)
    }

    fn ttft(&self, prompt_tokens: usize) -> Duration {
        duration_from_ms(self.base_ttft_ms + prompt_tokens as f64 / self.prefill_tokens_per_ms)
    }

    fn tpot(&self) -> Duration {
        duration_from_ms(self.tpot_ms)
    }
}

impl Default for SimulatedEngineConfig {
    fn default() -> Self {
        Self {
            base_ttft_ms: 5.0,
            prefill_tokens_per_ms: 100.0,
            tpot_ms: 12.0,
            fallback_token_id: 0,
            scripted_completion: Vec::new(),
            spec_decode: None,
            profile: None,
        }
    }
}

/// One scheduler, no KV, no LoRA. `partitions` is the frontend-visible engine
/// count (tests that declare N engines must spawn N schedulers).
pub fn start_engine(config: &SimulatedEngineConfig) -> Engine {
    start_engine_with_partitions(config, 1)
}

pub fn start_engine_with_partitions(config: &SimulatedEngineConfig, partitions: usize) -> Engine {
    assert!(
        partitions > 0,
        "an engine must expose at least one scheduler"
    );
    Engine {
        schedulers: (0..partitions)
            .map(|index| {
                spawn_scheduler(
                    &format!("pegainfer-sim-{index}"),
                    SimScheduler::new(config.clone()),
                )
            })
            .collect(),
        info: EngineInfo {
            kv_capacity: None,
            servable_len: None,
        },
        lora: None,
    }
}

struct SimScheduler {
    config: SimulatedEngineConfig,
    queued: Vec<QueuedRequest>,
    running: Vec<RunningRequest>,
    spec_decode: Option<SpecDecodeCounters>,
    profiled: Option<ProfiledRuntime>,
}

struct RunningRequest {
    id: RequestId,
    /// Tokens not yet emitted, stored reversed for cheap pops.
    pending: Vec<u32>,
    next_token_at: Instant,
    finish_reason: FinishReason,
    logprobs: usize,
}

struct ProfiledRuntime {
    profile: EngineProfile,
    out_of_domain: OutOfDomainPolicy,
    worker: WorkerState<RequestId>,
    requests: HashMap<RequestId, ProfiledRequest>,
    in_flight: Option<ProfiledInFlight>,
}

struct ProfiledRequest {
    completion_tokens: Vec<u32>,
    finish_reason: FinishReason,
    logprobs: usize,
    prompt_echo: Option<PromptEcho>,
}

#[derive(Clone, Copy)]
struct ProfiledInFlight {
    step_id: StepId,
    ready_at: Instant,
    decode_reqs: usize,
}

impl ProfiledRuntime {
    fn new(config: &ProfileConfig) -> Self {
        let worker = WorkerState::new(config.profile.scheduler.clone())
            .expect("profile was validated before scheduler construction");
        Self {
            profile: config.profile.clone(),
            out_of_domain: config.out_of_domain,
            worker,
            requests: HashMap::new(),
            in_flight: None,
        }
    }

    fn step(
        &mut self,
        config: &SimulatedEngineConfig,
        queued: &mut Vec<QueuedRequest>,
        ledger: &mut RequestLedger,
        spec_decode: &mut Option<SpecDecodeCounters>,
    ) -> Result<()> {
        self.drain_submissions(config, queued, ledger)?;
        self.cancel_aborted(ledger)?;

        if let Some(in_flight) = self.in_flight {
            let now = Instant::now();
            if in_flight.ready_at > now {
                std::thread::sleep((in_flight.ready_at - now).min(WAIT_SLICE));
                return Ok(());
            }
            let outcome = self.worker.complete_step(in_flight.step_id)?;
            self.apply_outcome(outcome, in_flight.decode_reqs, config, ledger, spec_decode)?;
            self.in_flight = None;
            return Ok(());
        }

        let Some(plan) = self.worker.plan_step()? else {
            return Ok(());
        };
        let step_id = plan.id();
        let shape = plan.shape();
        let estimate = self.profile.estimate_step(shape, self.out_of_domain)?;
        for &request_id in plan.admitted() {
            let request = self
                .requests
                .get_mut(&request_id)
                .with_context(|| format!("missing profiled request {request_id}"))?;
            ledger.admit(request_id);
            if request
                .prompt_echo
                .as_ref()
                .is_some_and(|echo| echo.ids.is_empty())
            {
                ledger.echo_prompt(
                    request_id,
                    request
                        .prompt_echo
                        .take()
                        .expect("empty prompt echo was checked above"),
                );
            }
        }
        let ready_at = Instant::now()
            .checked_add(duration_from_us(estimate.duration_us))
            .context("profiled worker step deadline overflow")?;
        self.in_flight = Some(ProfiledInFlight {
            step_id,
            ready_at,
            decode_reqs: plan.decode().len(),
        });
        Ok(())
    }

    fn drain_submissions(
        &mut self,
        config: &SimulatedEngineConfig,
        queued: &mut Vec<QueuedRequest>,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        for QueuedRequest { id, request } in std::mem::take(queued) {
            if ledger.is_aborted(id) {
                ledger.retire(id);
                continue;
            }
            let prompt_tokens = u32::try_from(request.prompt_tokens.len())
                .context("profiled request prompt length exceeds u32")?;
            let (mut completion_tokens, finish_reason) =
                planned_completion(config, &request.prompt_tokens, request.max_tokens);
            completion_tokens.reverse();
            let output_tokens =
                u32::try_from(completion_tokens.len()).context("profiled output exceeds u32")?;
            match self.worker.submit(WorkerRequest {
                id,
                prompt_tokens,
                output_tokens,
            })? {
                SubmissionResult::Queued => {
                    let previous = self.requests.insert(
                        id,
                        ProfiledRequest {
                            completion_tokens,
                            finish_reason,
                            logprobs: request.logprobs,
                            prompt_echo: request.echo.then_some(PromptEcho {
                                logprobs: vec![None; request.prompt_tokens.len()],
                                ids: request.prompt_tokens,
                            }),
                        },
                    );
                    ensure!(
                        previous.is_none(),
                        "profiled request metadata was duplicated"
                    );
                }
                SubmissionResult::Finished => {
                    let prompt_len = request.prompt_tokens.len();
                    ledger.admit(id);
                    if request.echo {
                        ledger.echo_prompt(
                            id,
                            PromptEcho {
                                ids: request.prompt_tokens,
                                logprobs: vec![None; prompt_len],
                            },
                        );
                    }
                    ledger.finish(id, finish_reason);
                }
                SubmissionResult::Rejected(rejection) => {
                    ledger.reject(id, reject_reason(rejection, &request));
                }
            }
        }
        Ok(())
    }

    fn cancel_aborted(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let ids: Vec<_> = self.requests.keys().copied().collect();
        for id in ids {
            if !ledger.is_aborted(id) {
                continue;
            }
            match self.worker.cancel(id) {
                CancelResult::Cancelled => {
                    ledger.retire(id);
                    self.requests.remove(&id);
                }
                CancelResult::Deferred | CancelResult::AlreadyRequested => {}
                CancelResult::NotFound => {
                    bail!("profiled request {id} disappeared before cancellation")
                }
            }
        }
        Ok(())
    }

    fn apply_outcome(
        &mut self,
        outcome: StepOutcome<RequestId>,
        decode_reqs: usize,
        config: &SimulatedEngineConfig,
        ledger: &mut RequestLedger,
        spec_decode: &mut Option<SpecDecodeCounters>,
    ) -> Result<()> {
        let mut cancelled = outcome.cancelled;
        let generated_ids: Vec<_> = outcome
            .generated
            .iter()
            .map(|token| token.request_id)
            .collect();
        for request_id in generated_ids {
            if ledger.is_aborted(request_id) && !cancelled.contains(&request_id) {
                cancelled.push(request_id);
            }
        }
        for &request_id in &cancelled {
            if ledger.is_active(request_id) {
                ledger.retire(request_id);
            }
            self.requests.remove(&request_id);
        }

        for progress in outcome.prefill {
            if progress.remaining_tokens != 0 || cancelled.contains(&progress.request_id) {
                continue;
            }
            let request = self
                .requests
                .get_mut(&progress.request_id)
                .with_context(|| format!("missing profiled request {}", progress.request_id))?;
            if let Some(prompt_echo) = request.prompt_echo.take() {
                ledger.echo_prompt(progress.request_id, prompt_echo);
            }
        }
        for GeneratedToken {
            request_id,
            token_index,
        } in outcome.generated
        {
            if cancelled.contains(&request_id) {
                continue;
            }
            let request = self
                .requests
                .get(&request_id)
                .with_context(|| format!("missing profiled request {request_id}"))?;
            let token_index = token_index
                .checked_sub(1)
                .context("profiled worker returned an invalid zero token index")?;
            let token = *request
                .completion_tokens
                .get(usize::try_from(token_index).context("token index overflow")?)
                .with_context(|| {
                    format!("missing token {token_index} for profiled request {request_id}")
                })?;
            let logprobs = if request.logprobs > 0 {
                vec![Some(TokenLogprob {
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                })]
            } else {
                Vec::new()
            };
            ledger.push_tokens(request_id, &[token], &logprobs);
        }
        for request_id in outcome.finished {
            if cancelled.contains(&request_id) {
                continue;
            }
            let request = self
                .requests
                .remove(&request_id)
                .with_context(|| format!("missing profiled request {request_id}"))?;
            ledger.finish(request_id, request.finish_reason);
        }
        if let (Some(counters), Some((k, accepted))) = (spec_decode.as_mut(), config.spec_decode) {
            for _ in 0..decode_reqs {
                counters.observe_draft(k, accepted);
            }
        }
        Ok(())
    }

    fn metrics(&self, spec_decode: Option<&SpecDecodeCounters>) -> SchedulerMetrics {
        SchedulerMetrics {
            num_running_reqs: self.worker.running_len() as u64,
            num_waiting_reqs: self.worker.waiting_len() as u64,
            spec_decode: spec_decode.copied(),
            ..SchedulerMetrics::default()
        }
    }
}

fn reject_reason(rejection: RequestRejection, request: &Request) -> RejectReason {
    match rejection {
        RequestRejection::ModelLengthExceeded {
            total_tokens: _,
            max_model_len,
        } => RejectReason::ContextLength {
            prompt_tokens: request.prompt_tokens.len(),
            max_tokens: request.max_tokens,
            limit: max_model_len as usize,
        },
        // The frontend contract has no generic step-budget refusal yet. The
        // context-length variant still gives the caller a typed rejection and
        // a non-retryable admission result; CLI/profile validation will keep
        // this case out of normal target profiles.
        RequestRejection::WholePrefillExceedsStepBudget {
            prompt_tokens,
            max_num_batched_tokens,
        } => RejectReason::ContextLength {
            prompt_tokens: prompt_tokens as usize,
            max_tokens: request.max_tokens,
            limit: max_num_batched_tokens as usize,
        },
    }
}

impl SimScheduler {
    fn new(config: SimulatedEngineConfig) -> Self {
        let spec_decode = config
            .spec_decode
            .map(|(k, _)| SpecDecodeCounters::new(k).expect("K checked at config time"));
        let profiled = config.profile.as_ref().map(ProfiledRuntime::new);
        Self {
            config,
            queued: Vec::new(),
            running: Vec::new(),
            spec_decode,
            profiled,
        }
    }

    fn park_if_waiting(&self) {
        let Some(next) = self.running.iter().map(|r| r.next_token_at).min() else {
            return;
        };
        let now = Instant::now();
        if next <= now {
            return;
        }
        std::thread::sleep((next - now).min(WAIT_SLICE));
    }
}

impl Scheduler for SimScheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.queued.push(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        if self.profiled.is_some() {
            let mut profiled = self
                .profiled
                .take()
                .expect("profiled runtime presence was checked above");
            let result = profiled.step(
                &self.config,
                &mut self.queued,
                ledger,
                &mut self.spec_decode,
            );
            self.profiled = Some(profiled);
            return result;
        }
        self.step_legacy(ledger);
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        if let Some(profiled) = &self.profiled {
            return profiled.metrics(self.spec_decode.as_ref());
        }
        SchedulerMetrics {
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.queued.len() as u64,
            spec_decode: self.spec_decode,
            ..SchedulerMetrics::default()
        }
    }
}

impl SimScheduler {
    fn step_legacy(&mut self, ledger: &mut RequestLedger) {
        for QueuedRequest { id, request } in self.queued.drain(..) {
            if ledger.is_aborted(id) {
                ledger.retire(id);
                continue;
            }
            if request.echo {
                ledger.echo_prompt(
                    id,
                    PromptEcho {
                        ids: request.prompt_tokens.clone(),
                        logprobs: vec![None; request.prompt_tokens.len()],
                    },
                );
            }
            let prompt_len = request.prompt_tokens.len();
            let (pending, finish_reason) =
                planned_completion(&self.config, &request.prompt_tokens, request.max_tokens);
            ledger.admit(id);
            if pending.is_empty() {
                ledger.finish(id, finish_reason);
                continue;
            }
            self.running.push(RunningRequest {
                id,
                pending,
                next_token_at: Instant::now() + self.config.ttft(prompt_len),
                finish_reason,
                logprobs: request.logprobs,
            });
        }

        let now = Instant::now();
        let mut still_running = Vec::new();
        for mut running in self.running.drain(..) {
            if ledger.is_aborted(running.id) {
                ledger.retire(running.id);
                continue;
            }
            if now < running.next_token_at {
                still_running.push(running);
                continue;
            }
            let Some(token) = running.pending.pop() else {
                ledger.finish(running.id, running.finish_reason);
                continue;
            };
            let logprob = (running.logprobs > 0).then_some(TokenLogprob {
                logprob: 0.0,
                top_logprobs: Vec::new(),
            });
            let logprobs = match logprob {
                Some(lp) => vec![Some(lp)],
                None => Vec::new(),
            };
            // Every decode step of a drafted engine is one verify step.
            if let (Some(counters), Some((k, accepted))) =
                (self.spec_decode.as_mut(), self.config.spec_decode)
            {
                counters.observe_draft(k, accepted);
            }
            ledger.push_tokens(running.id, &[token], &logprobs);
            if running.pending.is_empty() {
                ledger.finish(running.id, running.finish_reason);
            } else {
                running.next_token_at = Instant::now() + self.config.tpot();
                still_running.push(running);
            }
        }
        self.running = still_running;
        self.park_if_waiting();
    }
}

/// Remaining tokens (reversed) plus the terminal reason. Empty pending means
/// finish immediately after admit.
fn planned_completion(
    config: &SimulatedEngineConfig,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, FinishReason) {
    let script = &config.scripted_completion;
    let emit_count = if script.is_empty() {
        max_tokens
    } else {
        max_tokens.min(script.len())
    };
    let finish_reason = if !script.is_empty() && emit_count == script.len() {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };
    let mut pending: Vec<u32> = if script.is_empty() {
        (0..emit_count)
            .map(|index| fake_token_id(prompt_tokens, index, config.fallback_token_id))
            .collect()
    } else {
        script[..emit_count].to_vec()
    };
    pending.reverse();
    (pending, finish_reason)
}

fn fake_token_id(prompt_tokens: &[u32], index: usize, fallback_token_id: u32) -> u32 {
    if prompt_tokens.is_empty() {
        return fallback_token_id;
    }
    prompt_tokens[index % prompt_tokens.len()]
}

fn duration_from_ms(ms: f64) -> Duration {
    Duration::from_secs_f64(ms / 1000.0)
}

fn duration_from_us(microseconds: u64) -> Duration {
    Duration::new(
        microseconds / 1_000_000,
        ((microseconds % 1_000_000) * 1_000) as u32,
    )
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::engine::Request;
    use pegainfer_frontend::engine::Terminal;
    use pegainfer_frontend::sampler::SamplingParams;

    use super::*;
    use crate::profile::ENGINE_PROFILE_SCHEMA_VERSION;
    use crate::profile::ParametricFallback;
    use crate::profile::PrefillPolicy;
    use crate::profile::ProfileProvenance;
    use crate::profile::SchedulerPolicy;
    use crate::profile::SchedulerProfile;
    use crate::profile::StepTimingProfile;
    use crate::profile::TimingGrid;

    fn request(prompt_tokens: Vec<u32>, max_tokens: usize, logprobs: usize) -> Request {
        Request {
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            logprobs,
            echo: false,
            trace_parent: None,
            client_label: None,
        }
    }

    fn collect_completion(
        config: &SimulatedEngineConfig,
        req: Request,
    ) -> (Vec<u32>, Option<usize>, Terminal) {
        let mut engine = start_engine(config);
        assert_eq!(engine.schedulers.len(), 1);
        let mut partition = engine.schedulers.remove(0);
        let mut steps = partition.handle.take_steps().expect("step stream");
        let _control = partition.handle.submit(req);

        let mut tokens = Vec::new();
        let mut prompt_tokens = None;
        let mut terminal = None;
        while terminal.is_none() {
            let step = steps.blocking_recv().expect("step message");
            for update in step.updates {
                if let Some(scheduled) = update.scheduled {
                    prompt_tokens = Some(scheduled.prompt_tokens);
                }
                tokens.extend(update.tokens);
                if let Some(t) = update.terminal {
                    terminal = Some(t);
                }
            }
        }
        drop(partition.handle);
        partition.join.join().expect("driver thread exits");
        (tokens, prompt_tokens, terminal.expect("terminal"))
    }

    fn zero_cost_profile() -> EngineProfile {
        EngineProfile {
            schema_version: ENGINE_PROFILE_SCHEMA_VERSION,
            profile_id: "online-test".to_string(),
            provenance: ProfileProvenance {
                target_engine: "test-engine".to_string(),
                engine_version: "0.0.0".to_string(),
                model_id: "test-model".to_string(),
                model_revision: "test-revision".to_string(),
                model_config_sha256: "00".repeat(32),
                gpu: "test-gpu".to_string(),
                server_flags: Vec::new(),
            },
            scheduler: SchedulerProfile {
                policy: SchedulerPolicy::VllmV1,
                max_num_seqs: 2,
                max_num_batched_tokens: 4,
                max_model_len: 32,
                prefill: PrefillPolicy::Whole,
            },
            timing: StepTimingProfile {
                grid: TimingGrid {
                    decode_reqs: vec![0, 2],
                    sum_decode_ctx_tokens: vec![0, 8],
                    prefill_tokens_in_step: vec![0, 4],
                    step_duration_us: vec![0; 8],
                },
                fallback: ParametricFallback {
                    t0_us: 0.0,
                    prefill_token_us: 0.0,
                    decode_request_us: 0.0,
                    decode_context_token_us: 0.0,
                },
            },
        }
    }

    #[test]
    fn fake_token_id_cycles_prompt_tokens() {
        assert_eq!(fake_token_id(&[7, 9], 0, 42), 7);
        assert_eq!(fake_token_id(&[7, 9], 1, 42), 9);
        assert_eq!(fake_token_id(&[7, 9], 2, 42), 7);
        assert_eq!(fake_token_id(&[], 0, 42), 42);
    }

    #[test]
    fn scripted_completion_replays_ids_and_stops() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 0)
            .unwrap()
            .with_scripted_completion(vec![11, 22, 33]);
        let (tokens, _, terminal) = collect_completion(&config, request(vec![7, 9], 8, 0));
        assert_eq!(tokens, [11, 22, 33]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Stop,
                completion_tokens: 3,
                ..
            }
        ));
    }

    #[test]
    fn scripted_completion_truncated_by_max_tokens_is_length() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 0)
            .unwrap()
            .with_scripted_completion(vec![11, 22, 33]);
        let (tokens, _, terminal) = collect_completion(&config, request(vec![7], 2, 0));
        assert_eq!(tokens, [11, 22]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 2,
                ..
            }
        ));
    }

    #[test]
    fn profiled_scheduler_replays_scripted_completion() {
        let config = SimulatedEngineConfig::default()
            .with_engine_profile(zero_cost_profile(), OutOfDomainPolicy::Strict)
            .unwrap()
            .with_scripted_completion(vec![11, 22, 33]);
        let (tokens, prompt_tokens, terminal) =
            collect_completion(&config, request(vec![7, 9], 3, 0));
        assert_eq!(prompt_tokens, Some(2));
        assert_eq!(tokens, [11, 22, 33]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Stop,
                prompt_tokens: 2,
                completion_tokens: 3,
            }
        ));
    }

    #[test]
    fn config_rejects_invalid_timing_values() {
        assert!(SimulatedEngineConfig::new(-1.0, 100.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 0.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 100.0, -1.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(f64::NAN, 100.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, f64::INFINITY, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 100.0, f64::INFINITY, 0).is_err());
    }

    #[test]
    fn simulated_request_emits_scheduled_tokens_and_finished() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 42).unwrap();
        let (tokens, prompt_tokens, terminal) =
            collect_completion(&config, request(vec![7, 9], 3, 1));
        assert_eq!(prompt_tokens, Some(2));
        assert_eq!(tokens, [7, 9, 7]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                prompt_tokens: 2,
                completion_tokens: 3,
            }
        ));
    }

    #[test]
    fn start_engine_with_partitions_exposes_that_many_schedulers() {
        let engine = start_engine_with_partitions(&SimulatedEngineConfig::default(), 3);
        assert_eq!(engine.schedulers.len(), 3);
    }
}
