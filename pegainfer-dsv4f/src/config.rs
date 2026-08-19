//! Dual-source DeepSeek V4 Flash configuration validation.

use std::fmt::Debug;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

pub const MODEL_TYPE: &str = "deepseek_v4";
pub const ARCHITECTURE: &str = "DeepseekV4ForCausalLM";
pub const TARGET_LAYERS: usize = 43;
pub const HASH_LAYERS: usize = 3;
pub const MTP_LAYERS: usize = 3;
pub const ROUTED_EXPERTS: usize = 256;
pub const ACTIVATED_EXPERTS: usize = 6;
pub const HIDDEN_SIZE: usize = 4096;
pub const EXPERT_INTERMEDIATE_SIZE: usize = 2048;
pub const VOCAB_SIZE: usize = 129_280;

const ROOT_CONFIG: &str = "config.json";
const INFERENCE_CONFIG: &str = "inference/config.json";

#[derive(Clone, Debug, Serialize)]
pub struct Dsv4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub expert_intermediate_size: usize,
    pub num_layers: usize,
    pub num_hash_layers: usize,
    pub num_mtp_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_routed_experts: usize,
    pub num_shared_experts: usize,
    pub num_activated_experts: usize,
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub max_position_embeddings: usize,
    pub original_seq_len: usize,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub compress_rope_theta: f64,
    pub compress_ratios: Vec<usize>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

#[derive(Debug, Deserialize)]
// This is a wire-format mirror of config.json, not runtime state.
#[allow(clippy::struct_excessive_bools)]
struct RootConfig {
    architectures: Vec<String>,
    attention_bias: bool,
    attention_dropout: f64,
    bos_token_id: u32,
    eos_token_id: u32,
    expert_dtype: String,
    hc_eps: f64,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    index_head_dim: usize,
    index_n_heads: usize,
    index_topk: usize,
    max_position_embeddings: usize,
    model_type: String,
    moe_intermediate_size: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    norm_topk_prob: bool,
    num_attention_heads: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_hash_layers: usize,
    num_key_value_heads: usize,
    num_nextn_predict_layers: usize,
    o_groups: usize,
    o_lora_rank: usize,
    q_lora_rank: usize,
    qk_rope_head_dim: usize,
    quantization_config: QuantizationConfig,
    rms_norm_eps: f64,
    rope_scaling: RopeScaling,
    rope_theta: f64,
    routed_scaling_factor: f64,
    scoring_func: String,
    sliding_window: usize,
    swiglu_limit: f64,
    tie_word_embeddings: bool,
    topk_method: String,
    torch_dtype: String,
    use_cache: bool,
    vocab_size: usize,
    compress_rope_theta: f64,
    compress_ratios: Vec<usize>,
    dspark_block_size: usize,
    dspark_noise_token_id: usize,
    dspark_target_layer_ids: Vec<usize>,
    dspark_markov_rank: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantizationConfig {
    activation_scheme: String,
    fmt: String,
    quant_method: String,
    scale_fmt: String,
    weight_block_size: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RopeScaling {
    beta_fast: f64,
    beta_slow: f64,
    factor: f64,
    original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InferenceConfig {
    vocab_size: usize,
    dim: usize,
    moe_inter_dim: usize,
    n_layers: usize,
    n_hash_layers: usize,
    n_mtp_layers: usize,
    dspark_block_size: usize,
    dspark_noise_token_id: usize,
    dspark_target_layer_ids: Vec<usize>,
    dspark_markov_rank: usize,
    n_heads: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    n_activated_experts: usize,
    score_func: String,
    route_scale: f64,
    swiglu_limit: f64,
    q_lora_rank: usize,
    head_dim: usize,
    rope_head_dim: usize,
    o_groups: usize,
    o_lora_rank: usize,
    window_size: usize,
    original_seq_len: usize,
    rope_theta: f64,
    rope_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    dtype: String,
    scale_fmt: String,
    expert_dtype: String,
    compress_rope_theta: f64,
    compress_ratios: Vec<usize>,
}

impl Dsv4Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let root_path = model_dir.join(ROOT_CONFIG);
        let inference_path = model_dir.join(INFERENCE_CONFIG);
        let root = read_json(&root_path)?;
        let inference = read_json(&inference_path).with_context(|| {
            format!(
                "DSV4F G0 requires the bundled secondary config at {}",
                inference_path.display()
            )
        })?;
        Self::validate_pair(&root, &inference)
    }

    pub fn validate_pair(root_json: &Value, inference_json: &Value) -> Result<Self> {
        let root: RootConfig = serde_json::from_value(root_json.clone())
            .context("parse DSV4F root config.json contract")?;
        let inference: InferenceConfig = serde_json::from_value(inference_json.clone())
            .context("parse DSV4F inference/config.json contract")?;

        validate_root_contract(&root)?;
        validate_inference_contract(&inference)?;
        cross_validate(&root, &inference)?;

        Ok(Self {
            vocab_size: root.vocab_size,
            hidden_size: root.hidden_size,
            expert_intermediate_size: root.moe_intermediate_size,
            num_layers: root.num_hidden_layers,
            num_hash_layers: root.num_hash_layers,
            num_mtp_layers: inference.n_mtp_layers,
            num_attention_heads: root.num_attention_heads,
            num_key_value_heads: root.num_key_value_heads,
            num_routed_experts: root.n_routed_experts,
            num_shared_experts: root.n_shared_experts,
            num_activated_experts: root.num_experts_per_tok,
            q_lora_rank: root.q_lora_rank,
            head_dim: root.head_dim,
            rope_head_dim: root.qk_rope_head_dim,
            o_groups: root.o_groups,
            o_lora_rank: root.o_lora_rank,
            window_size: root.sliding_window,
            max_position_embeddings: root.max_position_embeddings,
            original_seq_len: root.rope_scaling.original_max_position_embeddings,
            rope_theta: root.rope_theta,
            rope_factor: root.rope_scaling.factor,
            beta_fast: root.rope_scaling.beta_fast,
            beta_slow: root.rope_scaling.beta_slow,
            index_n_heads: root.index_n_heads,
            index_head_dim: root.index_head_dim,
            index_topk: root.index_topk,
            hc_mult: root.hc_mult,
            hc_sinkhorn_iters: root.hc_sinkhorn_iters,
            compress_rope_theta: root.compress_rope_theta,
            compress_ratios: root.compress_ratios,
            bos_token_id: root.bos_token_id,
            eos_token_id: root.eos_token_id,
        })
    }
}

pub fn probe_root_config(json: &Value) -> Result<()> {
    let model_type = json.get("model_type").and_then(Value::as_str);
    ensure!(
        model_type == Some(MODEL_TYPE),
        "model_type {model_type:?} is not {MODEL_TYPE:?}"
    );
    let claims_architecture = json
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(ARCHITECTURE)));
    ensure!(
        claims_architecture,
        "architectures must contain {ARCHITECTURE:?}"
    );
    for (field, expected) in [
        ("num_hidden_layers", TARGET_LAYERS),
        ("hidden_size", HIDDEN_SIZE),
        ("n_routed_experts", ROUTED_EXPERTS),
    ] {
        let actual = json.get(field).and_then(Value::as_u64);
        ensure!(
            actual == Some(expected as u64),
            "{field} {actual:?} is not {expected}"
        );
    }
    Ok(())
}

fn validate_root_contract(root: &RootConfig) -> Result<()> {
    ensure!(
        root.model_type == MODEL_TYPE,
        "unsupported model_type {:?}",
        root.model_type
    );
    ensure!(
        root.architectures.len() == 1 && root.architectures[0] == ARCHITECTURE,
        "unsupported architectures {:?}; expected exactly [{ARCHITECTURE:?}]",
        root.architectures
    );
    ensure!(!root.attention_bias, "DSV4F attention_bias must be false");
    ensure_float_value("attention_dropout", root.attention_dropout, 0.0)?;
    ensure!(root.bos_token_id == 0, "DSV4F bos_token_id must be 0");
    ensure!(root.eos_token_id == 1, "DSV4F eos_token_id must be 1");
    ensure!(root.expert_dtype == "fp4", "DSV4F expert_dtype must be fp4");
    ensure_float_value("hc_eps", root.hc_eps, 1.0e-6)?;
    ensure!(root.hc_mult == 4, "DSV4F hc_mult must be 4");
    ensure!(
        root.hc_sinkhorn_iters == 20,
        "DSV4F hc_sinkhorn_iters must be 20"
    );
    ensure!(root.head_dim == 512, "DSV4F head_dim must be 512");
    ensure!(root.hidden_act == "silu", "DSV4F hidden_act must be silu");
    ensure!(
        root.hidden_size == HIDDEN_SIZE,
        "DSV4F hidden_size must be {HIDDEN_SIZE}"
    );
    ensure!(
        root.index_head_dim == 128,
        "DSV4F index_head_dim must be 128"
    );
    ensure!(root.index_n_heads == 64, "DSV4F index_n_heads must be 64");
    ensure!(root.index_topk == 512, "DSV4F index_topk must be 512");
    ensure!(
        root.max_position_embeddings == 1_048_576,
        "DSV4F max_position_embeddings must be 1048576"
    );
    ensure!(
        root.moe_intermediate_size == EXPERT_INTERMEDIATE_SIZE,
        "DSV4F moe_intermediate_size must be {EXPERT_INTERMEDIATE_SIZE}"
    );
    ensure!(
        root.n_routed_experts == ROUTED_EXPERTS,
        "DSV4F n_routed_experts must be {ROUTED_EXPERTS}"
    );
    ensure!(
        root.n_shared_experts == 1,
        "DSV4F n_shared_experts must be 1"
    );
    ensure!(root.norm_topk_prob, "DSV4F norm_topk_prob must be true");
    ensure!(
        root.num_attention_heads == 64,
        "DSV4F num_attention_heads must be 64"
    );
    ensure!(
        root.num_experts_per_tok == ACTIVATED_EXPERTS,
        "DSV4F num_experts_per_tok must be {ACTIVATED_EXPERTS}"
    );
    ensure!(
        root.num_hidden_layers == TARGET_LAYERS,
        "DSV4F num_hidden_layers must be {TARGET_LAYERS}"
    );
    ensure!(
        root.num_hash_layers == HASH_LAYERS,
        "DSV4F num_hash_layers must be {HASH_LAYERS}"
    );
    ensure!(
        root.num_key_value_heads == 1,
        "DSV4F num_key_value_heads must be 1"
    );
    ensure!(
        root.num_nextn_predict_layers == 1,
        "DSV4F root num_nextn_predict_layers must be 1 for the explicit M1 DSpark exception"
    );
    ensure!(root.o_groups == 8, "DSV4F o_groups must be 8");
    ensure!(root.o_lora_rank == 1024, "DSV4F o_lora_rank must be 1024");
    ensure!(root.q_lora_rank == 1024, "DSV4F q_lora_rank must be 1024");
    ensure!(
        root.qk_rope_head_dim == 64,
        "DSV4F qk_rope_head_dim must be 64"
    );
    validate_quantization(&root.quantization_config)?;
    ensure_float_value("rms_norm_eps", root.rms_norm_eps, 1.0e-6)?;
    validate_rope(&root.rope_scaling)?;
    ensure_float_value("rope_theta", root.rope_theta, 10_000.0)?;
    ensure_float_value("routed_scaling_factor", root.routed_scaling_factor, 1.5)?;
    ensure!(
        root.scoring_func == "sqrtsoftplus",
        "DSV4F scoring_func must be sqrtsoftplus"
    );
    ensure!(
        root.sliding_window == 128,
        "DSV4F sliding_window must be 128"
    );
    ensure_float_value("swiglu_limit", root.swiglu_limit, 10.0)?;
    ensure!(
        !root.tie_word_embeddings,
        "DSV4F tie_word_embeddings must be false"
    );
    ensure!(
        root.topk_method == "noaux_tc",
        "DSV4F topk_method must be noaux_tc"
    );
    ensure!(
        root.torch_dtype == "bfloat16",
        "DSV4F torch_dtype must be bfloat16"
    );
    ensure!(root.use_cache, "DSV4F use_cache must be true");
    ensure!(
        root.vocab_size == VOCAB_SIZE,
        "DSV4F vocab_size must be {VOCAB_SIZE}"
    );
    ensure_float_value("compress_rope_theta", root.compress_rope_theta, 160_000.0)?;
    validate_compress_ratios(&root.compress_ratios)?;
    ensure!(
        root.dspark_block_size == 5,
        "DSV4F dspark_block_size must be 5"
    );
    ensure!(
        root.dspark_noise_token_id == 128_799,
        "DSV4F dspark_noise_token_id must be 128799"
    );
    ensure!(
        root.dspark_target_layer_ids == [40, 41, 42],
        "DSV4F dspark_target_layer_ids must be [40, 41, 42]"
    );
    ensure!(
        root.dspark_markov_rank == 256,
        "DSV4F dspark_markov_rank must be 256"
    );
    Ok(())
}

fn validate_inference_contract(config: &InferenceConfig) -> Result<()> {
    ensure!(
        config.n_mtp_layers == MTP_LAYERS,
        "DSV4F inference n_mtp_layers must be {MTP_LAYERS} for the explicit M1 DSpark skip contract"
    );
    ensure!(config.dtype == "fp8", "DSV4F inference dtype must be fp8");
    ensure!(
        config.scale_fmt == "ue8m0",
        "DSV4F inference scale_fmt must be ue8m0"
    );
    ensure!(
        config.expert_dtype == "fp4",
        "DSV4F inference expert_dtype must be fp4"
    );
    validate_compress_ratios(&config.compress_ratios)
}

fn cross_validate(root: &RootConfig, inference: &InferenceConfig) -> Result<()> {
    ensure_same("vocab_size", &root.vocab_size, &inference.vocab_size)?;
    ensure_same("hidden_size/dim", &root.hidden_size, &inference.dim)?;
    ensure_same(
        "moe_intermediate_size/moe_inter_dim",
        &root.moe_intermediate_size,
        &inference.moe_inter_dim,
    )?;
    ensure_same(
        "num_hidden_layers/n_layers",
        &root.num_hidden_layers,
        &inference.n_layers,
    )?;
    ensure_same(
        "num_hash_layers/n_hash_layers",
        &root.num_hash_layers,
        &inference.n_hash_layers,
    )?;
    ensure_same(
        "dspark_block_size",
        &root.dspark_block_size,
        &inference.dspark_block_size,
    )?;
    ensure_same(
        "dspark_noise_token_id",
        &root.dspark_noise_token_id,
        &inference.dspark_noise_token_id,
    )?;
    ensure_same(
        "dspark_target_layer_ids",
        &root.dspark_target_layer_ids,
        &inference.dspark_target_layer_ids,
    )?;
    ensure_same(
        "dspark_markov_rank",
        &root.dspark_markov_rank,
        &inference.dspark_markov_rank,
    )?;
    ensure_same(
        "num_attention_heads/n_heads",
        &root.num_attention_heads,
        &inference.n_heads,
    )?;
    ensure_same(
        "n_routed_experts",
        &root.n_routed_experts,
        &inference.n_routed_experts,
    )?;
    ensure_same(
        "n_shared_experts",
        &root.n_shared_experts,
        &inference.n_shared_experts,
    )?;
    ensure_same(
        "num_experts_per_tok/n_activated_experts",
        &root.num_experts_per_tok,
        &inference.n_activated_experts,
    )?;
    ensure_same(
        "scoring_func/score_func",
        &root.scoring_func,
        &inference.score_func,
    )?;
    ensure_float_same(
        "routed_scaling_factor/route_scale",
        root.routed_scaling_factor,
        inference.route_scale,
    )?;
    ensure_float_same("swiglu_limit", root.swiglu_limit, inference.swiglu_limit)?;
    ensure_same("q_lora_rank", &root.q_lora_rank, &inference.q_lora_rank)?;
    ensure_same("head_dim", &root.head_dim, &inference.head_dim)?;
    ensure_same(
        "qk_rope_head_dim/rope_head_dim",
        &root.qk_rope_head_dim,
        &inference.rope_head_dim,
    )?;
    ensure_same("o_groups", &root.o_groups, &inference.o_groups)?;
    ensure_same("o_lora_rank", &root.o_lora_rank, &inference.o_lora_rank)?;
    ensure_same(
        "sliding_window/window_size",
        &root.sliding_window,
        &inference.window_size,
    )?;
    ensure_same(
        "rope original sequence length",
        &root.rope_scaling.original_max_position_embeddings,
        &inference.original_seq_len,
    )?;
    ensure_float_same("rope_theta", root.rope_theta, inference.rope_theta)?;
    ensure_float_same(
        "rope factor",
        root.rope_scaling.factor,
        inference.rope_factor,
    )?;
    ensure_float_same(
        "beta_fast",
        root.rope_scaling.beta_fast,
        inference.beta_fast,
    )?;
    ensure_float_same(
        "beta_slow",
        root.rope_scaling.beta_slow,
        inference.beta_slow,
    )?;
    ensure_same(
        "index_n_heads",
        &root.index_n_heads,
        &inference.index_n_heads,
    )?;
    ensure_same(
        "index_head_dim",
        &root.index_head_dim,
        &inference.index_head_dim,
    )?;
    ensure_same("index_topk", &root.index_topk, &inference.index_topk)?;
    ensure_same("hc_mult", &root.hc_mult, &inference.hc_mult)?;
    ensure_same(
        "hc_sinkhorn_iters",
        &root.hc_sinkhorn_iters,
        &inference.hc_sinkhorn_iters,
    )?;
    ensure_same(
        "quantization scale_fmt",
        &root.quantization_config.scale_fmt,
        &inference.scale_fmt,
    )?;
    ensure_same("expert_dtype", &root.expert_dtype, &inference.expert_dtype)?;
    ensure_float_same(
        "compress_rope_theta",
        root.compress_rope_theta,
        inference.compress_rope_theta,
    )?;
    ensure_same(
        "compress_ratios",
        &root.compress_ratios,
        &inference.compress_ratios,
    )?;
    Ok(())
}

fn validate_quantization(config: &QuantizationConfig) -> Result<()> {
    ensure!(
        config.activation_scheme == "dynamic",
        "DSV4F FP8 activation_scheme must be dynamic"
    );
    ensure!(config.fmt == "e4m3", "DSV4F FP8 fmt must be e4m3");
    ensure!(
        config.quant_method == "fp8",
        "DSV4F quant_method must be fp8"
    );
    ensure!(config.scale_fmt == "ue8m0", "DSV4F scale_fmt must be ue8m0");
    ensure!(
        config.weight_block_size == [128, 128],
        "DSV4F weight_block_size must be [128, 128]"
    );
    Ok(())
}

fn validate_rope(config: &RopeScaling) -> Result<()> {
    ensure!(
        config.kind == "yarn",
        "DSV4F rope scaling type must be yarn"
    );
    ensure_float_value("rope_scaling.factor", config.factor, 16.0)?;
    ensure_float_value("rope_scaling.beta_fast", config.beta_fast, 32.0)?;
    ensure_float_value("rope_scaling.beta_slow", config.beta_slow, 1.0)?;
    ensure!(
        config.original_max_position_embeddings == 65_536,
        "DSV4F original_max_position_embeddings must be 65536"
    );
    Ok(())
}

fn validate_compress_ratios(actual: &[usize]) -> Result<()> {
    let mut expected = vec![0, 0];
    expected.extend((2..TARGET_LAYERS).map(|layer| if layer.is_multiple_of(2) { 4 } else { 128 }));
    expected.extend([0; MTP_LAYERS]);
    ensure!(
        actual == expected,
        "unsupported DSV4F compress_ratios: got {actual:?}, expected {expected:?}"
    );
    Ok(())
}

fn ensure_same<T: PartialEq + Debug>(field: &str, root: &T, inference: &T) -> Result<()> {
    ensure!(
        root == inference,
        "DSV4F config mismatch for {field}: root={root:?}, inference={inference:?}"
    );
    Ok(())
}

fn ensure_float_same(field: &str, root: f64, inference: f64) -> Result<()> {
    ensure!(
        root.to_bits() == inference.to_bits(),
        "DSV4F config mismatch for {field}: root={root:?}, inference={inference:?}"
    );
    Ok(())
}

fn ensure_float_value(field: &str, actual: f64, expected: f64) -> Result<()> {
    ensure!(
        actual.to_bits() == expected.to_bits(),
        "unsupported DSV4F {field}: got {actual:?}, expected {expected:?}"
    );
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> (Value, Value) {
        let root = serde_json::from_str(include_str!("../test_data/config.json")).unwrap();
        let inference =
            serde_json::from_str(include_str!("../test_data/inference-config.json")).unwrap();
        (root, inference)
    }

    #[test]
    fn official_pair_passes_with_only_the_explicit_mtp_difference() {
        let (root, inference) = fixtures();
        let config = Dsv4Config::validate_pair(&root, &inference).unwrap();
        assert_eq!(config.num_layers, TARGET_LAYERS);
        assert_eq!(config.num_mtp_layers, MTP_LAYERS);
        assert_eq!(config.compress_ratios.len(), TARGET_LAYERS + MTP_LAYERS);
    }

    #[test]
    fn target_path_mismatch_fails_closed() {
        let (root, mut inference) = fixtures();
        inference["window_size"] = Value::from(256);
        let error = Dsv4Config::validate_pair(&root, &inference)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sliding_window/window_size"), "{error}");
    }

    #[test]
    fn unsupported_quantization_variant_fails_closed() {
        let (mut root, inference) = fixtures();
        root["quantization_config"]["fmt"] = Value::from("e5m2");
        let error = Dsv4Config::validate_pair(&root, &inference)
            .unwrap_err()
            .to_string();
        assert!(error.contains("FP8 fmt"), "{error}");
    }

    #[test]
    fn root_probe_rejects_a_foreign_architecture() {
        let (mut root, _) = fixtures();
        root["model_type"] = Value::from("deepseek_v3");
        let error = probe_root_config(&root).unwrap_err().to_string();
        assert!(error.contains("model_type"), "{error}");
    }

    #[test]
    fn a_different_mtp_count_is_not_a_second_exception() {
        let (root, mut inference) = fixtures();
        inference["n_mtp_layers"] = Value::from(2);
        let error = Dsv4Config::validate_pair(&root, &inference)
            .unwrap_err()
            .to_string();
        assert!(error.contains("n_mtp_layers"), "{error}");
    }

    #[test]
    fn missing_inference_config_is_a_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(ROOT_CONFIG),
            include_str!("../test_data/config.json"),
        )
        .unwrap();
        let error = Dsv4Config::load(dir.path()).unwrap_err().to_string();
        assert!(
            error.contains("requires the bundled secondary config"),
            "{error}"
        );
    }
}
