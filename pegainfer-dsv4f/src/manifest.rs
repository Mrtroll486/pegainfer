//! G0 safetensors manifest generation and validation.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;
use serde::Serialize;

use crate::config::Dsv4Config;

const WEIGHT_INDEX: &str = "model.safetensors.index.json";
const FP8_BLOCK: usize = 128;
const FP4_PACK: usize = 2;
const FP4_GROUP: usize = 32;

pub const EXPECTED_TARGET_TENSORS: usize = 67_612;
pub const EXPECTED_DSPARK_TENSORS: usize = 4_705;
pub const EXPECTED_TARGET_BYTES: u64 = 156_015_698_140;
pub const EXPECTED_DSPARK_BYTES: u64 = 10_862_838_300;
pub const EXPECTED_TOTAL_BYTES: u64 = EXPECTED_TARGET_BYTES + EXPECTED_DSPARK_BYTES;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDisposition {
    Target,
    SkipDspark,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadAction {
    DirectUpload,
    DirectToExpertBank,
    RepackFp8Scale,
    RepackFp4Scale,
    SkipDspark,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DestinationLayout {
    OwnedTensor,
    Fp8ScaleLayout,
    ExpertW13Bank,
    ExpertW13ScaleBank,
    ExpertW2Bank,
    ExpertW2ScaleBank,
    Skipped,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorLedgerEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub scale_partner: Option<String>,
    pub shard: String,
    pub source_bytes: u64,
    pub disposition: TensorDisposition,
    pub load_action: LoadAction,
    pub destination_layout: DestinationLayout,
}

#[derive(Clone, Debug, Serialize)]
pub struct G0Report {
    pub model_path: String,
    pub config: Dsv4Config,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub target_tensor_count: usize,
    pub dspark_skip_tensor_count: usize,
    pub source_bytes: u64,
    pub target_source_bytes: u64,
    pub dspark_skip_source_bytes: u64,
    pub load_action_counts: BTreeMap<LoadAction, usize>,
    pub ledger: Vec<TensorLedgerEntry>,
}

pub struct Dsv4Manifest;

#[derive(Clone, Debug)]
struct TensorContract {
    dtype: Dtype,
    shape: Vec<usize>,
    scale_partner: Option<String>,
    disposition: TensorDisposition,
    load_action: LoadAction,
    destination_layout: DestinationLayout,
}

impl TensorContract {
    fn source_bytes(&self) -> Result<u64> {
        let elements = self.shape.iter().try_fold(1u64, |total, dim| {
            total
                .checked_mul(*dim as u64)
                .ok_or_else(|| anyhow::anyhow!("tensor shape {:?} overflows", self.shape))
        })?;
        elements
            .checked_mul(dtype_bytes(self.dtype)?)
            .ok_or_else(|| anyhow::anyhow!("tensor byte count {:?} overflows", self.shape))
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct IndexMetadata {
    total_size: u64,
}

#[derive(Debug)]
struct ActualTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    source_bytes: u64,
    shard: String,
}

impl Dsv4Manifest {
    pub fn inspect(model_dir: &Path, config: &Dsv4Config) -> Result<G0Report> {
        let expected = expected_contracts(config)?;
        let index_path = model_dir.join(WEIGHT_INDEX);
        let index_content = std::fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_content)
            .with_context(|| format!("parse {}", index_path.display()))?;

        validate_index_coverage(&index.weight_map, &expected)?;
        let actual = inspect_shards(model_dir, &index.weight_map, &expected)?;

        let mut ledger = Vec::with_capacity(expected.len());
        let mut target_source_bytes = 0u64;
        let mut dspark_skip_source_bytes = 0u64;
        let mut load_action_counts = BTreeMap::new();
        for (name, contract) in expected {
            let observed = actual
                .get(&name)
                .with_context(|| format!("safetensors headers missing tensor {name}"))?;
            validate_observed_contract(
                &name,
                observed.dtype,
                &observed.shape,
                observed.source_bytes,
                &contract,
            )?;
            match contract.disposition {
                TensorDisposition::Target => {
                    target_source_bytes = target_source_bytes
                        .checked_add(observed.source_bytes)
                        .context("DSV4F target source byte count overflow")?;
                }
                TensorDisposition::SkipDspark => {
                    dspark_skip_source_bytes = dspark_skip_source_bytes
                        .checked_add(observed.source_bytes)
                        .context("DSV4F DSpark source byte count overflow")?;
                }
            }
            *load_action_counts.entry(contract.load_action).or_default() += 1;
            ledger.push(TensorLedgerEntry {
                name,
                shape: observed.shape.clone(),
                dtype: format!("{:?}", observed.dtype),
                scale_partner: contract.scale_partner,
                shard: observed.shard.clone(),
                source_bytes: observed.source_bytes,
                disposition: contract.disposition,
                load_action: contract.load_action,
                destination_layout: contract.destination_layout,
            });
        }

        let source_bytes = target_source_bytes
            .checked_add(dspark_skip_source_bytes)
            .context("DSV4F total source byte count overflow")?;
        ensure!(
            target_source_bytes == EXPECTED_TARGET_BYTES,
            "DSV4F target payload mismatch: got {target_source_bytes}, expected {EXPECTED_TARGET_BYTES}"
        );
        ensure!(
            dspark_skip_source_bytes == EXPECTED_DSPARK_BYTES,
            "DSV4F DSpark payload mismatch: got {dspark_skip_source_bytes}, expected {EXPECTED_DSPARK_BYTES}"
        );
        ensure!(
            source_bytes == index.metadata.total_size,
            "DSV4F index metadata total_size mismatch: headers={source_bytes}, index={} ",
            index.metadata.total_size
        );

        let shards = index.weight_map.values().collect::<BTreeSet<_>>();
        Ok(G0Report {
            model_path: model_dir.display().to_string(),
            config: config.clone(),
            shard_count: shards.len(),
            tensor_count: ledger.len(),
            target_tensor_count: ledger
                .iter()
                .filter(|entry| entry.disposition == TensorDisposition::Target)
                .count(),
            dspark_skip_tensor_count: ledger
                .iter()
                .filter(|entry| entry.disposition == TensorDisposition::SkipDspark)
                .count(),
            source_bytes,
            target_source_bytes,
            dspark_skip_source_bytes,
            load_action_counts,
            ledger,
        })
    }
}

fn inspect_shards(
    model_dir: &Path,
    weight_map: &BTreeMap<String, String>,
    expected: &BTreeMap<String, TensorContract>,
) -> Result<BTreeMap<String, ActualTensor>> {
    let shard_names = weight_map.values().cloned().collect::<BTreeSet<_>>();
    let mut actual = BTreeMap::new();
    for shard in shard_names {
        let relative = Path::new(&shard);
        ensure!(
            relative.file_name() == Some(relative.as_os_str()),
            "DSV4F shard path must be a plain file name, got {shard:?}"
        );
        let path = model_dir.join(relative);
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the mapping is read-only and remains alive for the complete
        // SafeTensors view lifetime inside this loop iteration.
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parse safetensors header {}", path.display()))?;
        for name in tensors.names() {
            let indexed_shard = weight_map.get(name).with_context(|| {
                format!("unknown tensor {name} is present in shard {shard} but absent from index")
            })?;
            ensure!(
                indexed_shard == &shard,
                "DSV4F index maps tensor {name} to {indexed_shard}, but it is present in {shard}"
            );
            let contract = expected
                .get(name)
                .with_context(|| format!("no DSV4F contract for tensor {name}"))?;
            let view = tensors
                .tensor(name)
                .with_context(|| format!("read tensor metadata {name} from {shard}"))?;
            let source_bytes = u64::try_from(view.data().len())
                .with_context(|| format!("tensor {name} source byte count does not fit u64"))?;
            validate_observed_contract(name, view.dtype(), view.shape(), source_bytes, contract)?;
            ensure!(
                actual
                    .insert(
                        name.to_owned(),
                        ActualTensor {
                            dtype: view.dtype(),
                            shape: view.shape().to_vec(),
                            source_bytes,
                            shard: shard.clone(),
                        },
                    )
                    .is_none(),
                "DSV4F tensor {name} appears in more than one shard"
            );
        }
    }

    let indexed = weight_map.keys().cloned().collect::<BTreeSet<_>>();
    let observed = actual.keys().cloned().collect::<BTreeSet<_>>();
    let missing = indexed.difference(&observed).take(5).collect::<Vec<_>>();
    let extra = observed.difference(&indexed).take(5).collect::<Vec<_>>();
    ensure!(
        missing.is_empty() && extra.is_empty(),
        "DSV4F index/header coverage mismatch: missing_header_sample={missing:?}, extra_header_sample={extra:?}, index={}, headers={}",
        indexed.len(),
        observed.len()
    );
    Ok(actual)
}

fn validate_observed_contract(
    name: &str,
    dtype: Dtype,
    shape: &[usize],
    source_bytes: u64,
    contract: &TensorContract,
) -> Result<()> {
    ensure!(
        dtype == contract.dtype,
        "DSV4F tensor {name} dtype mismatch: got {dtype:?}, expected {:?}",
        contract.dtype
    );
    ensure!(
        shape == contract.shape,
        "DSV4F tensor {name} shape mismatch: got {shape:?}, expected {:?}",
        contract.shape
    );
    let expected_bytes = contract.source_bytes()?;
    ensure!(
        source_bytes == expected_bytes,
        "DSV4F tensor {name} source byte mismatch: got {source_bytes}, expected {expected_bytes}"
    );
    Ok(())
}

fn validate_index_coverage(
    weight_map: &BTreeMap<String, String>,
    expected: &BTreeMap<String, TensorContract>,
) -> Result<()> {
    let checkpoint = weight_map.keys().cloned().collect::<BTreeSet<_>>();
    let generated = expected.keys().cloned().collect::<BTreeSet<_>>();
    let unknown = checkpoint
        .difference(&generated)
        .take(5)
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        unknown.is_empty(),
        "DSV4F checkpoint has unknown tensors outside the generated target and DSpark contracts: sample={unknown:?}"
    );

    let mut missing_target = Vec::new();
    let mut missing_dspark = Vec::new();
    for name in generated.difference(&checkpoint).take(10) {
        match expected
            .get(name)
            .expect("generated name came from the expected contract")
            .disposition
        {
            TensorDisposition::Target => missing_target.push((*name).clone()),
            TensorDisposition::SkipDspark => missing_dspark.push((*name).clone()),
        }
    }
    ensure!(
        missing_target.is_empty(),
        "DSV4F checkpoint is missing required target tensors: sample={missing_target:?}"
    );
    ensure!(
        missing_dspark.is_empty(),
        "DSV4F checkpoint is missing tensors from the explicit DSpark skip contract: sample={missing_dspark:?}"
    );
    ensure!(
        checkpoint.len() == generated.len(),
        "DSV4F checkpoint tensor count mismatch: checkpoint={}, generated={}",
        checkpoint.len(),
        generated.len()
    );
    Ok(())
}

fn expected_contracts(config: &Dsv4Config) -> Result<BTreeMap<String, TensorContract>> {
    let mut builder = ContractBuilder::default();
    builder.plain(
        "embed.weight",
        Dtype::BF16,
        [config.vocab_size, config.hidden_size],
        false,
    )?;
    builder.plain(
        "head.weight",
        Dtype::BF16,
        [config.vocab_size, config.hidden_size],
        false,
    )?;
    builder.plain("norm.weight", Dtype::BF16, [config.hidden_size], false)?;
    push_hc_head(&mut builder, "", config, false)?;

    for layer in 0..config.num_layers {
        let prefix = format!("layers.{layer}");
        push_decoder_block(
            &mut builder,
            &prefix,
            config.compress_ratios[layer],
            layer < config.num_hash_layers,
            config,
            false,
        )?;
    }

    for stage in 0..config.num_mtp_layers {
        let prefix = format!("mtp.{stage}");
        push_decoder_block(&mut builder, &prefix, 0, false, config, true)?;
        if stage == 0 {
            builder.plain(
                format!("{prefix}.main_norm.weight"),
                Dtype::BF16,
                [config.hidden_size],
                true,
            )?;
            builder.fp8(
                &format!("{prefix}.main_proj"),
                config.hidden_size,
                3 * config.hidden_size,
                true,
            )?;
        }
        if stage + 1 == config.num_mtp_layers {
            builder.plain(
                format!("{prefix}.confidence_head.proj.weight"),
                Dtype::BF16,
                [1, config.hidden_size + 256],
                true,
            )?;
            push_hc_head(&mut builder, &format!("{prefix}."), config, true)?;
            builder.plain(
                format!("{prefix}.markov_head.markov_w1.weight"),
                Dtype::BF16,
                [config.vocab_size, 256],
                true,
            )?;
            builder.plain(
                format!("{prefix}.markov_head.markov_w2.weight"),
                Dtype::BF16,
                [config.vocab_size, 256],
                true,
            )?;
            builder.plain(
                format!("{prefix}.norm.weight"),
                Dtype::BF16,
                [config.hidden_size],
                true,
            )?;
        }
    }

    let target_count = builder
        .contracts
        .values()
        .filter(|contract| contract.disposition == TensorDisposition::Target)
        .count();
    let dspark_count = builder.contracts.len() - target_count;
    ensure!(
        target_count == EXPECTED_TARGET_TENSORS,
        "generated DSV4F target tensor count {target_count} != {EXPECTED_TARGET_TENSORS}"
    );
    ensure!(
        dspark_count == EXPECTED_DSPARK_TENSORS,
        "generated DSV4F DSpark tensor count {dspark_count} != {EXPECTED_DSPARK_TENSORS}"
    );
    Ok(builder.contracts)
}

fn push_decoder_block(
    builder: &mut ContractBuilder,
    prefix: &str,
    compress_ratio: usize,
    hash_routed: bool,
    config: &Dsv4Config,
    skip: bool,
) -> Result<()> {
    builder.plain(
        format!("{prefix}.attn.attn_sink"),
        Dtype::F32,
        [config.num_attention_heads],
        skip,
    )?;
    builder.plain(
        format!("{prefix}.attn.kv_norm.weight"),
        Dtype::BF16,
        [config.head_dim],
        skip,
    )?;
    builder.plain(
        format!("{prefix}.attn.q_norm.weight"),
        Dtype::BF16,
        [config.q_lora_rank],
        skip,
    )?;
    builder.fp8(
        &format!("{prefix}.attn.wkv"),
        config.head_dim,
        config.hidden_size,
        skip,
    )?;
    builder.fp8(
        &format!("{prefix}.attn.wq_a"),
        config.q_lora_rank,
        config.hidden_size,
        skip,
    )?;
    builder.fp8(
        &format!("{prefix}.attn.wq_b"),
        config.num_attention_heads * config.head_dim,
        config.q_lora_rank,
        skip,
    )?;
    let grouped_output = config.index_n_heads * config.index_head_dim;
    builder.fp8(
        &format!("{prefix}.attn.wo_a"),
        grouped_output,
        config.hidden_size,
        skip,
    )?;
    builder.fp8(
        &format!("{prefix}.attn.wo_b"),
        config.hidden_size,
        grouped_output,
        skip,
    )?;

    if compress_ratio != 0 {
        let overlap_factor = if compress_ratio == 4 { 2 } else { 1 };
        builder.plain(
            format!("{prefix}.attn.compressor.ape"),
            Dtype::F32,
            [compress_ratio, overlap_factor * config.head_dim],
            skip,
        )?;
        builder.plain(
            format!("{prefix}.attn.compressor.norm.weight"),
            Dtype::BF16,
            [config.head_dim],
            skip,
        )?;
        for projection in ["wgate", "wkv"] {
            builder.plain(
                format!("{prefix}.attn.compressor.{projection}.weight"),
                Dtype::BF16,
                [overlap_factor * config.head_dim, config.hidden_size],
                skip,
            )?;
        }
        if compress_ratio == 4 {
            builder.plain(
                format!("{prefix}.attn.indexer.compressor.ape"),
                Dtype::F32,
                [compress_ratio, 2 * config.index_head_dim],
                skip,
            )?;
            builder.plain(
                format!("{prefix}.attn.indexer.compressor.norm.weight"),
                Dtype::BF16,
                [config.index_head_dim],
                skip,
            )?;
            for projection in ["wgate", "wkv"] {
                builder.plain(
                    format!("{prefix}.attn.indexer.compressor.{projection}.weight"),
                    Dtype::BF16,
                    [2 * config.index_head_dim, config.hidden_size],
                    skip,
                )?;
            }
            builder.plain(
                format!("{prefix}.attn.indexer.weights_proj.weight"),
                Dtype::BF16,
                [config.index_n_heads, config.hidden_size],
                skip,
            )?;
            builder.fp8(
                &format!("{prefix}.attn.indexer.wq_b"),
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
                skip,
            )?;
        }
    }

    builder.plain(
        format!("{prefix}.attn_norm.weight"),
        Dtype::BF16,
        [config.hidden_size],
        skip,
    )?;
    for expert in 0..config.num_routed_experts {
        let expert_prefix = format!("{prefix}.ffn.experts.{expert}");
        builder.fp4_expert(
            &format!("{expert_prefix}.w1"),
            config.expert_intermediate_size,
            config.hidden_size,
            DestinationLayout::ExpertW13Bank,
            DestinationLayout::ExpertW13ScaleBank,
            skip,
        )?;
        builder.fp4_expert(
            &format!("{expert_prefix}.w2"),
            config.hidden_size,
            config.expert_intermediate_size,
            DestinationLayout::ExpertW2Bank,
            DestinationLayout::ExpertW2ScaleBank,
            skip,
        )?;
        builder.fp4_expert(
            &format!("{expert_prefix}.w3"),
            config.expert_intermediate_size,
            config.hidden_size,
            DestinationLayout::ExpertW13Bank,
            DestinationLayout::ExpertW13ScaleBank,
            skip,
        )?;
    }
    builder.plain(
        format!("{prefix}.ffn.gate.weight"),
        Dtype::BF16,
        [config.num_routed_experts, config.hidden_size],
        skip,
    )?;
    if hash_routed {
        builder.plain(
            format!("{prefix}.ffn.gate.tid2eid"),
            Dtype::I64,
            [config.vocab_size, config.num_activated_experts],
            skip,
        )?;
    } else {
        builder.plain(
            format!("{prefix}.ffn.gate.bias"),
            Dtype::F32,
            [config.num_routed_experts],
            skip,
        )?;
    }
    for projection in ["w1", "w2", "w3"] {
        let (rows, cols) = if projection == "w2" {
            (config.hidden_size, config.expert_intermediate_size)
        } else {
            (config.expert_intermediate_size, config.hidden_size)
        };
        builder.fp8(
            &format!("{prefix}.ffn.shared_experts.{projection}"),
            rows,
            cols,
            skip,
        )?;
    }
    builder.plain(
        format!("{prefix}.ffn_norm.weight"),
        Dtype::BF16,
        [config.hidden_size],
        skip,
    )?;
    push_hc_block(builder, prefix, config, skip)
}

fn push_hc_block(
    builder: &mut ContractBuilder,
    prefix: &str,
    config: &Dsv4Config,
    skip: bool,
) -> Result<()> {
    let routes = 2 * config.hc_mult * (config.hc_mult - 1);
    for family in ["attn", "ffn"] {
        builder.plain(
            format!("{prefix}.hc_{family}_base"),
            Dtype::F32,
            [routes],
            skip,
        )?;
        builder.plain(
            format!("{prefix}.hc_{family}_fn"),
            Dtype::F32,
            [routes, config.hc_mult * config.hidden_size],
            skip,
        )?;
        builder.plain(
            format!("{prefix}.hc_{family}_scale"),
            Dtype::F32,
            [config.hc_mult - 1],
            skip,
        )?;
    }
    Ok(())
}

fn push_hc_head(
    builder: &mut ContractBuilder,
    prefix: &str,
    config: &Dsv4Config,
    skip: bool,
) -> Result<()> {
    builder.plain(
        format!("{prefix}hc_head_base"),
        Dtype::F32,
        [config.hc_mult],
        skip,
    )?;
    builder.plain(
        format!("{prefix}hc_head_fn"),
        Dtype::F32,
        [config.hc_mult, config.hc_mult * config.hidden_size],
        skip,
    )?;
    builder.plain(format!("{prefix}hc_head_scale"), Dtype::F32, [1], skip)
}

#[derive(Default)]
struct ContractBuilder {
    contracts: BTreeMap<String, TensorContract>,
}

impl ContractBuilder {
    fn plain<const N: usize>(
        &mut self,
        name: impl Into<String>,
        dtype: Dtype,
        shape: [usize; N],
        skip: bool,
    ) -> Result<()> {
        self.insert(
            name.into(),
            TensorContract {
                dtype,
                shape: shape.to_vec(),
                scale_partner: None,
                disposition: disposition(skip),
                load_action: action(skip, LoadAction::DirectUpload),
                destination_layout: destination(skip, DestinationLayout::OwnedTensor),
            },
        )
    }

    fn fp8(&mut self, stem: &str, rows: usize, cols: usize, skip: bool) -> Result<()> {
        let weight = format!("{stem}.weight");
        let scale = format!("{stem}.scale");
        self.insert(
            weight.clone(),
            TensorContract {
                dtype: Dtype::F8_E4M3,
                shape: vec![rows, cols],
                scale_partner: Some(scale.clone()),
                disposition: disposition(skip),
                load_action: action(skip, LoadAction::DirectUpload),
                destination_layout: destination(skip, DestinationLayout::OwnedTensor),
            },
        )?;
        self.insert(
            scale,
            TensorContract {
                dtype: Dtype::F8_E8M0,
                shape: vec![rows.div_ceil(FP8_BLOCK), cols.div_ceil(FP8_BLOCK)],
                scale_partner: Some(weight),
                disposition: disposition(skip),
                load_action: action(skip, LoadAction::RepackFp8Scale),
                destination_layout: destination(skip, DestinationLayout::Fp8ScaleLayout),
            },
        )
    }

    fn fp4_expert(
        &mut self,
        stem: &str,
        rows: usize,
        logical_cols: usize,
        weight_destination: DestinationLayout,
        scale_destination: DestinationLayout,
        skip: bool,
    ) -> Result<()> {
        ensure!(
            logical_cols.is_multiple_of(FP4_GROUP),
            "FP4 tensor {stem} logical K={logical_cols} is not group-aligned"
        );
        let weight = format!("{stem}.weight");
        let scale = format!("{stem}.scale");
        self.insert(
            weight.clone(),
            TensorContract {
                dtype: Dtype::I8,
                shape: vec![rows, logical_cols / FP4_PACK],
                scale_partner: Some(scale.clone()),
                disposition: disposition(skip),
                load_action: action(skip, LoadAction::DirectToExpertBank),
                destination_layout: destination(skip, weight_destination),
            },
        )?;
        self.insert(
            scale,
            TensorContract {
                dtype: Dtype::F8_E8M0,
                shape: vec![rows, logical_cols / FP4_GROUP],
                scale_partner: Some(weight),
                disposition: disposition(skip),
                load_action: action(skip, LoadAction::RepackFp4Scale),
                destination_layout: destination(skip, scale_destination),
            },
        )
    }

    fn insert(&mut self, name: String, contract: TensorContract) -> Result<()> {
        match self.contracts.entry(name) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(contract);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                anyhow::bail!("duplicate generated DSV4F tensor contract {}", entry.key())
            }
        }
    }
}

const fn disposition(skip: bool) -> TensorDisposition {
    if skip {
        TensorDisposition::SkipDspark
    } else {
        TensorDisposition::Target
    }
}

const fn action(skip: bool, target: LoadAction) -> LoadAction {
    if skip { LoadAction::SkipDspark } else { target }
}

const fn destination(skip: bool, target: DestinationLayout) -> DestinationLayout {
    if skip {
        DestinationLayout::Skipped
    } else {
        target
    }
}

fn dtype_bytes(dtype: Dtype) -> Result<u64> {
    match dtype {
        Dtype::I8 | Dtype::F8_E4M3 | Dtype::F8_E8M0 => Ok(1),
        Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        Dtype::I64 => Ok(8),
        other => anyhow::bail!("unsupported DSV4F safetensors dtype {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Dsv4Config {
        let root = serde_json::from_str(include_str!("../test_data/config.json")).unwrap();
        let inference =
            serde_json::from_str(include_str!("../test_data/inference-config.json")).unwrap();
        Dsv4Config::validate_pair(&root, &inference).unwrap()
    }

    #[test]
    fn generated_contract_has_exact_target_skip_counts_and_bytes() {
        let contracts = expected_contracts(&config()).unwrap();
        let mut target_count = 0;
        let mut skip_count = 0;
        let mut target_bytes = 0;
        let mut skip_bytes = 0;
        for contract in contracts.values() {
            match contract.disposition {
                TensorDisposition::Target => {
                    target_count += 1;
                    target_bytes += contract.source_bytes().unwrap();
                }
                TensorDisposition::SkipDspark => {
                    skip_count += 1;
                    skip_bytes += contract.source_bytes().unwrap();
                }
            }
        }
        assert_eq!(target_count, EXPECTED_TARGET_TENSORS);
        assert_eq!(skip_count, EXPECTED_DSPARK_TENSORS);
        assert_eq!(target_bytes, EXPECTED_TARGET_BYTES);
        assert_eq!(skip_bytes, EXPECTED_DSPARK_BYTES);

        let action_counts = contracts.values().fold(
            BTreeMap::<LoadAction, usize>::new(),
            |mut counts, contract| {
                *counts.entry(contract.load_action).or_default() += 1;
                counts
            },
        );
        assert_eq!(action_counts[&LoadAction::DirectUpload], 1199);
        assert_eq!(action_counts[&LoadAction::DirectToExpertBank], 33_024);
        assert_eq!(action_counts[&LoadAction::RepackFp8Scale], 365);
        assert_eq!(action_counts[&LoadAction::RepackFp4Scale], 33_024);
        assert_eq!(action_counts[&LoadAction::SkipDspark], 4705);
    }

    #[test]
    fn every_scale_partner_is_present_and_symmetric() {
        let contracts = expected_contracts(&config()).unwrap();
        for (name, contract) in &contracts {
            let Some(partner) = &contract.scale_partner else {
                continue;
            };
            assert_eq!(
                contracts[partner].scale_partner.as_deref(),
                Some(name.as_str()),
                "{name} -> {partner}"
            );
        }
    }

    #[test]
    fn dtype_shape_and_byte_mismatches_fail_closed() {
        let contracts = expected_contracts(&config()).unwrap();
        let name = "layers.0.ffn.experts.0.w1.weight";
        let contract = &contracts[name];
        let bytes = contract.source_bytes().unwrap();

        let dtype_error =
            validate_observed_contract(name, Dtype::F8_E4M3, &contract.shape, bytes, contract)
                .unwrap_err()
                .to_string();
        assert!(dtype_error.contains("dtype mismatch"), "{dtype_error}");

        let shape_error =
            validate_observed_contract(name, contract.dtype, &[2048, 1024], bytes, contract)
                .unwrap_err()
                .to_string();
        assert!(shape_error.contains("shape mismatch"), "{shape_error}");

        let byte_error =
            validate_observed_contract(name, contract.dtype, &contract.shape, bytes - 1, contract)
                .unwrap_err()
                .to_string();
        assert!(byte_error.contains("source byte mismatch"), "{byte_error}");
    }

    #[test]
    fn routed_fp4_contracts_land_in_final_expert_banks() {
        let contracts = expected_contracts(&config()).unwrap();
        let w1 = &contracts["layers.0.ffn.experts.255.w1.weight"];
        assert_eq!(w1.dtype, Dtype::I8);
        assert_eq!(w1.shape, [2048, 2048]);
        assert_eq!(w1.load_action, LoadAction::DirectToExpertBank);
        assert_eq!(w1.destination_layout, DestinationLayout::ExpertW13Bank);
        assert_eq!(
            w1.scale_partner.as_deref(),
            Some("layers.0.ffn.experts.255.w1.scale")
        );

        let w2_scale = &contracts["layers.42.ffn.experts.0.w2.scale"];
        assert_eq!(w2_scale.dtype, Dtype::F8_E8M0);
        assert_eq!(w2_scale.shape, [4096, 64]);
        assert_eq!(w2_scale.load_action, LoadAction::RepackFp4Scale);
        assert_eq!(
            w2_scale.destination_layout,
            DestinationLayout::ExpertW2ScaleBank
        );
    }

    #[test]
    fn mtp_namespace_is_complete_and_always_skipped() {
        let contracts = expected_contracts(&config()).unwrap();
        let stage_counts = (0..3)
            .map(|stage| {
                let prefix = format!("mtp.{stage}.");
                contracts
                    .keys()
                    .filter(|name| name.starts_with(&prefix))
                    .count()
            })
            .collect::<Vec<_>>();
        assert_eq!(stage_counts, [1568, 1565, 1572]);
        assert!(contracts.iter().all(|(name, contract)| {
            !name.starts_with("mtp.")
                || (contract.disposition == TensorDisposition::SkipDspark
                    && contract.load_action == LoadAction::SkipDspark
                    && contract.destination_layout == DestinationLayout::Skipped)
        }));
    }

    #[test]
    fn unknown_or_missing_names_fail_closed() {
        let contracts = expected_contracts(&config()).unwrap();
        let mut index = contracts
            .keys()
            .map(|name| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
            .collect::<BTreeMap<_, _>>();
        index.insert(
            "mtp.3.unreviewed.weight".to_owned(),
            "model-00001-of-00001.safetensors".to_owned(),
        );
        let error = validate_index_coverage(&index, &contracts)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown tensors"), "{error}");

        index.remove("mtp.3.unreviewed.weight");
        index.remove("layers.0.attn.wkv.weight");
        let error = validate_index_coverage(&index, &contracts)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing required target"), "{error}");
    }
}
