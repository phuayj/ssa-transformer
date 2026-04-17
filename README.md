# Can Transformers Learn to Verify During Backtracking Search?

[![Paper: MLJ](https://img.shields.io/badge/Paper-MLJ-blue)](#citation)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Public code release for the *Machine Learning* (MLJ) paper **"Can Transformers Learn to Verify During Backtracking Search?"** by **Yin Jun Phua, Tony Ribeiro, Tuan Nguyen, and Katsumi Inoue**.

## TL;DR

Backtracking search makes two decisions at each step: a **policy** chooses the next branch, and a **verification** step decides whether to continue or backtrack. When propagation already exposes conflicts, verification reduces to a binary continue-or-backtrack rule on the current state: the **reactive verification** target.

Causal transformers trained on cumulative traces fail this apparently simple rule through two architectural mechanisms: **scattered retrieval** (an accessibility failure) and **history entanglement** (an invariance failure). This paper focuses on the second mechanism.

**Selective State Attention (SSA)** is a fixed attention mask that blocks cross-block trajectory tokens while preserving the current decision block. It enforces structural prior-block-content invariance with **no parameter or objective change**.

On the primary **n=50 3-SAT** setting, SSA solves **93.4%** of instances under **state-rebuilt inference**, while the cumulative-trained causal baseline drops to **4.8%**.

## Abstract

> Backtracking search combines a **policy** decision about which branch to take with a **verification** decision about whether the current path should continue or backtrack. With constraint propagation already exposing conflicts, the verification target becomes a binary reactive continue-or-backtrack rule on the current state. Causal transformers trained on a **cumulative trace** fail this rule through two mechanisms: **scattered retrieval**, where state features are hard to recompose from many positions, and **history entanglement**, where predictions depend on the trajectory that produced the state rather than on the state itself. **Selective State Attention (SSA)** fixes the second mechanism by imposing **state isolation** with a fixed mask that blocks cross-block trajectory attention, adding no parameters and leaving the objective unchanged. Across **four domains**—**3-SAT, graph coloring, Blocks World, and backtracking parsing**—SSA trained on cumulative traces transfers to **state-rebuilt** deployment; on n=50 3-SAT it reaches **93.4%** solve rate while the cumulative-trained causal baseline reaches **4.8%**. In same-state/different-history tests, SSA has **100.0%** argmax agreement while causal has **71.4%**, a **28.6 percentage-point** drop. The same invariance principle motivates inference-time context-clearing protocols for pretrained LLMs as an **outlook hypothesis**, not as a demonstrated capability.

## What is in this repository?

This release contains:

- the core **SSA decoder** and supporting slot-memory infrastructure,
- task environments and tokenizers for multiple backtracking domains,
- scripts for **trace generation**, **training**, **evaluation**, **tables**, and **figures**,
- MATH scripts for exploring the LLM **context-clearing outlook hypothesis** motivated by the symbolic evidence,
- the **r^k** mechanistic benchmark used to probe state isolation.

### Main model configuration

- **SSA decoder**: 6 layers, 256 hidden size, 8 attention heads, 32 slot registers
- **Model size**: ~6M parameters
- **Baselines**:
  - full causal transformer,
  - LSTM,
  - MLP state-feature models,
  - history-reduction variants and Panel B baselines (current-block-only mask, block dropout, sliding window, null history, history-transplant, contrastive invariance, Factor GNN),
  - rule-based heuristics such as **VSIDS**, **occurrence**, and **random**.

## Installation

### Requirements

- Python **3.10+**
- PyTorch **2.x**
- `numpy`, `matplotlib`, `pydantic`, `pyyaml`
- Optional for some experiments:
  - `pysat` for SAT solvability oracles,
  - Hugging Face tooling (`transformers`, `accelerate`) for LLM transfer experiments.

### GPU memory

Training with the default model (6 layers, 256 hidden, 8 heads, 32 slots) and `--max_seq_len 4096`:

| Attention mode | Batch size | Approx. VRAM |
|---|---|---|
| SSA (`selective_ssa`) | 8 | ~75 GB |
| SSA (`selective_ssa`) | 4 | ~40 GB |
| Full causal (`full_causal`) | 4 | ~37 GB |
| Full causal (`full_causal`) | 2 | ~20 GB |

For GPUs with less memory, reduce `--batch_size` accordingly. Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` may help with fragmentation.

### Install from source

```bash
cd /workspace/public-release
pip install -e .
```

For SAT experiments with the optional oracle dependency:

```bash
pip install -e ".[sat]"
```

For the LLM transfer experiments (`scripts/llm/eval_pasv.py`, the FP/FN and precision-cliff analyses, etc.) which use scikit-learn:

```bash
pip install -e ".[llm]"
```

Both extras can be combined:

```bash
pip install -e ".[sat,llm]"
```

If you want the import path behavior used in the scripts, run commands from the repository root with:

```bash
PYTHONPATH=src
```

For example:

```bash
PYTHONPATH=src python scripts/training/train_gc_ssa.py --help
```

## Project structure

```text
├── README.md
├── LICENSE
├── pyproject.toml
├── src/
│   ├── universal/          # Core models and infrastructure
│   │   ├── ssa_decoder.py  # ★ SSA (Selective State Attention) decoder
│   │   ├── slot_decoder.py # Slot memory architecture
│   │   ├── cdcl_decoder.py # Base CDCL-style decoder
│   │   ├── lstm_decoder.py # LSTM baseline
│   │   ├── decoder_only.py # Causal transformer
│   │   ├── slot_wrapper.py # LLM slot injection for HuggingFace models
│   │   ├── delta_local_verifier.py  # Delta-local verification head
│   │   ├── cdcl_tokenizer.py        # Tokenizer for search traces
│   │   ├── constraint_transformer.py # Constraint-aware transformer
│   │   ├── model.py                  # FactorGNN / MLP models
│   │   ├── base_env.py, base_oracle.py, types.py, wrapper.py, dataset.py
│   │   ├── backtrack_agent.py, autoregressive_agent.py
│   │   └── __init__.py
│   ├── graph_coloring/     # Graph k-coloring domain
│   ├── sat/                # 3-SAT domain (DPLL + unit propagation)
│   ├── blocks_world/       # Blocks World planning domain
│   ├── parsing/            # Backtracking PEG parser domain
│   ├── baselines/          # Classical solvers (DSATUR)
│   └── rk_benchmark/       # r^k mechanistic benchmark
├── scripts/
│   ├── data_generation/    # Trace generation scripts
│   ├── training/           # Model training scripts
│   ├── evaluation/         # Evaluation scripts
│   ├── tables/             # LaTeX table generation
│   ├── figures/            # Figure generation
│   └── llm/                # LLM transfer study scripts
└── tests/
```

## Quick start

The standard workflow is:

1. **Generate search traces** for a domain.
2. **Train** an SSA model and a baseline.
3. **Evaluate** them in closed-loop or autonomous search.

Below is a minimal graph-coloring pipeline using the scripts in this release.

### 1) Generate graph-coloring traces

```bash
PYTHONPATH=src python scripts/data_generation/generate_gc_enriched_traces.py \
  --num-graphs 5000 \
  --num-traces-per-graph 1 \
  --num-nodes 30 \
  --num-colors 4 \
  --edge-prob 0.35 \
  --output data/gc-traces/traces.pkl
```

### 2) Train an SSA model

```bash
PYTHONPATH=src python scripts/training/train_gc_ssa.py \
  --data_path data/gc-traces/traces.pkl \
  --output_dir experiments/gc-ssa \
  --mode ssa \
  --epochs 30 \
  --seed 42
```

### 3) Train a causal baseline

```bash
PYTHONPATH=src python scripts/training/train_gc_ssa.py \
  --data_path data/gc-traces/traces.pkl \
  --output_dir experiments/gc-causal \
  --mode causal \
  --epochs 30 \
  --seed 42
```

### 4) Evaluate closed-loop performance

```bash
PYTHONPATH=src python scripts/evaluation/eval_gc_ssa.py \
  --ssa_checkpoint experiments/gc-ssa/best.pt \
  --causal_checkpoint experiments/gc-causal/best.pt \
  --num_instances 200 \
  --num_nodes 30 \
  --num_colors 4 \
  --edge_prob 0.35 \
  --output_dir experiments/gc-eval
```

## Key experiments

The paper studies **reactive verification** across several search domains and transfer settings.

### 1) Graph Coloring

- Setting: **30-node** and **50-node** graph coloring with **4 colors**
- Goal: compare SSA against causal and classical baselines under backtracking search

Generate traces:

```bash
PYTHONPATH=src python scripts/data_generation/generate_gc_enriched_traces.py \
  --num-graphs 5000 \
  --num-traces-per-graph 1 \
  --num-nodes 30 \
  --num-colors 4 \
  --edge-prob 0.35 \
  --output data/gc-traces/traces.pkl
```

Train:

```bash
PYTHONPATH=src python scripts/training/train_gc_ssa.py \
  --data_path data/gc-traces/traces.pkl \
  --output_dir experiments/gc-ssa \
  --mode ssa
```

Evaluate SSA vs causal:

```bash
PYTHONPATH=src python scripts/evaluation/eval_gc_ssa.py \
  --ssa_checkpoint experiments/gc-ssa/best.pt \
  --causal_checkpoint experiments/gc-causal/best.pt \
  --num_instances 200 \
  --output_dir experiments/gc-eval
```

Related analyses:

- `scripts/evaluation/eval_gc_mask_ablation.py`
- `scripts/evaluation/eval_gc_history_transplant.py`
- `scripts/evaluation/eval_gc_explicit_state.py`
- `scripts/evaluation/eval_gc_e4_verification.py`

### 2) 3-SAT

- Settings:
  - **50 variables**, planted instances with **α = 4.0**
  - **75 variables**, phase-transition study
- Goal: test whether SSA supports robust state-based solvability prediction during DPLL/CDCL-style backtracking

Generate SAT traces:

```bash
PYTHONPATH=src python scripts/data_generation/generate_sat_backtracking_traces.py \
  --num-sat 5000 \
  --num-vars 50 \
  --alpha-sat 4.0 \
  --max-seq-len 4096 \
  --output-path data/sat-traces/traces.pkl
```

Train SSA on SAT traces:

```bash
PYTHONPATH=src python scripts/training/train_history_ablation.py \
  --data_path data/sat-traces/traces.pkl \
  --output_dir experiments/sat-ssa \
  --mask_mode selective_ssa \
  --model_type transformer \
  --max_seq_len 4096 \
  --batch_size 8
```

Train causal baseline:

```bash
PYTHONPATH=src python scripts/training/train_history_ablation.py \
  --data_path data/sat-traces/traces.pkl \
  --output_dir experiments/sat-causal \
  --mask_mode full_causal \
  --model_type transformer \
  --max_seq_len 4096 \
  --batch_size 4
```

Evaluate autonomous SAT solving:

```bash
PYTHONPATH=src python scripts/evaluation/eval_sat_autonomous.py \
  --checkpoints experiments/sat-ssa/best.pt,experiments/sat-causal/best.pt \
  --labels ssa,causal \
  --num-instances 200 \
  --num-vars 50 \
  --alpha 4.0 \
  --budget 4096 \
  --output-dir experiments/sat-autonomous
```

Evaluate heuristic baselines:

```bash
PYTHONPATH=src python scripts/evaluation/eval_sat_heuristic.py \
  --heuristic vsids_domain \
  --num-instances 200 \
  --budget 4096
```

Additional SAT analyses:

- `scripts/evaluation/eval_sat_mask_ablation.py`
- `scripts/evaluation/eval_sat_oracle_decomposition.py`
- `scripts/evaluation/eval_sat_e4_verification.py`
- `scripts/evaluation/eval_sat_cdcl_closedloop.py`
- `scripts/figures/plot_phase_diagram.py`

#### Additional MLJ baselines (paper tab_flagship Panel B)

Train the **current-block-only** mask baseline:

```bash
PYTHONPATH=src python scripts/training/train_history_ablation.py \
  --data_path data/sat-traces/traces.pkl \
  --output_dir experiments/sat-current-block-only \
  --mask_mode local_block_only \
  --model_type transformer \
  --max_seq_len 4096 \
  --batch_size 8
```

Train the **history-transplant** causal baseline (donor blocks at p=1):

```bash
PYTHONPATH=src python scripts/training/train_history_ablation.py \
  --data_path data/sat-traces/traces.pkl \
  --output_dir experiments/sat-history-transplant \
  --mask_mode full_causal \
  --history_mode history_transplant \
  --transplant_prob 1.0 \
  --model_type transformer \
  --max_seq_len 4096 \
  --batch_size 4
```

Add `--partial_transplant` to replace prior history blocks independently with probability `--transplant_prob` instead of swapping the full donor history at once.

Train the **contrastive invariance (λ=1)** causal baseline:

```bash
PYTHONPATH=src python scripts/training/train_contrastive_invariance.py \
  --data_path data/sat-traces/traces.pkl \
  --output_dir experiments/sat-contrastive-invariance \
  --mask_mode full_causal \
  --lambda_kl 1.0 \
  --model_type transformer \
  --max_seq_len 4096 \
  --batch_size 4
```

Train the **Factor GNN** non-transformer reference (~675k params):

```bash
PYTHONPATH=src python scripts/training/train_factor_gnn_sat.py \
  --data_path data/sat-traces/traces.pkl \
  --output_dir experiments/sat-factor-gnn \
  --batch_size 32
```

### 3) Blocks World

- Setting: **7-block** planning with DFS-style backtracking
- Goal: test whether SSA helps separate the current arrangement from obsolete action history

Generate traces:

```bash
PYTHONPATH=src python scripts/data_generation/generate_bw_traces.py \
  --num_instances 5000 \
  --num_blocks 7 \
  --output data/bw-traces/traces.pkl
```

Train SSA and causal models:

```bash
PYTHONPATH=src python scripts/training/train_bw_ssa.py \
  --data_path data/bw-traces/traces.pkl \
  --output_dir experiments/bw-ssa \
  --mode ssa

PYTHONPATH=src python scripts/training/train_bw_ssa.py \
  --data_path data/bw-traces/traces.pkl \
  --output_dir experiments/bw-causal \
  --mode causal
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluation/eval_bw_ssa.py \
  --ssa_checkpoint experiments/bw-ssa/best.pt \
  --causal_checkpoint experiments/bw-causal/best.pt \
  --num_instances 200 \
  --num_blocks 7 \
  --output_dir experiments/bw-eval
```

### 4) Backtracking Parsing

- Setting: backtracking **PEG parsing**
- Goal: study verification under recursive, compositional search traces

Generate traces:

```bash
PYTHONPATH=src python scripts/data_generation/generate_parsing_traces.py \
  --num-traces 5000 \
  --output-dir data/parsing-traces
```

Evaluate trained checkpoints autonomously:

```bash
PYTHONPATH=src python scripts/evaluation/eval_parsing_autonomous.py \
  --checkpoints experiments/parsing-ssa/best.pt,experiments/parsing-causal/best.pt \
  --labels ssa,causal \
  --num-instances 200 \
  --budget 2048 \
  --output-dir experiments/parsing-eval
```

### 5) LLM transfer on MATH

- Models: **Ministral-14B** and **Qwen-4B**
- Goal: explore whether the state-isolation principle behind SSA can motivate context-clearing or verifier-style protocols for large language models. The manuscript frames this LLM connection as an **outlook hypothesis** motivated by symbolic evidence, not as a primary demonstrated capability.

Generate candidate solutions:

```bash
PYTHONPATH=src python scripts/llm/generate_math_candidates.py \
  --model mistralai/Ministral-3-14B-Base-2512 \
  --k 16 \
  --output data/math/ministral_candidates.jsonl
```

Train slot verifier:

```bash
PYTHONPATH=src python scripts/llm/train_slot_verifier.py \
  --data data/math/ministral_candidates.jsonl \
  --output_dir experiments/llm-slot-verifier
```

Evaluate PASV-style gating on MATH:

```bash
PYTHONPATH=src python scripts/llm/eval_pasv.py \
  --eval-path data/math/ministral_eval.jsonl \
  --candidates-path data/math/ministral_candidates.jsonl \
  --output-dir experiments/llm-pasv \
  --include-qwen
```

Reproduce the FP/FN asymmetry analysis on the LLM candidate set:

```bash
# Run on Ministral-14B candidates (primary)
PYTHONPATH=src python scripts/llm/analyze_fpfn_stratified.py \
  --eval-file experiments/math-14b/eval_full_5000.json \
  --candidates-file experiments/math-14b/candidates_test_full_5000.jsonl \
  --output-dir experiments/llm-fpfn/ministral

# Run on Qwen-14B (if eval + candidates were generated)
PYTHONPATH=src python scripts/llm/analyze_fpfn_stratified.py \
  --eval-file experiments/qwen-14b/eval_test.json \
  --candidates-file experiments/qwen-14b/candidates_test_full.jsonl \
  --output-dir experiments/llm-fpfn/qwen14b
```

To include Qwen-14B in the main PASV reranking:

```bash
PYTHONPATH=src python scripts/llm/eval_pasv.py \
  --eval-path data/math/ministral_eval.jsonl \
  --candidates-path data/math/ministral_candidates.jsonl \
  --output-dir experiments/llm-pasv \
  --include-qwen \
  --include-qwen-14b
```

LLM precision-cliff analysis:

```bash
PYTHONPATH=src python scripts/llm/analyze_precision_cliff.py \
  --eval-14b experiments/math-14b/eval_full_5000.json \
  --candidates-14b experiments/math-14b/candidates_test_full_5000.jsonl \
  --eval-qwen experiments/qwen-4b/eval_test.json \
  --candidates-qwen experiments/qwen-4b/candidates_test_full.jsonl \
  --output-dir experiments/llm-precision-cliff
```

> **Note:** LLM experiments require substantially more compute and access to the corresponding Hugging Face checkpoints.

### 6) r^k mechanistic benchmark

- Setting: star-tree-style **r^k** benchmark
- Goal: isolate whether models learn the intended state abstraction rather than shortcut history correlations

Train transformer benchmark model:

```bash
PYTHONPATH=src python scripts/training/train_rk_benchmark.py \
  --model_type transformer \
  --d_model 256 \
  --num_layers 6 \
  --output_dir experiments/rk-transformer
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluation/eval_rk_benchmark.py \
  --checkpoint experiments/rk-transformer/best_model.pt \
  --model_type transformer \
  --output_dir experiments/rk-transformer-eval
```

### 7) Delta-local verification (appendix experiment)

Train the delta-local verification head on graph-coloring traces (paper appendix, Section "Delta-Local Verification Head"):

```bash
PYTHONPATH=src python scripts/training/train_delta_local.py \
  --data data/gc-traces/traces.pkl \
  --output_dir experiments/gc-delta-local \
  --epochs 30 \
  --batch_size 32
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluation/eval_delta_local.py \
  --checkpoint experiments/gc-delta-local/best_model.pt \
  --num_instances 200 \
  --output experiments/gc-delta-local-eval
```

### 8) Verifier-only state-equivalence benchmark

The verifier-only benchmark is the paper's primary diagnostic for **history entanglement** (§sec:verifier-only). It probes whether a model's verification decisions depend on the prior trajectory or only on the current state, by pairing canonical search states with multiple histories and measuring same-state/different-history disagreement.

The pipeline has three steps and is **model-independent**: the probe bank is built once and reused for every checkpoint and protocol.

#### Step 1 — Build a model-free probe bank

```bash
PYTHONPATH=src python scripts/data_generation/build_verifier_probe_bank.py \
  --domain sat \
  --num_vars 50 \
  --alpha 4.0 \
  --n_instances 50 \
  --n_traces_per_instance 10 \
  --max_seq_len 4096 \
  --seed 42 \
  --state_sort lexical \
  --min_histories_per_state 2 \
  --max_histories_per_state 4 \
  --output_path data/verifier-probe-bank/sat_n50.json
```

For graph coloring:

```bash
PYTHONPATH=src python scripts/data_generation/build_verifier_probe_bank.py \
  --domain gc \
  --num_nodes 30 \
  --num_colors 4 \
  --edge_prob 0.35 \
  --n_instances 100 \
  --n_traces_per_instance 6 \
  --max_seq_len 4096 \
  --seed 42 \
  --output_path data/verifier-probe-bank/gc_n30.json
```

The probe bank caches canonical states and their multi-history pairs (paper reports 1,314 SAT and 682 GC canonical states for the main setting).

#### Step 2 — Evaluate a checkpoint on the probe bank

```bash
PYTHONPATH=src python scripts/evaluation/eval_verifier_benchmark.py \
  --probe_bank data/verifier-probe-bank/sat_n50.json \
  --checkpoint experiments/sat-ssa/best.pt \
  --architecture ssa \
  --protocol cumulative \
  --device cuda \
  --bootstrap_iters 1000 \
  --bootstrap_seed 11 \
  --output_path experiments/verifier-bench/sat_n50_ssa_cumulative.json
```

Repeat the call with `--protocol state_rebuilt` and with each architecture you want to compare (`ssa`, `causal`, `lstm`, `factor_gnn`, `current_block_only`, `contrastive`, `block_dropout`, `sliding_window`, `null_history`, `history_transplant`).

The output JSON contains:
- `panel_a`: `alpha_v_overall`, `alpha_v_by_depth_quartile`, `alpha_v_by_size_quartile`, `beta_overall`, `auroc`, `auprc`
- `panel_b`: `argmax_agreement`, `mean_symmetric_kl` (the two same-state/different-history invariance metrics)
- `calibration`: `ece_15bins`, `brier`, `reliability_curve`

#### Step 3 — Aggregate into LaTeX tables

```bash
PYTHONPATH=src python scripts/tables/gen_tab_verifier_benchmark.py \
  --eval_dir experiments/verifier-bench \
  --output_panel_a output/tables/tab_verifier_panel_a.tex \
  --output_panel_b output/tables/tab_verifier_panel_b.tex \
  --output_summary output/tables/verifier_summary.json
  # --output_calibration is optional and defaults next to panel_b
```

This produces the manuscript tables:
- `tab_verifier_panel_a.tex` — invariance metrics (α_v, β, AUROC, AUPRC) by architecture × protocol
- `tab_verifier_panel_b.tex` — same-state/different-history argmax agreement, KL divergence
- `tab_verifier_calibration.tex` — ECE and Brier calibration

#### Smoke test

A self-contained end-to-end smoke test runs the full pipeline (probe-bank build → checkpoint eval) on a tiny CPU model in <10 seconds:

```bash
pytest tests/test_verifier_benchmark.py -v
```

## Reproducing tables and figures

The repository also includes utilities used for paper artifacts:

- `scripts/tables/` for LaTeX-ready summary tables,
- `scripts/figures/` for figure generation,
- `scripts/llm/generate_latex_tables.py` for the LLM transfer section.

Representative examples:

```bash
PYTHONPATH=src python scripts/tables/generate_paper_tables.py
PYTHONPATH=src python scripts/figures/plot_paper_figures.py
```

## Citation

If you use this code or build on the SSA formulation, please cite:

```bibtex
TBD
```

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
