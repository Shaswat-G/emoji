# Unicode Emoji Standardization Research Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Research-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

**A comprehensive data science pipeline for analyzing Unicode Technical Consortium (UTC) decision-making processes through multi-source data integration, NLP enrichment, and computational social science methods.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Pipeline Components](#pipeline-components)
  - [Document Register Suite](#1-document-register-processing-suite)
  - [Email Archive Suite](#2-email-archive-processing-suite)
  - [Proposal Analysis Suite](#3-emoji-proposal-processing-suite)
- [Data Products](#data-products)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Research Applications](#research-applications)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Citation](#citation)

---

## 🎯 Overview

This repository provides an end-to-end research infrastructure for studying institutional standardization processes through the lens of emoji proposals at the Unicode Consortium. The pipeline transforms 20+ years of heterogeneous UTC data into a unified analytical corpus suitable for:

- **Computational Social Science**: Network analysis of authorship, organizational dynamics, and attention patterns
- **Digital Humanities**: Discourse analysis of technical deliberations and proposal evolution
- **Science & Technology Studies**: Institutional decision-making and standards governance research
- **Natural Language Processing**: Large-scale semantic enrichment and entity extraction

**Data Sources:**
- UTC Document Registry (1993–present): ~10,000+ technical documents
- Mailing List Archives (1993–present): ~50,000+ email threads
- Emoji Proposal Tables: 1,000+ proposals with acceptance/rejection outcomes

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🔄 Multi-Format Parsing** | Handles PDF, HTML, TXT, and legacy archive formats with robust error recovery |
| **🤖 LLM Enrichment** | OpenAI-based semantic extraction of entities, people, references, and summaries |
| **🔗 Cross-Source Triangulation** | Links proposals across documents, emails, and outcomes with confidence scoring |
| **📊 Temporal Analytics** | Tracks proposal lifecycles, processing velocity, and attention dynamics over time |
| **🌐 Network Construction** | Builds authorship, citation, and discourse networks for graph analysis |
| **⚡ Production-Ready** | Batching, checkpointing, logging, and parallel processing throughout |

---

## 🏗️ Architecture

```
┌─────────────────────┐
│  Raw Data Sources   │
│  ┌───────────────┐  │
│  │ UTC Registry  │  │ → 📄 Document scraping & classification
│  │ Email Archives│  │ → 📧 Email parsing & threading
│  │ Proposal HTML │  │ → 🎨 Emoji proposal extraction
│  └───────────────┘  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│        Processing Pipeline              │
│  ┌───────────────────────────────────┐  │
│  │ Extraction → Enrichment → Linking│  │
│  │  • Text/metadata extraction       │  │
│  │  • LLM semantic annotation        │  │
│  │  • Entity & reference mining      │  │
│  │  • Cross-source triangulation     │  │
│  └───────────────────────────────────┘  │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Analytical Outputs                 │
│  • Enriched datasets (Excel/CSV)        │
│  • Proposal timelines & reports (MD)    │
│  • Network graphs (PNG/GraphML)         │
│  • Temporal metrics (CSV/visualizations)│
└─────────────────────────────────────────┘
```

---

## 🔧 Pipeline Components

### 1. Document Register Processing Suite

**Purpose:** Transform UTC's document registry into a semantically enriched, analytically-ready corpus.

| Script | Function | Inputs | Outputs |
|--------|----------|--------|---------|
| `utc_doc_reg_scraper.py` | Scrape yearly UTC registers; compile metadata | Web pages | `utc_register_all.xlsx` |
| `utc_doc_reg_process_classify.py` | NLP-based classification & emoji-relevance tagging | Registry Excel | `utc_register_all_classified.xlsx` |
| `utc_doc_extractor.py` | Bulk download & text extraction (PDF/HTML/DOC) | Classified registry | `utc_register_with_text.xlsx` + text files |
| `utc_doc_summarizer.py` | RAKE/YAKE keyword extraction + LSA summarization | Text files | `utc_register_with_text_and_summary.xlsx` |
| `utc_doc_llmsweep.py` | LLM-powered entity extraction (people, orgs, emoji refs) | Document texts | `utc_register_with_llm_extraction.xlsx` |
| `utc_doc_reg_analyze.py` | Statistical analysis & visualization generation | Enriched registry | Plots, metrics CSVs |

**Key Outputs:**
- 📊 `utc_register_with_llm_extraction.xlsx` – Fully enriched document corpus
- 📈 Temporal flow metrics, authorship networks, category evolution plots

---

### 2. Email Archive Processing Suite

**Purpose:** Parse 20+ years of mailing list archives into structured, semantically annotated datasets.

#### New Format (1999–Present)

| Script | Function | Inputs | Outputs |
|--------|----------|--------|---------|
| `utc_new_email_parser.py` | Parse TXT digests → threaded Excel workbooks | Raw mailing list TXT | `parsed_excels/*.xlsx` |
| `utc_new_email_doc_ref_extractor.py` | Extract emoji glyphs, Unicode refs, L2 doc citations | Parsed emails | `*_doc_refs.xlsx` |
| `utc_new_email_llmsweep.py` | LLM extraction: people, entities, abstracts | Parsed emails | `*_llm.xlsx` |
| `utc_new_email_concatenator.py` | Merge monthly files → master corpus | Monthly LLM outputs | `email_archive_llmsweep.xlsx` |

#### Legacy Format (1993–1998)

| Script | Function | Inputs | Outputs |
|--------|----------|--------|---------|
| `utc_old_email_parser.py` | Parse Hypermail HTML archives | HTML archive pages | `utc_email_old_archive_parsed.xlsx` |
| `utc_old_email_doc_ref_extractor.py` | Entity extraction for legacy emails | Parsed old emails | `*_doc_refs.xlsx` |
| `utc_old_email_llmsweep.py` | LLM semantic enrichment | Old email corpus | `*_with_llm_extraction.xlsx` |

#### Unification

| Script | Function | Inputs | Outputs |
|--------|----------|--------|---------|
| `utc_email_new_old_concatenator.py` | Schema harmonization & corpus merging | New + old email datasets | `utc_email_combined_with_llm_extraction_doc_ref.xlsx` |

**Key Outputs:**
- 📧 `utc_email_combined_with_llm_extraction_doc_ref.xlsx` – Unified 30-year email corpus
- 🔗 Document reference mappings, thread structures, participant networks

---

### 3. Emoji Proposal Processing Suite

**Purpose:** Extract proposal metadata, map to emoji outcomes, and triangulate across documents & emails.

| Script | Function | Inputs | Outputs |
|--------|----------|--------|---------|
| `utc_emoji_proposal_extractor.py` | Scrape & deduplicate proposal tables | HTML proposal pages | `emoji_proposal_table.csv/.json` |
| `utc_proposal_emoji_mapper.py` | Map emoji glyphs → proposal documents | HTML mapping page | `emoji_to_proposal_map.csv` |
| `utc_proposal_computer.py` | Cross-reference proposals with UTC registry | Proposals + registry | Console validation report |
| `utc_proposal_email_triangulator.py` | Match proposals to emails via doc-ID + semantic similarity | Proposals + emails | `emoji_proposal_email_matches.csv` |
| `utc_proposal_flow_velocity.py` | Compute lifecycle metrics (velocity, dormancy, revival) | Triangulated data | `proposal_flow_velocity_metrics.csv` |
| `utc_proposal_summarizer.py` | Aggregate attention dynamics & processing statistics | All proposal data | In-memory summary DataFrames |
| `utc_proposal_triangulator.py` | Generate per-proposal reports with timelines & actors | All sources | `proposal_reports/*.md` |

**Additional Analysis Scripts:**
- `analyze_accepted_proposal_dataset.py` – Acceptance patterns, processing time analysis
- `analyze_rejected_proposal_dataset.py` – Rejection patterns, outcome prediction features
- `utc_throughput_analyzer[_v2|_new].py` – Institutional throughput & bottleneck analysis

**Key Outputs:**
- 🎯 `emoji_proposal_email_matches.csv` – High-confidence proposal-email links
- 📝 `proposal_reports/` – Per-proposal Markdown reports with full provenance
- 📊 `proposal_flow_velocity_metrics.csv` – Temporal dynamics metrics
- 🔍 `proposal_attention_dynamics_metrics.csv` – Stakeholder engagement patterns

---

## 📦 Data Products

The pipeline produces three primary analytical datasets:

1. **Enriched Document Registry** (`utc_register_with_llm_extraction.xlsx`)
   - 10,000+ UTC documents with extracted entities, classifications, and summaries
   - Columns: `doc_num`, `date`, `source`, `subject`, `document_type`, `emoji_relevance`, `people`, `entities`, `emoji_references`, `summary`

2. **Unified Email Corpus** (`utc_email_combined_with_llm_extraction_doc_ref.xlsx`)
   - 50,000+ emails (1993–present) with threaded metadata and semantic annotations
   - Columns: `year`, `month`, `date`, `from_name`, `from_email`, `subject`, `people`, `entities`, `doc_references`, `emoji_mentions`

3. **Proposal Triangulation Dataset** (multiple files)
   - `emoji_proposal_table.csv` – Base proposal metadata
   - `emoji_proposal_email_matches.csv` – Proposal-email linkages with confidence scores
   - `proposal_flow_velocity_metrics.csv` – Temporal processing metrics
   - `proposal_attention_dynamics_metrics.csv` – Stakeholder engagement patterns

---

## 🚀 Installation & Setup

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages

```txt
pandas>=1.3.0
numpy>=1.21.0
beautifulsoup4>=4.9.3
requests>=2.26.0
openpyxl>=3.0.7
lxml>=4.6.3
openai>=1.0.0
pyyaml>=5.4.1
tqdm>=4.62.0
emoji>=1.7.0
matplotlib>=3.4.2
seaborn>=0.11.1
networkx>=2.6.0
scikit-learn>=0.24.2
rake-nltk>=1.0.4
yake>=0.4.8
```

### Configuration

1. **API Key Setup** (for LLM enrichment):
```bash
echo "your-openai-api-key" > api_key.txt
```

2. **Configure LLM Parameters** (optional):
Edit `config.yml`, `email_config.yml`, `proposal_config.yml` to adjust:
- Model selection (`gpt-4o-mini-2024-07-18`)
- Temperature, max tokens, response format
- Prompt templates (`prompt.txt`, `email_prompt.txt`)

---

## 💻 Usage

### Full Pipeline Execution

Run the complete pipeline in sequence:

```bash
# 1. Document Registry Processing
python utc_doc_reg_scraper.py
python utc_doc_reg_process_classify.py
python utc_doc_extractor.py
python utc_doc_summarizer.py
python utc_doc_llmsweep.py

# 2. Email Archive Processing
python utc_new_email_parser.py
python utc_new_email_llmsweep.py
python utc_new_email_doc_ref_extractor.py
python utc_new_email_concatenator.py
python utc_old_email_parser.py
python utc_old_email_llmsweep.py
python utc_old_email_doc_ref_extractor.py
python utc_email_new_old_concatenator.py

# 3. Proposal Analysis
python utc_emoji_proposal_extractor.py
python utc_proposal_emoji_mapper.py
python utc_proposal_email_triangulator.py
python utc_proposal_flow_velocity.py
python utc_proposal_triangulator.py

# 4. Analysis & Visualization
python analyze_accepted_proposal_dataset.py
python analyze_rejected_proposal_dataset.py
python utc_doc_reg_analyze.py
python utc_throughput_analyzer_v2.py
```

### Modular Execution

Each script is standalone and can be run independently if input data exists:

```bash
# Quick proposal-email matching (requires pre-existing inputs)
python utc_proposal_email_triangulator.py

# Generate individual proposal reports
python utc_proposal_triangulator.py

# Analyze only document registry statistics
python utc_doc_reg_analyze.py
```

---

## 🔬 Research Applications

This pipeline enables multiple research paradigms:

### Computational Social Science
- **Network Analysis**: Author collaboration networks, citation graphs, organizational influence mapping
- **Temporal Dynamics**: Proposal lifecycle modeling, institutional throughput analysis, attention evolution
- **Predictive Modeling**: Acceptance/rejection prediction, processing time estimation

### Digital Humanities & Discourse Analysis
- **Corpus Linguistics**: Technical vocabulary evolution, discourse marker analysis
- **Thematic Analysis**: Topic modeling, semantic clustering of proposals
- **Rhetorical Analysis**: Argumentation patterns in proposal justifications

### Science & Technology Studies
- **Standardization Politics**: Stakeholder coalitions, organizational gatekeeping
- **Innovation Diffusion**: Proposal adoption patterns, cultural representation in emoji
- **Governance Analysis**: Decision-making transparency, procedural evolution

### Data Science & Machine Learning
- **Feature Engineering**: Rich temporal, network, and textual features for classification
- **Information Extraction**: Named entity recognition, relation extraction benchmarking
- **Transfer Learning**: Fine-tuning LLMs on domain-specific UTC discourse

---

## 📁 Project Structure

```
emoji/
├── 📜 Core Pipeline Scripts
│   ├── utc_doc_reg_*.py          # Document registry suite (6 scripts)
│   ├── utc_*_email_*.py          # Email processing suite (8 scripts)
│   └── utc_*_proposal_*.py       # Proposal analysis suite (7 scripts)
│
├── 🔍 Analysis Scripts
│   ├── analyze_*.py              # Acceptance/rejection analysis
│   └── utc_throughput_*.py       # Throughput & velocity analysis
│
├── 🛠️ Utilities
│   ├── llm_api_infra.py          # OpenAI client wrapper
│   ├── *_scraper_*.py            # Web scraping utilities
│   └── gender_llmsweep_*.py      # Gender analysis modules
│
├── 📊 Data Files
│   ├── data/                     # Raw input data
│   ├── extracted_texts/          # Extracted document texts
│   ├── parsed_excels/            # Parsed email workbooks
│   └── *.xlsx / *.csv            # Processed datasets
│
├── 📈 Outputs
│   ├── proposal_reports/         # Per-proposal Markdown reports
│   ├── plots/                    # Visualization outputs
│   ├── throughput_analysis/      # Institutional metrics
│   └── *_metrics.csv             # Quantitative results
│
├── ⚙️ Configuration
│   ├── config*.yml               # LLM & pipeline configs
│   ├── *_prompt.txt              # LLM prompt templates
│   └── stack-overflow-environment.yml  # Conda environment
│
└── 📚 Documentation
    ├── README.md                 # This file
    ├── code_documentation.xlsx   # Comprehensive script inventory
    ├── data_documentation.xlsx   # Dataset schemas & dictionaries
    └── utc_doc_reg_datadictionary.md  # Registry field descriptions
```

---

## 📖 Documentation

- **`code_documentation.xlsx`**: Comprehensive inventory of all scripts with detailed function descriptions, I/O specifications, and dependencies
- **`data_documentation.xlsx`**: Complete data dictionary for all generated datasets including schema definitions, field descriptions, and example values
- **`utc_doc_reg_datadictionary.md`**: Detailed documentation of UTC document registry structure and classification scheme
- **Script Headers**: Each Python file includes an embedded header describing purpose, inputs, outputs, and engineering notes

---

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{utc_emoji_pipeline,
  title={Unicode Emoji Standardization Research Pipeline},
  author={[Your Name/Organization]},
  year={2024},
  url={https://github.com/Shaswat-G/emoji},
  note={A comprehensive data science infrastructure for analyzing Unicode Technical Consortium decision-making processes}
}
```

---

## 📧 Contact & Collaboration

For questions, bug reports, feature requests, or research collaboration inquiries:

- **GitHub Issues**: [Report issues](https://github.com/Shaswat-G/emoji/issues)
- **Email**: [Contact maintainer]
- **Collaboration**: Open to academic partnerships and data sharing agreements

---

## 🙏 Acknowledgments

This research infrastructure was developed to support computational social science research on standardization processes. We acknowledge:
- Unicode Consortium for maintaining open archives
- OpenAI for LLM API access
- Open-source Python scientific computing community

---

<div align="center">

**⭐ Star this repository if you find it useful for your research! ⭐**

![Emoji Process Flow](Emoji_Process_Flow.png)

</div>
