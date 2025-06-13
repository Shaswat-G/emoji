# Unicode Technical Consortium (UTC) Decision-Making Data Pipeline

A modular, research-grade pipeline for transforming the Unicode Technical Consortium’s (UTC) public data—spanning document registers, email archives, and emoji proposals—into a semantically enriched corpus for quantitative, network, and content analysis of UTC’s emoji standardization and decision-making processes.

---

## Table of Contents

- [Unicode Technical Consortium (UTC) Decision-Making Data Pipeline](#unicode-technical-consortium-utc-decision-making-data-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Document-Register Processing Suite](#document-register-processing-suite)
  - [Email-Archive Processing Suite](#email-archive-processing-suite)
  - [Emoji-Proposal Processing Suite](#emoji-proposal-processing-suite)
  - [Pipeline Highlights](#pipeline-highlights)
  - [Research Utility](#research-utility)
  - [Project Context](#project-context)

---

## Overview

This repository is organized into three main suites:

- **Document-Register Processing Suite**: Scrapes, classifies, extracts, and annotates UTC document registers and their contents.
- **Email-Archive Processing Suite**: Harvests, parses, and semantically enriches two decades of UTC mailing list traffic.
- **Emoji-Proposal Processing Suite**: Extracts, maps, and triangulates emoji proposals, linking them to documents and email discourse.

Each suite is composed of standalone scripts that can be run sequentially for a full pipeline or individually for targeted analyses.

---

## Document-Register Processing Suite

<details>
<summary><strong>Click to expand: Scripts & Functions</strong></summary>

| #   | Script                            | Function (one-liner)                                                                                | Principal I/O                                                                                |
| --- | --------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 1   | `utc_doc_reg_scraper.py`          | Scrape yearly register pages and collate raw document metadata.                                     | In: none Out: `utc_register_<year>.csv`, `utc_register_all.xlsx`                             |
| 2   | `utc_doc_reg_process_classify.py` | Tag each record with hierarchical type labels and an emoji-relevance flag via rule-based NLP.       | In: `utc_register_all.xlsx` Out: `utc_register_all_classified.xlsx`                          |
| 3   | `utc_doc_extractor.py`            | Bulk-download source files, extract & clean text, and append content-level emoji/Unicode citations. | In: `utc_register_all_classified.xlsx` Out: `utc_register_with_text.xlsx`, text files        |
| 4   | `utc_doc_summarizer.py`           | Enrich each document with RAKE/YAKE keywords and LSA summaries using multithreaded NLP.             | In: `utc_register_with_text.xlsx`, text files Out: `utc_register_with_text_and_summary.xlsx` |
| 5   | `utc_doc_llmsweep.py`             | Invoke an LLM to yield structured JSON (people, entities, emoji refs, abstract) for every text.     | In: document texts, `prompt.txt`, `config.yml` Out: tabular JSON → CSV/Excel                 |
| 6   | `utc_doc_reg_analyze.py`          | Generate plots & stats on document flow, authorship, category trends, and emoji salience.           | In: `utc_register_all_classified.xlsx` (or later outputs) Out: PNGs, PDFs, XLSX summaries    |

</details>

---

## Email-Archive Processing Suite

<details>
<summary><strong>Click to expand: Scripts & Functions</strong></summary>

| #   | Script                               | Archive Epoch                                      | Purpose (one-liner)                                                               | Principal I/O                                                                             |
| --- | ------------------------------------ | -------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1   | `utc_new_email_parser.py`            | 1999 → present (`corp.unicode.org/pipermail/…`)    | Crawl, download, parse, and thread TXT digests → mailbox-quality Excel.           | In: raw `.txt` digests Out: per-month Excel in `parsed_excels/`                           |
| 2   | `utc_new_email_doc_ref_extractor.py` | new                                                | Detect emoji glyphs, code points, short-codes, and L2 references in email bodies. | In: `utc_email_with_llm_extraction.xlsx` Out: `_doc_refs.xlsx`                            |
| 3   | `utc_new_email_llmsweep.py`          | new                                                | LLM-extract high-level fields (people, entities, abstract, emoji relevance).      | In: parsed Excels + `email_prompt.txt` Out: `_llm.xlsx`                                   |
| 4   | `utc_new_email_concatenator.py`      | new                                                | Union all `_llm.xlsx` files into a master workbook.                               | In: `parsed_excels/*llm.xlsx` Out: `email_archive_llmsweep.xlsx`                          |
| 5   | `utc_old_email_parser.py`            | 1993 – 1998 (`unicode.org/mail-arch/unicode-ml/…`) | Fetch HTML pages, parse Hypermail 2.x, normalise → Excel.                         | In: `mail_archive_old_format.xlsx` Out: `utc_email_old_archive_parsed.xlsx`               |
| 6   | `utc_old_email_doc_ref_extractor.py` | old                                                | Same entity extraction as #2 for legacy corpus.                                   | In: `*_llm_extraction.xlsx` Out: `_doc_refs.xlsx`                                         |
| 7   | `utc_old_email_llmsweep.py`          | old                                                | LLM sweep for semantic fields (people, summary, etc.).                            | In: parsed old Excel Out: `_with_llm_extraction_testing.xlsx`                             |
| 8   | `utc_email_new_old_concatenator.py`  | both                                               | Harmonise schemas and merge new + old corpora into a unified workbook.            | In: new + old `_doc_refs.xlsx` Out: `utc_email_combined_with_llm_extraction_doc_ref.xlsx` |

</details>

---

## Emoji-Proposal Processing Suite

<details>
<summary><strong>Click to expand: Scripts & Functions</strong></summary>

| #   | Script                               | Function (one-liner)                                                                                   | Principal I/O                                                                                                                 |
| --- | ------------------------------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1   | `utc_emoji_proposal_extractor.py`    | Harvest and deduplicate proposal rows from two HTML tables; emit canonical JSON/CSV.                   | In: `emoji_proposals_v16.html`, `emoji_proposals_v13.html` Out: `emoji_proposal_table.{json,csv}`                             |
| 2   | `utc_proposal_emoji_mapper.py`       | Map each emoji glyph to its proposal doc # and category header.                                        | In: `emoji_to_proposal_map.html` Out: `emoji_to_proposal_map.csv`                                                             |
| 3   | `utc_proposal_computer.py`           | Cross-check proposal sets against the classified UTC register; log proposal-doc counts.                | In: `emoji_to_proposal_map.csv`, `emoji_proposal_table.csv`, `utc_register_all_classified.xlsx` Out: console tally            |
| 4   | `utc_proposal_email_triangulator.py` | Match proposals to emails via doc-ID hits and n-gram semantic overlap within ±2 yrs; score confidence. | In: `emoji_proposal_table.csv`, `utc_email_combined_with_llm_extraction_doc_ref.xlsx` Out: `emoji_proposal_email_matches.csv` |
| 5   | `utc_proposal_summarizer.py`         | Normalise doc IDs and compute flow / attention metrics across register + email evidence.               | In: proposal table, enriched register, email matches Out: in-memory DataFrames for analytics                                  |
| 6   | `utc_proposal_triangulator.py`       | Generate per-proposal Markdown reports (timeline, actors, citations) and a ranked summary CSV.         | In: same three datasets as #5 Out: `proposal_reports/*.md`, `proposal_summary.csv`                                            |

</details>

---

## Pipeline Highlights

<details>
<summary><strong>Click to expand: Key Features</strong></summary>

- **Format-aware harvesting**: Handles both new and legacy email formats, HTML and PDF document types, and robustly parses and threads messages.
- **Entity & reference mining**: Uses regex and the `emoji` library to extract Unicode glyphs, shortcodes, and document cross-references.
- **LLM semantic enrichment**: Configurable OpenAI prompts extract structured metadata and summaries from both documents and emails.
- **Schema harmonisation**: Standardizes date formats, thread IDs, and column names for seamless merging and longitudinal analysis.
- **Batching, checkpointing, and error handling**: All scripts are designed for large-scale, fault-tolerant processing.

</details>

---

## Research Utility

<details>
<summary><strong>Click to expand: Analytical Possibilities</strong></summary>

- **Quantitative history** of UTC deliberations and proposal lifecycles
- **Network analysis** of authorship and organizational attention
- **Thematic trend detection**—especially around emoji standardization
- **Corpus linguistics** on technical discourse evolution
- **Lifecycle analytics** for emoji proposals and decision-making
- **Discourse mapping** linking formal proposals to informal debate
- **Category evolution** tracking and cluster identification

</details>

---

## Project Context

This pipeline is designed for research into the UTC’s decision-making process, with a focus on emoji standardization. By integrating proposals, documents, and email communications, it enables robust, reproducible studies of institutional workflows, stakeholder engagement, and the evolution of Unicode standards.

Each script is documented and modular. Run sequentially for end-to-end processing, or select individual stages for custom analyses. All outputs are designed for interoperability and downstream quantitative or qualitative research.

---

For questions, collaboration, or citation, please contact the repository maintainer.
