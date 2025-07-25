# -----------------------------------------------------------------------------
# Script: utc_proposal_triangulator.py
# Summary: Generates detailed, proposal-centric reports triangulating UTC
#          document flow without matched email communications, but with contextual and
#          statistical analysis for each emoji proposal.
# Inputs:  rejected_proposals.csv (Charlotte Buff's list), utc_register_with_llm_document_classification.xlsx,
# Outputs: Markdown reports per proposal (rejected_proposal_reports/), summary CSV
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------

import os
import pandas as pd
import json
import ast
import re
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from tqdm import tqdm


def safe_literal_eval(val):
    """Safely parse string representation of dictionary back to dict"""
    try:
        if pd.isna(val) or val == "":
            return {}
        if isinstance(val, dict):
            return val
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}


def load_set_from_excel(filename, column, filter_func=None):
    """Load a set from Excel file with optional filtering"""
    df = pd.read_excel(os.path.join(os.getcwd(), filename))
    if filter_func:
        df = df[filter_func(df)]
    return set(df[column].dropna())


def load_set_from_csv(filename, column, filter_func=None):
    """Load a set from CSV file"""
    df = pd.read_csv(os.path.join(os.getcwd(), filename))
    if filter_func:
        df = df[filter_func(df)]
    return set(df[column].dropna())


# --- Filter helpers ---
def in_range(s):
    return (
        isinstance(s, str)
        and len(s) >= 5
        and s[3:5].isdigit()
        and 11 <= int(s[3:5]) <= 20
    )


def filter_utc_doc_reg(df):
    return df["is_emoji_proposal"] == True


def filter_emoji_proposals(df):
    return df["doc_num"].apply(in_range)


def filter_cb_rejections(df):
    return df["document"].apply(in_range)


all_identified_emoji_proposals = load_set_from_excel(
    "utc_register_with_llm_document_classification.xlsx",
    "doc_num",
    filter_utc_doc_reg,
)

charlotte_buff_rejected = load_set_from_csv(
    "rejected_proposals.csv", "document", filter_cb_rejections
)
known_accepted_proposals = load_set_from_csv(
    "emoji_proposal_table.csv", "doc_num", filter_emoji_proposals
)

estimated_rejected_proposals = all_identified_emoji_proposals - known_accepted_proposals

base_path = os.getcwd()
# emoji_proposal_path = os.path.join(base_path, "rejected_proposals.csv")   # CB's incomplete list of rejected proposal
# emoji_proposal_df = pd.read_csv(emoji_proposal_path, dtype=str)
utc_doc_reg_path = os.path.join(
    base_path, "utc_register_with_llm_document_classification.xlsx"
)
# utc_email_path = os.path.join(base_path, "emoji_proposal_email_matches.csv") ---- we do not have email matches for rejected proposals yet.


def safe_literal_eval(val):
    try:
        # Handle empty or NaN values
        if pd.isna(val) or val == "":
            return None
        return ast.literal_eval(val)
    except Exception:
        return val


# Identify columns that need to be parsed as Python objects
columns_to_eval = [
    "document_classification",
    "extracted_doc_refs",
    "emoji_chars",
    "unicode_points",
    "emoji_keywords_found",
    "emoji_shortcodes",
    "people",
    "emoji_references",
    "entities",
]

try:
    utc_doc_reg_df = pd.read_excel(utc_doc_reg_path)

    # Then apply converters manually to specific columns after loading
    for col in columns_to_eval:
        if col in utc_doc_reg_df.columns:
            utc_doc_reg_df[col] = utc_doc_reg_df[col].apply(safe_literal_eval)
except Exception as e:
    print(f"Error loading or processing the Excel file: {e}")


# rejected_proposals = list(set(emoji_proposal_df["document"]))
rejected_proposals = list(estimated_rejected_proposals)
authors = {}

for proposal in rejected_proposals:
    # Get all author strings for this proposal
    author_strs = (
        utc_doc_reg_df[utc_doc_reg_df["doc_num"] == proposal]["source"]
        .dropna()
        .tolist()
    )
    # Split each string by comma, strip whitespace, flatten, and deduplicate
    author_list = []
    for s in author_strs:
        author_list.extend([a.strip() for a in s.split(",") if a.strip()])
    authors[proposal] = list(set(author_list))

relevant_columns = ["doc_num", "subject"]
new_proposal_df = utc_doc_reg_df[utc_doc_reg_df["doc_num"].isin(rejected_proposals)][
    relevant_columns
].reset_index(drop=True)
renamed_columns = ["doc_num", "proposal_title"]
new_proposal_df.columns = renamed_columns

# Add proposer column using the authors map
new_proposal_df["proposer"] = new_proposal_df["doc_num"].map(
    lambda doc: ", ".join(authors.get(doc, []))
)

# del emoji_proposal_df
emoji_proposal_df = new_proposal_df.copy()
del new_proposal_df


def normalize_doc_num(doc_num):
    """
    Normalize document numbers to handle encoding issues and format variations.
    Examples: "L2/23â€'261", "L2/23-261", "l2/23-261" should all match.
    """
    if not isinstance(doc_num, str):
        return ""

    # Extract the standard pattern: L2/YY-XXX
    # Handle various dash characters (hyphen, en-dash, em-dash)
    match = re.search(r"L2/(\d{2})[-\u2013\u2014](\d{3})", doc_num, re.IGNORECASE)
    if match:
        year, number = match.groups()
        return f"L2/{year}-{number}"
    return doc_num


def track_proposal_through_time(proposal_id, utc_df):
    """
    Track a specific proposal through time in the UTC document registry.

    Args:
        proposal_id: The document number to track (e.g., "L2/19-080")
        utc_df: The UTC document register dataframe

    Returns:
        DataFrame containing all mentions of this proposal in chronological order
    """
    # Normalize the proposal ID to handle encoding issues
    normalized_proposal_id = normalize_doc_num(proposal_id)

    # Normalize all doc_nums in the dataframe for matching
    utc_df_copy = utc_df.copy()
    utc_df_copy["normalized_doc_num"] = utc_df_copy["doc_num"].apply(normalize_doc_num)

    # 1. Find the original proposal document (direct match on doc_num)
    direct_matches = utc_df_copy[
        utc_df_copy["normalized_doc_num"] == normalized_proposal_id
    ].copy()

    # 2. Find all documents that reference this proposal
    def references_proposal(refs_list):
        if not isinstance(refs_list, list):
            return False

        # Normalize each reference and check for a match
        for ref in refs_list:
            if normalize_doc_num(ref) == normalized_proposal_id:
                return True
        return False

    reference_matches = utc_df_copy[
        utc_df_copy["extracted_doc_refs"].apply(references_proposal)
    ].copy()

    # 3. Combine direct and reference matches
    all_matches = pd.concat([direct_matches, reference_matches]).drop_duplicates(
        subset=["doc_num"]
    )

    # 4. Mark each document as either the original proposal or a reference
    all_matches["reference_type"] = all_matches["normalized_doc_num"].apply(
        lambda x: "Original Proposal" if x == normalized_proposal_id else "Reference"
    )

    # 5. Sort by date to create a chronological timeline
    if not all_matches.empty and "date" in all_matches.columns:
        try:
            all_matches = all_matches.sort_values("date")
        except Exception as e:
            print(f"Error sorting dates for {proposal_id}: {e}")
            # Fallback: try to convert dates if needed
            if "date" in all_matches.columns:
                all_matches["date"] = pd.to_datetime(
                    all_matches["date"], errors="coerce"
                )
                all_matches = all_matches.sort_values("date")

    return all_matches.drop(columns=["normalized_doc_num"])


def analyze_proposal_context(timeline_df):
    """
    Analyze the context of a proposal's mentions over time.

    Args:
        timeline_df: DataFrame from track_proposal_through_time

    Returns:
        Dictionary with contextual analysis
    """
    context = {
        "doc_count": len(timeline_df),
        "date_range": (
            (timeline_df["date"].min(), timeline_df["date"].max())
            if not timeline_df.empty
            else ("Unknown", "Unknown")
        ),
        "people_involved": set(),
        "entities_involved": set(),
        "emoji_mentioned": set(),
        "unicode_points": set(),
        "key_topics": defaultdict(int),
    }

    # Collect people involved
    for people_list in timeline_df["people"]:
        if isinstance(people_list, list):
            context["people_involved"].update(people_list)

    # Collect entities involved
    for entity_list in timeline_df["entities"]:
        if isinstance(entity_list, list):
            context["entities_involved"].update(entity_list)

    # Collect emoji mentioned
    for emoji_list in timeline_df["emoji_chars"]:
        if isinstance(emoji_list, list):
            context["emoji_mentioned"].update(emoji_list)

    # Collect unicode points
    for points_list in timeline_df["unicode_points"]:
        if isinstance(points_list, list):
            context["unicode_points"].update(points_list)

    # Extract key topics from summaries
    for summary in timeline_df["summary"]:
        if isinstance(summary, str):
            # Simple frequency analysis of key terms
            words = re.findall(r"\b\w{5,}\b", summary.lower())
            for word in words:
                # Filter out common stopwords
                if word not in ["which", "there", "their", "about", "would"]:
                    context["key_topics"][word] += 1

    # Sort key topics by frequency
    context["key_topics"] = dict(
        sorted(context["key_topics"].items(), key=lambda item: item[1], reverse=True)[
            :30
        ]
    )  # Keep top 30 topics

    return context


def analyze_all_emoji_proposals():
    """Analyze all emoji proposals in the dataset"""
    # Create output directory for reports
    reports_dir = os.path.join(base_path, "rejected_proposal_reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Track all proposals and generate reports
    all_timelines = {}
    proposal_stats = []

    # Directly iterate over emoji proposal dataframe rows
    # Assuming all required columns exist: doc_num, proposal_title, proposer, proposal_link
    for _, row in tqdm(
        emoji_proposal_df.iterrows(),
        total=emoji_proposal_df.shape[0],
        desc="Analyzing proposals",
    ):
        proposal_id = normalize_doc_num(row["doc_num"])
        proposal_title = row["proposal_title"]
        proposer = row["proposer"]

        print(f"Analyzing proposal: {proposal_id}")

        # Track this proposal through time in the UTC document registry
        timeline = track_proposal_through_time(proposal_id, utc_doc_reg_df)
        all_timelines[proposal_id] = timeline

        # Collect stats for this proposal
        if not timeline.empty:
            context = analyze_proposal_context(timeline)

            stats = {
                "proposal_id": proposal_id,
                "proposal_title": proposal_title,
                "proposer": proposer,
                "reference_count": len(timeline) - 1,  # Subtract the original document
                "date_range": f"{context['date_range'][0]} to {context['date_range'][1]}",
                "contributor_count": len(context["people_involved"]),
                "entity_count": len(context["entities_involved"]),
                "emoji_count": len(context["emoji_mentioned"]),
                "top_contributors": ", ".join(list(context["people_involved"])[:3]),
            }
            proposal_stats.append(stats)

            # Generate detailed report
            report = generate_proposal_timeline_report(
                proposal_id, timeline, proposal_title, proposer
            )

            # Save report to file
            safe_filename = re.sub(r'[\\/*?:"<>|]', "_", proposal_id)
            report_path = os.path.join(reports_dir, f"{safe_filename}_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

    # Create summary report of all proposals
    if proposal_stats:
        stats_df = pd.DataFrame(proposal_stats)
        stats_df = stats_df.sort_values("reference_count", ascending=False)
        stats_path = os.path.join(reports_dir, "all_proposals_summary.csv")
        stats_df.to_csv(stats_path, index=False)

        print(f"\nProposal analysis complete. Summary saved to {stats_path}")
        print("\nTop 5 most referenced proposals:")
        print(
            stats_df[["proposal_id", "proposal_title", "reference_count"]].head(5)
        )  # Create visualization for proposals with sufficient data
    active_proposals = [
        p_id for p_id, timeline in all_timelines.items() if len(timeline) >= 3
    ]
    if active_proposals:
        print(
            f"\nCreating timeline visualizations for {len(active_proposals)} proposals"
        )
        # We'll create individual timeline visualizations instead of a network
        # Each proposal's timeline will be visualized in its own report

    return all_timelines


def generate_proposal_timeline_report(
    proposal_id, timeline_df=None, proposal_title="Unknown", proposer="Unknown"
):
    """
    Generate a comprehensive report of a proposal's journey through time

    Args:
        proposal_id: The document number of the proposal
        timeline_df: Optional pre-computed timeline DataFrame
        proposal_title: Title of the proposal if known
        proposer: Name of the proposal's submitter(s)

    Returns:
        Formatted text report as a string
    """
    if timeline_df is None:
        timeline_df = track_proposal_through_time(proposal_id, utc_doc_reg_df)

    if timeline_df.empty:
        return f"# No mentions found for proposal {proposal_id}"

    context = analyze_proposal_context(timeline_df)

    # Sanitize proposal_title and proposer to remove newlines
    def sanitize_markdown_cell(val):
        if not isinstance(val, str):
            return val
        return val.replace("\n", " ").replace("\r", " ").strip()

    proposal_title = sanitize_markdown_cell(proposal_title)
    proposer = sanitize_markdown_cell(proposer)

    report = [f"# Timeline Report for Proposal {proposal_id}"]
    report.append(f"\n## {proposal_title}")
    report.append(f"\n**Proposer(s):** {proposer}")

    report.append(
        f"\nFound in **{context['doc_count']} documents** from {context['date_range'][0]} to {context['date_range'][1]}"
    )

    report.append("\n## Key Contributors")
    contributors = sorted(list(context["people_involved"]))
    report.append(", ".join(contributors[:30]) if contributors else "None identified")

    report.append("\n## Organizations Involved")
    entities = sorted(list(context["entities_involved"]))
    report.append(", ".join(entities[:30]) if entities else "None identified")

    report.append("\n## Emoji Characters Discussed")
    emoji_chars = list(context["emoji_mentioned"])
    report.append("".join(emoji_chars[:30]) if emoji_chars else "None identified")

    report.append("\n## Unicode Points Referenced")
    unicode_pts = sorted(list(context["unicode_points"]))
    report.append(", ".join(unicode_pts[:30]) if unicode_pts else "None identified")

    report.append("\n## Document Timeline")
    for idx, row in timeline_df.iterrows():
        date_str = (
            row["date"].strftime("%Y-%m-%d")
            if pd.notnull(row["date"])
            else "Unknown date"
        )
        report.append(f"\n### {date_str} | {row['doc_num']} | {row['reference_type']}")
        report.append(f"**Subject:** {row['subject']}")
        if "source" in row and pd.notnull(row["source"]):
            report.append(f"**Source:** {row['source']}")
        if "document_classification" in row and pd.notnull(
            row["document_classification"]
        ):
            doc_type = row["document_classification"]
            if isinstance(doc_type, dict):
                doc_type_str = ", ".join(
                    [f"{k}: {', '.join(v)}" for k, v in doc_type.items()]
                )
                report.append(f"**Document Type:** {doc_type_str}")
            else:
                report.append(f"**Document Type:** {doc_type}")
        if (
            "summary" in row
            and isinstance(row["summary"], str)
            and len(row["summary"]) > 0
        ):
            summary = (
                row["summary"][:500] + "..."
                if len(row["summary"]) > 500
                else row["summary"]
            )
            report.append(f"\n**Summary:** {summary}")
        if (
            "emoji_references" in row
            and isinstance(row["emoji_references"], list)
            and row["emoji_references"]
        ):
            report.append(
                f"\n**Emoji References:** {', '.join(row['emoji_references'])}"
            )

    return "\n".join(report)


def visualize_proposal_timeline(proposal_id, timeline_df, output_dir=None):
    """
    Create a left-to-right flowchart (timeline) of where a proposal appears in the UTC document register.
    Each node is a document (doc_num), annotated with date, subject, and reference type.

    Args:
        proposal_id: The document number of the proposal
        timeline_df: DataFrame containing the proposal's timeline
        output_dir: Directory to save the visualization (default: rejected_proposal_reports/timelines)

    Returns:
        Path to the generated visualization file
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import os
    import pandas as pd

    if output_dir is None:
        output_dir = os.path.join(base_path, "rejected_proposal_reports", "timelines")
    os.makedirs(output_dir, exist_ok=True)

    G = nx.DiGraph()
    labels = {}
    node_colors = []
    node_sizes = []

    # Define fixed color scheme for main document categories (consistent across visualizations)
    color_map = {
        "Proposals": "#E63946",  # Red
        "Meeting Documents": "#457B9D",  # Blue
        "Public Review & Feedback": "#2A9D8F",  # Teal
        "Liaison & External": "#F4A261",  # Orange
        "Administrative & Miscellaneous": "#A8DADC",  # Light blue
        "Standards & Specifications": "#90BE6D",  # Green
        "Reference": "#CCCCCC",  # Gray
    }
    # For any new/unexpected category, assign a color from a backup palette
    backup_colors = ["#FFB300", "#8E44AD", "#34495E", "#00B894", "#6C3483", "#B2BABB"]
    backup_idx = 0

    # Sort timeline by date
    timeline_df = timeline_df.copy()
    if "date" in timeline_df.columns:
        timeline_df = timeline_df.sort_values("date")

    # Process each document in the timeline
    for idx, row in timeline_df.iterrows():
        doc_num = row["doc_num"]
        date = row["date"] if "date" in row and pd.notnull(row["date"]) else None
        subject = row["subject"] if "subject" in row else ""
        ref_type = row["reference_type"] if "reference_type" in row else "Reference"

        # Use first key in document_classification dict for color assignment
        doc_class = row.get("document_classification", None)
        if isinstance(doc_class, dict) and len(doc_class) > 0:
            main_category = list(doc_class.keys())[0]
        elif isinstance(doc_class, list) and len(doc_class) > 0:
            main_category = doc_class[0]
        elif isinstance(doc_class, str) and doc_class:
            main_category = doc_class
        else:
            doc_type_val = row.get("doc_type", None)
            if isinstance(doc_type_val, list) and len(doc_type_val) > 0:
                main_category = doc_type_val[0]
            elif isinstance(doc_type_val, str) and doc_type_val:
                main_category = doc_type_val
            else:
                main_category = "Reference"

        # Assign color for main_category, using backup if needed
        if main_category not in color_map:
            color_map[main_category] = backup_colors[backup_idx % len(backup_colors)]
            backup_idx += 1

        # Create node label with metadata
        date_str = date.strftime("%Y-%m-%d") if date else "No date"
        label = f"{doc_num}\n{date_str}\n{main_category}\n{subject[:30]+'...' if len(subject) > 30 else subject}"

        # Add node to graph
        G.add_node(doc_num)
        labels[doc_num] = label
        node_colors.append(color_map.get(main_category, "#CCCCCC"))

        # Adjust node size based on importance
        if ref_type == "Original Proposal":
            node_sizes.append(2000)  # Larger size for the proposal
        elif main_category:
            node_sizes.append(1600)
        else:
            node_sizes.append(1400)

    # Add edges in timeline order
    doc_nums = list(timeline_df["doc_num"])
    for i in range(len(doc_nums) - 1):
        G.add_edge(doc_nums[i], doc_nums[i + 1])

    # Layout: left-to-right (horizontal timeline)
    pos = {doc: (i, 0) for i, doc in enumerate(doc_nums)}

    # Create visualization
    plt.figure(figsize=(max(12, len(doc_nums) * 3), 6))

    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors="black",
        linewidths=1,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        width=2,
        edge_color="#555555",
        alpha=0.7,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        font_size=9,
        font_color="black",
        font_weight="bold",
        verticalalignment="top",
    )

    # Add title and legend
    proposal_title = "Unknown"
    for idx, row in timeline_df.iterrows():
        if row["reference_type"] == "Original Proposal":
            proposal_title = row.get("subject", "Unknown Proposal")
            break

    plt.title(
        f"Timeline for Emoji Proposal: {proposal_id}\n{proposal_title}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add legend for all main categories used
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for label, color in color_map.items()
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.axis("off")
    plt.subplots_adjust(bottom=0.25)  # Add space for the legend

    # Save visualization
    safe_filename = re.sub(r'[\\/*?:"<>|]', "_", proposal_id)
    png_path = os.path.join(output_dir, f"{safe_filename}_timeline.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    return png_path


def generate_index_report(all_timelines, proposal_stats, timeline_paths=None):
    """
    Generate an index report in Markdown that links to all individual proposal reports

    Args:
        all_timelines: Dictionary of proposal timelines
        proposal_stats: List of proposal statistics
        timeline_paths: Dictionary mapping proposal_id to its timeline visualization path

    Returns:
        Markdown content for the index report
    """
    stats_df = pd.DataFrame(proposal_stats)
    stats_df = stats_df.sort_values("reference_count", ascending=False)

    # Start building the index markdown
    md_content = [
        "# Emoji Proposal Analysis Report",
        f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "\n## Overview",
        f"\nThis report analyzes **{len(all_timelines)}** emoji proposals from the Unicode Technical Committee (UTC) document registry.",
        "The analysis tracks how each proposal is referenced in the UTC document collection over time.",
        "\n## Timeline Visualizations",
        "\nFor each proposal, a timeline visualization has been created showing how the proposal",
        "flows through different UTC documents over time, with contextual metadata included.",
    ]

    # Highlight a few example visualizations if available
    if timeline_paths:
        top_proposals = stats_df.head(3)["proposal_id"].tolist()
        example_timelines = []

        for p_id in top_proposals:
            if p_id in timeline_paths and timeline_paths[p_id]:
                rel_path = os.path.relpath(
                    timeline_paths[p_id],
                    os.path.join(base_path, "rejected_proposal_reports"),
                )
                safe_filename = re.sub(r'[\\/*?:"<>|]', "_", p_id)
                example_timelines.append(
                    f"\n### {p_id} Timeline\n\n[![{p_id} Timeline]({rel_path})]({safe_filename}_report.md)"
                )

        if example_timelines:
            md_content.append("\n## Example Proposal Timelines")
            md_content.extend(example_timelines)

    # Add most referenced proposals section
    md_content.append("\n## Most Referenced Proposals")
    md_content.append(
        "\nThe following proposals have been referenced most frequently in UTC documents:"
    )
    md_content.append(
        "\n| Proposal ID | Title | Proposer | References | Contributors | Emojis |"
    )
    md_content.append(
        "| ----------- | ----- | -------- | ---------- | ------------ | ------ |"
    )

    for _, row in stats_df.head(10).iterrows():
        proposal_id = sanitize_markdown_cell(row["proposal_id"])
        proposal_title = sanitize_markdown_cell(row["proposal_title"])
        proposer = sanitize_markdown_cell(row["proposer"])
        safe_filename = re.sub(r'[\\/*?:"<>|]', "_", proposal_id)
        md_content.append(
            f"| [{proposal_id}]({safe_filename}_report.md) | {proposal_title} | {proposer} | {row['reference_count']} | {row['contributor_count']} | {row['emoji_count']} |"
        )

    # Add complete list of all proposals
    md_content.append("\n## All Analyzed Proposals")
    md_content.append("\n| Proposal ID | Title | Proposer | References | Date Range |")
    md_content.append("| ----------- | ----- | -------- | ---------- | ---------- |")

    for _, row in stats_df.iterrows():
        proposal_id = sanitize_markdown_cell(row["proposal_id"])
        proposal_title = sanitize_markdown_cell(row["proposal_title"])
        proposer = sanitize_markdown_cell(row["proposer"])
        safe_filename = re.sub(r'[\\/*?:"<>|]', "_", proposal_id)
        md_content.append(
            f"| [{proposal_id}]({safe_filename}_report.md) | {proposal_title} | {proposer} | {row['reference_count']} | {row['date_range']} |"
        )
    # ...existing code...
    return "\n".join(md_content)


def generate_all_reports():
    """Generate all proposal analysis reports in Markdown format with visualizations"""
    print("Starting emoji proposal analysis and report generation...")

    # Create output directory for reports
    reports_dir = os.path.join(base_path, "rejected_proposal_reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Analyze all emoji proposals
    all_timelines = analyze_all_emoji_proposals()

    if not all_timelines:
        print("No proposal timelines were generated. Check input data.")
        return

    # Collect stats for index report
    proposal_stats = []  # Identify proposals suitable for timeline visualization
    active_proposals = [
        p_id for p_id, timeline in all_timelines.items() if len(timeline) >= 3
    ]

    # Create directory for timeline visualizations
    timelines_dir = os.path.join(reports_dir, "timelines")
    os.makedirs(timelines_dir, exist_ok=True)

    # We'll store paths to all timeline visualizations
    timeline_paths = {}

    # Process each proposal
    for proposal_id, timeline in tqdm(
        all_timelines.items(), desc="Generating visualizations and reports"
    ):
        print(f"Generating report for proposal: {proposal_id}")

        # Find the proposal title using correct column names
        proposal_title = "Unknown"
        proposer = "Unknown"

        # Use correct column names according to data dictionary
        title_col = (
            "proposal_title"
            if "proposal_title" in emoji_proposal_df.columns
            else "Proposal Title"
        )
        doc_num_col = (
            "doc_num" if "doc_num" in emoji_proposal_df.columns else "Document Number"
        )
        proposer_col = (
            "proposer" if "proposer" in emoji_proposal_df.columns else "Proposer(s)"
        )

        if doc_num_col in emoji_proposal_df.columns:
            matches = emoji_proposal_df[
                emoji_proposal_df[doc_num_col].apply(
                    lambda x: (
                        normalize_doc_num(x) == proposal_id
                        if isinstance(x, str)
                        else False
                    )
                )
            ]

            if not matches.empty:
                if title_col in matches.columns:
                    proposal_title = matches.iloc[0][title_col]
                if proposer_col in matches.columns:
                    proposer = matches.iloc[0][
                        proposer_col
                    ]  # Collect stats for this proposal if it has a timeline
        if not timeline.empty:
            context = analyze_proposal_context(timeline)

            stats = {
                "proposal_id": proposal_id,
                "proposal_title": proposal_title,
                "proposer": proposer,
                "reference_count": len(timeline) - 1,  # Subtract the original document
                "date_range": f"{context['date_range'][0]} to {context['date_range'][1]}",
                "contributor_count": len(context["people_involved"]),
                "entity_count": len(context["entities_involved"]),
                "emoji_count": len(context["emoji_mentioned"]),
                "top_contributors": ", ".join(list(context["people_involved"])[:3]),
            }
            proposal_stats.append(stats)

            # Generate a timeline visualization for this proposal
            if (
                len(timeline) >= 2
            ):  # Only create visualization if there are at least 2 documents
                print(f"Creating timeline visualization for {proposal_id}")
                timeline_path = visualize_proposal_timeline(
                    proposal_id, timeline, output_dir=timelines_dir
                )
                timeline_paths[proposal_id] = timeline_path
            else:
                timeline_paths[proposal_id] = None

            # Generate report with timeline visualization reference
            report = generate_proposal_timeline_report(
                proposal_id, timeline, proposal_title, proposer
            )

            # Add timeline visualization reference to the report
            if proposal_id in timeline_paths and timeline_paths[proposal_id]:
                timeline_rel_path = os.path.relpath(
                    timeline_paths[proposal_id], reports_dir
                )
                report += f"\n\n## Proposal Timeline\n\n![Proposal Timeline]({timeline_rel_path})\n\n"
                report += "*This visualization shows the chronological journey of this proposal through the UTC documents.*\n"

            # Save report to file
            safe_filename = re.sub(r'[\\/*?:"<>|]', "_", proposal_id)
            report_path = os.path.join(reports_dir, f"{safe_filename}_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)  # Generate and save the index report
    index_md = generate_index_report(all_timelines, proposal_stats, timeline_paths)
    index_path = os.path.join(reports_dir, "index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_md)

    print(f"\nAll reports generated successfully. Main index: {index_path}")
    return index_path


def main():
    # Generate all Markdown reports with visualizations
    index_path = generate_all_reports()

    if index_path:
        print(f"✅ Analysis complete! View the main report at: {index_path}")
    else:
        print("❌ Analysis failed. Check the error messages above.")

    return index_path


def sanitize_markdown_cell(val):
    if not isinstance(val, str):
        return val
    return val.replace("\n", " ").replace("\r", " ").strip()


if __name__ == "__main__":
    main()
