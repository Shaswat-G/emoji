import os
import pandas as pd
import json
import ast
import re
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


"""
Goal: To track emoji proposals through time and quantify their time-flow characteristics. For example we would like to measure the time it takes from when the proposal document appears in the UTC docregistry as a row (not a reference) to its last reference in the registry.
We would like to know its velocity of flow through the UTC docregistry and how many times it has been referenced in other documents. We want to capture attention dynamics like who (people) are paying attention to the proposal and how many times it has been referenced in other documents. 
We have to enrich this summarizing data for each proposal with creatively defined relevant metrics with people, entities, emoji counts, etc.
We want to understand how and what are the things that Unicode and UTC members are paying attention to and how does that attention change over time.
We also want to understand year-wise what has been the velocity of processing proposals, time taken, people involved, emojis involved, etc.
DataFrames:
1. emoji_proposal_df: Contains emoji proposal data. Contains columns like 'doc_num', 'proposal_title', 'proposer', 'count' (counts #emojis in proposal).
2. utc_doc_reg_df: Contains UTC document registry - collection all documents whether proposal or not with columns:
- doc_num : document number
- subject : subject of the document
- source : Person who submitted the document
- date : date of the document
- emoji_relevance : relevance of the document to emoji
- extracted_doc_refs : Pythonic list with other doc-references extracted from the document
- emoji_chars : emoji characters found in the document
- unicode_points : unicode points for the emoji characters
- is_emoji_relevant : whether the document is relevant to emoji
- people : Pythonic list of people mentioned in the document
- emoji_references	['Emoji symbols', 'U+1F600', 'Emoji ad-hoc meeting report', 'Emoji: Review of FPDAM8', 'DoCoMo Input on Emoji', 'KDDI Input on Emoji', 'Willcom Input on Emoji']
- entities : Pythonic list of entities mentioned in the document
- summary : Summary of the document.
- description : Description of the document
3. email_match_df: Contains email matches that have matched with corresponding proposals from emoji_proposal_df, Contains columns:
- proposal_doc_num - comes from the emoji_proposal_df - this is the document number.
- proposal_for_1 - comes from the emoji_proposal_df - this is the primary keyword used for matching.
- proposal_for_2 - comes from the emoji_proposal_df - this is the secondary keyword used for matching.
- match_type - describes how the email matches the proposal - which fields triggered the match.
- confidence_score - a score indicating the strength of the match - basically the number of fields that matched.
- year - the year the email was sent.
- month - the month the email was sent.
- date - the date the email was sent.
- from_email - the email address of the sender.
- from_name - the name of the sender.
- subject - the subject line of the email.
- people - Pythonic list of people in that email
"""

base_path = os.getcwd()
emoji_proposal_path = os.path.join(base_path, "emoji_proposal_table.csv")


emoji_proposal_df = pd.read_csv(emoji_proposal_path, dtype=str)
utc_doc_reg_path = os.path.join(base_path, "utc_register_with_llm_extraction.xlsx")
utc_email_path = os.path.join(base_path, "emoji_proposal_email_matches.csv")


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
    "doc_type",
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


try:
    email_match_df = pd.read_csv(utc_email_path)

    # Then apply converters manually to specific columns after loading
    for col in columns_to_eval:
        if col in email_match_df.columns:
            email_match_df[col] = email_match_df[col].apply(safe_literal_eval)
except Exception as e:
    print(f"Error loading or processing the CSV file: {e}")


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
    Returns a dictionary with contextual analysis.
    """
    context = {
        "doc_count": len(timeline_df),
        "date_range": (
            (timeline_df["date"].min(), timeline_df["date"].max())
            if not timeline_df.empty else (None, None)
        ),
        "people_involved": set(),
        "entities_involved": set(),
        "emoji_mentioned": set(),
        "unicode_points": set(),
    }
    for col, key in [("people", "people_involved"), ("entities", "entities_involved"), ("emoji_chars", "emoji_mentioned"), ("unicode_points", "unicode_points")]:
        if col in timeline_df.columns:
            for item in timeline_df[col]:
                if isinstance(item, list):
                    context[key].update(item)
    return context


def summarize_all_proposals():
    """
    For each proposal, compute and save summary metrics as CSV and Excel.
    """
    summary_rows = []
    yearwise_rows = []
    for _, row in emoji_proposal_df.iterrows():
        proposal_id = normalize_doc_num(row["doc_num"])
        proposal_title = row.get("proposal_title", "")
        proposer = row.get("proposer", "")
        emoji_count = int(row.get("count", 0)) if str(row.get("count", "")).isdigit() else 0

        # Track proposal in UTC doc registry
        timeline = track_proposal_through_time(proposal_id, utc_doc_reg_df)
        context = analyze_proposal_context(timeline) if not timeline.empty else {}

        # Velocity: time from first to last reference (in days)
        date_min, date_max = context.get("date_range", (None, None))
        try:
            velocity_days = (pd.to_datetime(date_max) - pd.to_datetime(date_min)).days if date_min and date_max else None
        except Exception:
            velocity_days = None

        # Reference count (excluding original)
        reference_count = context.get("doc_count", 0) - 1 if context.get("doc_count", 0) > 0 else 0

        # People/entities/emoji
        people = context.get("people_involved", set())
        entities = context.get("entities_involved", set())
        emoji_mentioned = context.get("emoji_mentioned", set())

        # Email attention: count of matching emails, unique people
        email_matches = email_match_df[email_match_df["proposal_doc_num"].apply(lambda x: normalize_doc_num(x) == proposal_id if isinstance(x, str) else False)]
        email_count = len(email_matches)
        email_people = set()
        if "people" in email_matches.columns:
            for ppl in email_matches["people"]:
                if isinstance(ppl, list):
                    email_people.update(ppl)
        # Year-wise stats
        if not timeline.empty and "date" in timeline.columns:
            timeline["year"] = pd.to_datetime(timeline["date"], errors="coerce").dt.year
            for year, group in timeline.groupby("year"):
                if pd.isnull(year):
                    continue
                yearwise_rows.append({
                    "proposal_id": proposal_id,
                    "year": int(year),
                    "reference_count": len(group) - 1 if len(group) > 0 else 0,
                    "people_count": len(set().union(*[set(x) if isinstance(x, list) else set() for x in group.get("people", [])])),
                    "entities_count": len(set().union(*[set(x) if isinstance(x, list) else set() for x in group.get("entities", [])])),
                    "emoji_count": len(set().union(*[set(x) if isinstance(x, list) else set() for x in group.get("emoji_chars", [])])),
                })
        summary_rows.append({
            "proposal_id": proposal_id,
            "proposal_title": proposal_title,
            "proposer": proposer,
            "emoji_count": emoji_count,
            "velocity_days": velocity_days,
            "reference_count": reference_count,
            "people_count": len(people),
            "entities_count": len(entities),
            "emoji_mentioned_count": len(emoji_mentioned),
            "email_count": email_count,
            "email_people_count": len(email_people),
            "date_first": str(date_min) if date_min else "",
            "date_last": str(date_max) if date_max else "",
        })
    # Output summary
    summary_df = pd.DataFrame(summary_rows)
    yearwise_df = pd.DataFrame(yearwise_rows)
    summary_csv = os.path.join(base_path, "proposal_summary.csv")
    yearwise_csv = os.path.join(base_path, "proposal_yearwise_summary.csv")
    summary_xlsx = os.path.join(base_path, "proposal_summary.xlsx")
    summary_df.to_csv(summary_csv, index=False)
    yearwise_df.to_csv(yearwise_csv, index=False)
    with pd.ExcelWriter(summary_xlsx) as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        yearwise_df.to_excel(writer, index=False, sheet_name="Yearwise")
    print(f"Saved summary to {summary_csv}, {yearwise_csv}, {summary_xlsx}")


def compute_proposal_flow_velocity_metrics():
    """
    For each proposal, compute:
    - Time to last reference (days)
    - Reference count (excluding original)
    - Velocity (references per year/month)
    - Dormancy & revival (max gap between references, number of dormant years/months)
    - First/last reference dates
    Output as CSV and Excel.
    """
    results = []
    for _, row in emoji_proposal_df.iterrows():
        proposal_id = normalize_doc_num(row["doc_num"])
        proposal_title = row.get("proposal_title", "")
        proposer = row.get("proposer", "")
        # Track proposal in UTC doc registry
        timeline = track_proposal_through_time(proposal_id, utc_doc_reg_df)
        if timeline.empty or "date" not in timeline.columns:
            results.append({
                "proposal_id": proposal_id,
                "proposal_title": proposal_title,
                "proposer": proposer,
                "reference_count": 0,
                "time_to_last_reference_days": None,
                "velocity_per_year": None,
                "velocity_per_month": None,
                "first_reference_date": None,
                "last_reference_date": None,
                "max_dormancy_days": None,
                "num_dormant_years": None,
                "num_dormant_months": None,
            })
            continue
        # Sort and clean dates
        timeline = timeline.copy()
        timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
        timeline = timeline.sort_values("date")
        valid_dates = timeline["date"].dropna().tolist()
        if not valid_dates:
            first_date = last_date = None
        else:
            first_date = valid_dates[0]
            last_date = valid_dates[-1]
        # Reference count (excluding original)
        reference_count = len(timeline) - 1 if len(timeline) > 0 else 0
        # Time to last reference (days)
        time_to_last_reference_days = (last_date - first_date).days if first_date and last_date else None
        # Velocity
        years_span = ((last_date - first_date).days / 365.25) if first_date and last_date and (last_date > first_date) else None
        months_span = ((last_date - first_date).days / 30.44) if first_date and last_date and (last_date > first_date) else None
        velocity_per_year = reference_count / years_span if years_span and years_span > 0 else None
        velocity_per_month = reference_count / months_span if months_span and months_span > 0 else None
        # Dormancy & revival
        max_dormancy_days = None
        num_dormant_years = None
        num_dormant_months = None
        if len(valid_dates) > 1:
            gaps = [(valid_dates[i+1] - valid_dates[i]).days for i in range(len(valid_dates)-1)]
            max_dormancy_days = max(gaps) if gaps else None
            # Dormant years/months: years/months with no references
            years = [d.year for d in valid_dates]
            months = [(d.year, d.month) for d in valid_dates]
            year_range = range(min(years), max(years)+1) if years else []
            month_range = pd.period_range(min(valid_dates), max(valid_dates), freq='M') if months else []
            num_dormant_years = len(set(year_range) - set(years)) if years else None
            months_set = set((p.year, p.month) for p in month_range)
            num_dormant_months = len(months_set - set(months)) if months else None
        results.append({
            "proposal_id": proposal_id,
            "proposal_title": proposal_title,
            "proposer": proposer,
            "reference_count": reference_count,
            "time_to_last_reference_days": time_to_last_reference_days,
            "velocity_per_year": velocity_per_year,
            "velocity_per_month": velocity_per_month,
            "first_reference_date": str(first_date.date()) if first_date else None,
            "last_reference_date": str(last_date.date()) if last_date else None,
            "max_dormancy_days": max_dormancy_days,
            "num_dormant_years": num_dormant_years,
            "num_dormant_months": num_dormant_months,
        })
    df = pd.DataFrame(results)
    out_csv = os.path.join(base_path, "proposal_flow_velocity_metrics.csv")
    out_xlsx = os.path.join(base_path, "proposal_flow_velocity_metrics.xlsx")
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)
    print(f"Saved proposal flow & velocity metrics to {out_csv} and {out_xlsx}")


def compute_attention_dynamics_social_metrics():
    """
    For each proposal, compute:
    - Unique people involved (count, list)
    - Unique entities involved (count, list)
    - Attention span: unique people per year
    - Key contributors: top 3-5 people by frequency
    - Attention shifts: new people/entities per year (compared to previous years)
    - Attention drift: change in people/entities/emoji from first to last year
    - Processing speed trends: velocity per year
    - Email vs Doc attention: correlation between email count and reference count
    Output as CSV and Excel.
    """
    results = []
    all_velocity_per_year = []
    all_email_vs_doc = []
    for _, row in emoji_proposal_df.iterrows():
        proposal_id = normalize_doc_num(row["doc_num"])
        proposal_title = row.get("proposal_title", "")
        proposer = row.get("proposer", "")
        timeline = track_proposal_through_time(proposal_id, utc_doc_reg_df)
        email_matches = email_match_df[email_match_df["proposal_doc_num"].apply(lambda x: normalize_doc_num(x) == proposal_id if isinstance(x, str) else False)]
        if timeline.empty or "date" not in timeline.columns:
            results.append({
                "proposal_id": proposal_id,
                "proposal_title": proposal_title,
                "proposer": proposer,
                "unique_people_count": 0,
                "unique_people_list": "",
                "unique_entities_count": 0,
                "unique_entities_list": "",
                "key_contributors": "",
                "attention_span": "",
                "attention_shifts": "",
                "attention_drift": "",
                "velocity_per_year": "",
                "email_vs_doc_attention": "",
            })
            continue
        # Clean and sort dates
        timeline = timeline.copy()
        timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
        timeline = timeline.sort_values("date")
        # Unique people/entities/emoji
        all_people = []
        all_entities = []
        all_emoji = []
        if "people" in timeline.columns:
            for ppl in timeline["people"]:
                if isinstance(ppl, list):
                    all_people.extend(ppl)
        if "entities" in timeline.columns:
            for ent in timeline["entities"]:
                if isinstance(ent, list):
                    all_entities.extend(ent)
        if "emoji_chars" in timeline.columns:
            for emj in timeline["emoji_chars"]:
                if isinstance(emj, list):
                    all_emoji.extend(emj)
        unique_people = sorted(set(all_people))
        unique_entities = sorted(set(all_entities))
        unique_emoji = sorted(set(all_emoji))
        # Key contributors (top 3-5 by frequency)
        from collections import Counter
        people_counter = Counter(all_people)
        key_contributors = [p for p, _ in people_counter.most_common(5)]
        # Attention span: unique people per year
        attention_span = {}
        attention_shifts = {}
        velocity_per_year = {}
        prev_people = set()
        prev_entities = set()
        prev_emoji = set()
        # For drift
        year_people = {}
        year_entities = {}
        year_emoji = {}
        if not timeline.empty and "date" in timeline.columns:
            timeline["year"] = pd.to_datetime(timeline["date"], errors="coerce").dt.year
            for year, group in timeline.groupby("year"):
                if pd.isnull(year):
                    continue
                y = int(year)
                y_people = set()
                y_entities = set()
                y_emoji = set()
                for ppl in group.get("people", []):
                    if isinstance(ppl, list):
                        y_people.update(ppl)
                for ent in group.get("entities", []):
                    if isinstance(ent, list):
                        y_entities.update(ent)
                for emj in group.get("emoji_chars", []):
                    if isinstance(emj, list):
                        y_emoji.update(emj)
                year_people[y] = y_people
                year_entities[y] = y_entities
                year_emoji[y] = y_emoji
                attention_span[y] = len(y_people)
                # Attention shifts: new people/entities this year
                new_people = y_people - prev_people
                new_entities = y_entities - prev_entities
                attention_shifts[y] = {
                    "new_people": sorted(new_people),
                    "new_entities": sorted(new_entities),
                }
                prev_people |= y_people
                prev_entities |= y_entities
                prev_emoji |= y_emoji
                # Velocity per year
                velocity_per_year[y] = len(group) - 1 if len(group) > 0 else 0
        # Attention drift: change in people/entities/emoji from first to last year
        drift = {}
        if year_people:
            first_year = min(year_people.keys())
            last_year = max(year_people.keys())
            drift["people_added"] = sorted(year_people[last_year] - year_people[first_year])
            drift["people_lost"] = sorted(year_people[first_year] - year_people[last_year])
            drift["entities_added"] = sorted(year_entities[last_year] - year_entities[first_year])
            drift["entities_lost"] = sorted(year_entities[first_year] - year_entities[last_year])
            drift["emoji_added"] = sorted(year_emoji[last_year] - year_emoji[first_year])
            drift["emoji_lost"] = sorted(year_emoji[first_year] - year_emoji[last_year])
        # Processing speed trends: velocity per year (already computed)
        # Email vs Doc attention: correlation between email count and reference count
        reference_count = len(timeline) - 1 if len(timeline) > 0 else 0
        email_count = len(email_matches)
        email_vs_doc_attention = None
        if reference_count > 0:
            email_vs_doc_attention = email_count / reference_count
        else:
            email_vs_doc_attention = None
        results.append({
            "proposal_id": proposal_id,
            "proposal_title": proposal_title,
            "proposer": proposer,
            "unique_people_count": len(unique_people),
            "unique_people_list": ", ".join(unique_people),
            "unique_entities_count": len(unique_entities),
            "unique_entities_list": ", ".join(unique_entities),
            "key_contributors": ", ".join(key_contributors),
            "attention_span": json.dumps(attention_span),
            "attention_shifts": json.dumps(attention_shifts),
            "attention_drift": json.dumps(drift),
            "velocity_per_year": json.dumps(velocity_per_year),
            "email_vs_doc_attention": email_vs_doc_attention,
        })
    df = pd.DataFrame(results)
    out_csv = os.path.join(base_path, "proposal_attention_dynamics_metrics.csv")
    out_xlsx = os.path.join(base_path, "proposal_attention_dynamics_metrics.xlsx")
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)
    print(f"Saved proposal attention dynamics & social metrics to {out_csv} and {out_xlsx}")


def compute_yearwise_temporal_analysis():
    """
    For each proposal, compute:
    - Yearly velocity: references, people, entities, emoji, emails per year
    - Processing time: time from proposal submission to last reference, and to last email
    - Yearly attention shifts: people/entities/emoji per year
    Output as CSV and Excel.
    """
    results = []
    for _, row in emoji_proposal_df.iterrows():
        proposal_id = normalize_doc_num(row["doc_num"])
        proposal_title = row.get("proposal_title", "")
        proposer = row.get("proposer", "")
        timeline = track_proposal_through_time(proposal_id, utc_doc_reg_df)
        email_matches = email_match_df[email_match_df["proposal_doc_num"].apply(lambda x: normalize_doc_num(x) == proposal_id if isinstance(x, str) else False)]
        # Prepare yearwise doc timeline
        if not timeline.empty and "date" in timeline.columns:
            timeline = timeline.copy()
            timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
            timeline = timeline.sort_values("date")
            timeline["year"] = timeline["date"].dt.year
        else:
            timeline = pd.DataFrame(columns=["year", "people", "entities", "emoji_chars", "date"])
        # Prepare yearwise email timeline
        if not email_matches.empty and "date" in email_matches.columns:
            email_matches = email_matches.copy()
            email_matches["date"] = pd.to_datetime(email_matches["date"], errors="coerce")
            email_matches["year"] = email_matches["date"].dt.year
        else:
            email_matches = pd.DataFrame(columns=["year", "people", "entities", "emoji_chars", "date"])
        # Get all years present in either
        years = set(timeline["year"].dropna().astype(int).tolist()) | set(email_matches["year"].dropna().astype(int).tolist())
        if not years:
            # No data, still output a row for this proposal
            results.append({
                "proposal_id": proposal_id,
                "proposal_title": proposal_title,
                "proposer": proposer,
                "year": None,
                "reference_count": 0,
                "people_count": 0,
                "entities_count": 0,
                "emoji_count": 0,
                "email_count": 0,
                "processing_time_to_last_reference_days": None,
                "processing_time_to_last_email_days": None,
                "people_list": "",
                "entities_list": "",
                "emoji_list": "",
            })
            continue
        # For processing time
        proposal_submission_date = None
        if not timeline.empty and "date" in timeline.columns:
            valid_dates = timeline["date"].dropna().tolist()
            if valid_dates:
                proposal_submission_date = valid_dates[0]
        last_reference_date = valid_dates[-1] if valid_dates else None
        last_email_date = None
        if not email_matches.empty and "date" in email_matches.columns:
            email_dates = email_matches["date"].dropna().tolist()
            if email_dates:
                last_email_date = email_dates[-1]
        processing_time_to_last_reference_days = (last_reference_date - proposal_submission_date).days if proposal_submission_date and last_reference_date else None
        processing_time_to_last_email_days = (last_email_date - proposal_submission_date).days if proposal_submission_date and last_email_date else None
        # Yearwise stats
        for year in sorted(years):
            # Doc stats
            doc_group = timeline[timeline["year"] == year] if not timeline.empty else pd.DataFrame()
            people = set()
            entities = set()
            emoji = set()
            if not doc_group.empty:
                for ppl in doc_group.get("people", []):
                    if isinstance(ppl, list):
                        people.update(ppl)
                for ent in doc_group.get("entities", []):
                    if isinstance(ent, list):
                        entities.update(ent)
                for emj in doc_group.get("emoji_chars", []):
                    if isinstance(emj, list):
                        emoji.update(emj)
            # Email stats
            email_group = email_matches[email_matches["year"] == year] if not email_matches.empty else pd.DataFrame()
            email_count = len(email_group) if not email_group.empty else 0
            results.append({
                "proposal_id": proposal_id,
                "proposal_title": proposal_title,
                "proposer": proposer,
                "year": year,
                "reference_count": len(doc_group) - 1 if not doc_group.empty else 0,
                "people_count": len(people),
                "entities_count": len(entities),
                "emoji_count": len(emoji),
                "email_count": email_count,
                "processing_time_to_last_reference_days": processing_time_to_last_reference_days,
                "processing_time_to_last_email_days": processing_time_to_last_email_days,
                "people_list": ", ".join(sorted(people)),
                "entities_list": ", ".join(sorted(entities)),
                "emoji_list": ", ".join(sorted(emoji)),
            })
    df = pd.DataFrame(results)
    out_csv = os.path.join(base_path, "proposal_yearwise_temporal_analysis.csv")
    out_xlsx = os.path.join(base_path, "proposal_yearwise_temporal_analysis.xlsx")
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)
    print(f"Saved proposal yearwise & temporal analysis to {out_csv} and {out_xlsx}")


if __name__ == "__main__":
    summarize_all_proposals()
    compute_proposal_flow_velocity_metrics()
    compute_attention_dynamics_social_metrics()
    compute_yearwise_temporal_analysis()
