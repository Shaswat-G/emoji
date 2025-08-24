# -----------------------------------------------------------------------------
# Script: analyze_accepted_proposal_v2.py
# Summary: Re-analyzes accepted emoji proposals using the same methodology as rejected
#          proposals - finds acceptance date by looking for the last reference in meeting
#          documents rather than using emoji release dates, then recalculates processing
#          times and nature classification.
# Inputs:  single_concept_accepted_proposals.xlsx (from analyze_accepted_proposal_dataset.py),
#          utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx
# Outputs: single_concept_accepted_proposals_v2.xlsx (with updated acceptance_date,
#          processing_time, and nature columns)
# Context: Alternative analysis method for accepted proposals using meeting document
#          references to determine acceptance timeline, enabling comparison with
#          rejected proposal processing patterns.
# -----------------------------------------------------------------------------

import os

import pandas as pd

from utc_proposal_triangulator import safe_literal_eval

EXCEPTION_LIMIT_DAYS = 1000  # Customize the exception threshold for processing time


META_COLS = [
    "subject",
    "source",
    "date",
    "summary",
    "description",
    "doc_type",
    "extracted_doc_refs",
]


def references_proposal(proposal_id, refs_list):
    """Check if a proposal ID is referenced in a list of document references."""
    if not isinstance(refs_list, list):
        return False

    # Normalize each reference and check for a match
    for ref in refs_list:
        if ref == proposal_id:
            return True
    return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load datasets
    accepted = pd.read_excel(
        os.path.join(base_dir, "single_concept_accepted_proposals.xlsx")
    )
    utc_doc_reg = pd.read_excel(
        os.path.join(
            base_dir,
            "utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx",
        )
    )
    utc_doc_reg = (
        utc_doc_reg[["doc_num"] + META_COLS]
        .drop_duplicates(subset="doc_num")
        .reset_index(drop=True)
    )
    utc_doc_reg["doc_type"] = utc_doc_reg["doc_type"].apply(safe_literal_eval)
    utc_doc_reg["extracted_doc_refs"] = utc_doc_reg["extracted_doc_refs"].apply(
        safe_literal_eval
    )

    # Create new columns for v2 analysis (preserve original data)
    accepted["acceptance_date_v2"] = None
    accepted["processing_time_v2"] = None
    accepted["nature_v2"] = None

    print(f"Analyzing {len(accepted)} accepted proposals...")

    for idx, row in accepted.iterrows():
        proposal_id = row["proposal_doc_num"]
        proposal_date = row["date"] if "date" in row else None

        if idx % 10 == 0:
            print(f"Processing proposal {idx + 1}/{len(accepted)}: {proposal_id}")

        # Find all documents that reference this proposal
        timeline = utc_doc_reg[
            utc_doc_reg["extracted_doc_refs"].apply(
                lambda refs: references_proposal(proposal_id, refs)
            )
        ].copy()

        if timeline.empty:
            accepted.at[idx, "acceptance_date_v2"] = None
            accepted.at[idx, "processing_time_v2"] = None
            accepted.at[idx, "nature_v2"] = "exception"
            continue

        # Find last reference in a Meeting Document (same logic as rejected proposals)
        meeting_mask = timeline["doc_type"].apply(
            lambda dt: (isinstance(dt, dict) and "Meeting Documents" in dt.keys())
            or (isinstance(dt, list) and "Meeting Documents" in dt)
            or (isinstance(dt, str) and dt == "Meeting Documents")
        )
        meeting_dates = timeline.loc[meeting_mask, "date"].dropna()

        if len(meeting_dates) > 0:
            acceptance_date = meeting_dates.max()
        else:
            # Fallback to any document reference if no meeting documents found
            acceptance_date = timeline["date"].dropna().max()

        # Set acceptance_date_v2
        if pd.notnull(acceptance_date):
            accepted.at[idx, "acceptance_date_v2"] = acceptance_date
        else:
            accepted.at[idx, "acceptance_date_v2"] = None

        # Calculate processing_time_v2 (days between proposal_date and acceptance_date_v2)
        if (
            proposal_date is not None
            and pd.notnull(proposal_date)
            and pd.notnull(acceptance_date)
        ):
            days = (
                pd.to_datetime(acceptance_date) - pd.to_datetime(proposal_date)
            ).days
            accepted.at[idx, "processing_time_v2"] = days
            if days <= 0 or days > EXCEPTION_LIMIT_DAYS:
                accepted.at[idx, "nature_v2"] = "exception"
            else:
                accepted.at[idx, "nature_v2"] = "normal"
        else:
            accepted.at[idx, "processing_time_v2"] = None
            accepted.at[idx, "nature_v2"] = "exception"

    # Generate summary statistics
    print("\n" + "=" * 60)
    print("ACCEPTANCE DATE ANALYSIS COMPARISON")
    print("=" * 60)

    # Original method statistics
    original_normal = accepted[accepted["nature"] == "normal"]
    original_exception = accepted[accepted["nature"] == "exception"]

    print("\nORIGINAL METHOD (emoji release dates):")
    print(f"  Normal proposals: {len(original_normal)}")
    print(f"  Exception proposals: {len(original_exception)}")
    if len(original_normal) > 0:
        print(
            f"  Average processing time (normal): {original_normal['processing_time'].mean():.1f} days"
        )
        print(
            f"  Median processing time (normal): {original_normal['processing_time'].median():.1f} days"
        )

    # V2 method statistics
    v2_normal = accepted[accepted["nature_v2"] == "normal"]
    v2_exception = accepted[accepted["nature_v2"] == "exception"]

    print("\nV2 METHOD (meeting document references):")
    print(f"  Normal proposals: {len(v2_normal)}")
    print(f"  Exception proposals: {len(v2_exception)}")
    if len(v2_normal) > 0:
        print(
            f"  Average processing time (normal): {v2_normal['processing_time_v2'].mean():.1f} days"
        )
        print(
            f"  Median processing time (normal): {v2_normal['processing_time_v2'].median():.1f} days"
        )

    # Comparison
    print("\nCOMPARISON:")
    both_normal = accepted[
        (accepted["nature"] == "normal") & (accepted["nature_v2"] == "normal")
    ]
    print(f"  Proposals normal in both methods: {len(both_normal)}")

    changed_to_normal = accepted[
        (accepted["nature"] == "exception") & (accepted["nature_v2"] == "normal")
    ]
    changed_to_exception = accepted[
        (accepted["nature"] == "normal") & (accepted["nature_v2"] == "exception")
    ]
    print(f"  Changed from exception to normal: {len(changed_to_normal)}")
    print(f"  Changed from normal to exception: {len(changed_to_exception)}")

    # Save the result
    output_path = os.path.join(base_dir, "single_concept_accepted_proposals_v2.xlsx")
    accepted.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
