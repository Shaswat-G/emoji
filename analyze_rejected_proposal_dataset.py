# -----------------------------------------------------------------------------
# Script: analyze_rejected_proposal_dataset.py
# Summary: For each rejected emoji proposal, finds the rejection date by analyzing UTC document references,
#          calculates the processing time (days between proposal and rejection), and classifies each proposal
#          as 'normal' or 'exception' based on customizable time thresholds.
# Inputs:  rejected_proposal_dataset.xlsx,
#          utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx
# Outputs: rejected_proposal_dataset.xlsx (with rejection_date, processing_time, and nature columns)
# Context: Part of the emoji proposal research pipeline, this script analyzes the timeline and outcome of
#          rejected proposals for further study of Unicode emoji standardization processes.
# -----------------------------------------------------------------------------

import pandas as pd
import os

EXCEPTION_LIMIT_DAYS = 500  # Customize the exception threshold for processing time
from utc_proposal_triangulator import safe_literal_eval


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
    rejected = pd.read_excel(os.path.join(base_dir, "rejected_proposal_dataset.xlsx"))
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

    # Prepare to store rejection dates, processing time, and nature
    rejected["rejection_date"] = None
    rejected["processing_time"] = None
    rejected["nature"] = None

    for idx, row in rejected.iterrows():
        proposal_id = row["doc_num"]
        proposal_date = row["date"] if "date" in row else None
        # Find all documents that reference this proposal
        timeline = utc_doc_reg[
            utc_doc_reg["extracted_doc_refs"].apply(
                lambda refs: references_proposal(proposal_id, refs)
            )
        ].copy()
        if timeline.empty:
            rejected.at[idx, "rejection_date"] = 0
            rejected.at[idx, "processing_time"] = None
            rejected.at[idx, "nature"] = "exception"
            continue
        # Find last reference in a Meeting Document
        meeting_mask = timeline["doc_type"].apply(
            lambda dt: (isinstance(dt, dict) and "Meeting Documents" in dt.keys())
            or (isinstance(dt, list) and "Meeting Documents" in dt)
            or (isinstance(dt, str) and dt == "Meeting Documents")
        )
        meeting_dates = timeline.loc[meeting_mask, "date"].dropna()
        if len(meeting_dates) > 0:
            rejection_date = meeting_dates.max()
        else:
            rejection_date = timeline["date"].dropna().max()
        # Set rejection_date as found (do not impute to zero)
        if pd.notnull(rejection_date):
            rejected.at[idx, "rejection_date"] = rejection_date
        else:
            rejected.at[idx, "rejection_date"] = None

        # Calculate processing_time (days between proposal_date and rejection_date)
        if (
            proposal_date is not None
            and pd.notnull(proposal_date)
            and pd.notnull(rejection_date)
        ):
            days = (pd.to_datetime(rejection_date) - pd.to_datetime(proposal_date)).days
            rejected.at[idx, "processing_time"] = days
            if days <= 0 or days > EXCEPTION_LIMIT_DAYS:
                rejected.at[idx, "nature"] = "exception"
            else:
                rejected.at[idx, "nature"] = "normal"
        else:
            rejected.at[idx, "processing_time"] = None
            rejected.at[idx, "nature"] = "exception"

    # Optionally, save the result
    rejected.to_excel(
        os.path.join(base_dir, "rejected_proposal_dataset.xlsx"),
        index=False,
    )


if __name__ == "__main__":
    main()
