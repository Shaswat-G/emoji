import pandas as pd
import numpy as np
import os
import re


unicode_docs_hierarchy_keywords_refined = {
    "Meeting Documents": {
        "Agendas": [
            "agenda", "meeting agenda", "discussion topics", "topics for discussion", "planned agenda"
        ],
        "Minutes": [
            "minutes", "mom", "m.o.m.", "notes from meeting", "meeting summary", "meeting record", "record of meeting"
        ],
        "Decisions & Action Items": [
            "decision", "resolution", "conclusion", "recommendation", "approved", "approve", 
            "reject", "rejection", "denial", "action item", "motion", "follow-up", "follow up", 
            "heads-up", "notice", "notification", "announcement", "bulletin", "advisory"
        ]
    },
    "Public Review & Feedback": {
        "Public Review Issues (PRI)": [
            "pri", "p.r.i.", "pri feedback", "p.r.i. feedback", "public review issue"
        ],
        "Clarification & Queries": [
            "clarification", "explain", "explanation", "clarify", "query", "queries", "question",
            "inquiry", "ask", "please clarify"
        ],
        "General Feedback & Correspondence": [
            "feedback", "feed-back", "feed back", "comments", "comment", "remarks", "observations", 
            "review", "response", "responses"
        ]
    },
    "Proposals": {
        "Character Encoding Proposals": [
            "proposal", "proposed", "prop.", "submission", "encode", "addition", "add", 
            "new character", "new script", "character proposal", "script proposal"
        ],
        "Technical & Process Proposals": [
            "change request", "change order", "addendum", "amendment", "extension", "revision", 
            "revised doc", "appendix", "modification", "process proposal", "technical proposal"
        ]
    },
    "Standards & Specifications": {
        "Unicode Standard & Annexes (UAX)": [
            "uax", "unicode standard annex", "unicode annex", "annex", "uax#"
        ],
        "Unicode Technical Standards (UTS)": [
            "uts", "unicode technical standard", "uts#"
        ],
        "Unicode Technical Reports (UTR)": [
            "utr", "unicode technical report", "utr#"
        ],
        "Unicode Technical Notes (UTN)": [
            "utn", "unicode technical note", "utn#"
        ],
        "Errata & Corrigenda": [
            "errata", "erratum", "corrigendum", "corrigenda", "correction"
        ]
    },
    "Liaison & External": {
        "ISO/IEC WG2 Documents & Ballots": [
            "iso", "wg2", "iec", "jtc1", "sc2", "ballot", "national body", "iso ballot", "iso comment"
        ],
        "Liaison Reports & Agreements": [
            "liaison", "mou", "memorandum of understanding", "agreement", "contract", 
            "service-level agreement", "sla", "liaison report"
        ]
    },
    "Administrative & Miscellaneous": {
        "Document Registers & Indexes": [
            "document register", "register", "index", "doc register"
        ],
        "Schedules & Planning": [
            "schedule", "planning", "timetable", "release plan", "roadmap", "calendar"
        ],
        "Memo/Circular": [
            "memo", "memorandum", "circular", "internal note", "bulletin"
        ]
    }
}

def match_keyword(subject: str, keyword: str) -> bool:
    """
    Uses a regex with negative lookbehind and lookahead to ensure that
    the keyword is not a part of a larger word.
    """
    # Construct the pattern to match whole words, accounting for phrases.
    pattern = r'(?<!\w)' + re.escape(keyword) + r'(?!\w)'
    return re.search(pattern, subject, re.IGNORECASE) is not None

def classify_subject(subject: str) -> dict:
    """
    Classifies the input subject string into the hierarchical categories.
    
    Returns a dictionary where keys are the broad categories and values 
    are lists of matched subcategories.
    """
    subject_lower = subject.lower()
    classification = {}
    for broad_category, subcategories in unicode_docs_hierarchy_keywords_refined.items():
        for subcat, keywords in subcategories.items():
            for keyword in keywords:
                if match_keyword(subject_lower, keyword):
                    if broad_category not in classification:
                        classification[broad_category] = []
                    if subcat not in classification[broad_category]:
                        classification[broad_category].append(subcat)
                    # Break on first match for the current subcategory.
                    break
    return classification if classification else {"Others/Miscellaneous": []}

if __name__ == "__main__":
    
    working_dir = os.getcwd()
    file_name = 'utc_register_all.xlsx'
    file_path = os.path.join(working_dir, file_name)

    df = pd.read_excel(file_path)
    df.dropna(subset=['doc_num', 'source', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df['doc_type'] = df['subject'].apply(classify_subject)
    
    df.to_excel('utc_register_all_classified.xlsx', index=False)