import pandas as pd
import numpy as np
import os
import re


unicode_docs_emoji_keywords = {
    "Emoji Relevant": ["emoji", "emojis", "zwj", "emoticon", "emoticons", "kaomoji", "afroji", "animoji",
                       "emojipedia", "emojify", "emojification"]
}

unicode_docs_hierarchy_keywords_refined = {
    "Meeting Documents": {
        "Agendas": [
            "agenda", "agendas", "meeting agenda", "discussion topic", "discussion topics", "topic for discussion", "topics for discussion"
        ],
        "Minutes": [
            "minutes", "mom", "m.o.m.", "notes from meeting", "meeting summary", "meeting record", "record of meeting"
        ],
        "Decisions & Action Items": [
            "decision", "decisions", "resolution", "resolutions", "conclusion", "conclusions", "recommendation", 
            "recommendations", "approved", "approve", "reject", "rejection", "denial", "action item", "action items",
            "motion", "motions", "follow-up", "follow up", "follow ups",  "heads-up", "notice", "notices",
            "notification", "notifications", "announcement", "announcements", "bulletin", "bulletins", "advisory", "advisories" 
        ]
    },
    "Public Review & Feedback": {
        "Public Review Issues (PRI)": [
            "pri", "pris", "p.r.i.", "pri feedback", "p.r.i. feedback", "public review issue", "public review issues"
        ],
        "Clarification & Queries": [
            "clarification", "clarifications","explain", "explanation", "explanations", "clarify", "query", "queries",
            "question", "questions", "inquiry", "inquiries", "ask", "please clarify"
        ],
        "General Feedback & Correspondence": [
            "feedback", "feed-back", "feed back", "comments", "comment", "remarks", "remark", "observation",
            "observations", "review", "reviews","response", "responses", "reply to", "rebuttal", "reaction to", "counter-proposal",
            "counter proposal", "issue", "issues", "problem", "problems", "concern", "concerns", "reaction", "reactions",
            "notes", "discussion", "letter", "letters", "mailing list", "misc mailing", "mails"
        ]
    },
    "Proposals": {
        "Character Encoding Proposals": [
            "proposal", "proposals", "proposed", "prop.", "submission", "submissions", "encode",
            "addition", "add", "new character", "new characters", "new script", "new scripts",
            "disunify", "unify", "un-encode", "unencode", "de-unify", "request to encode",
            "request for codepoints", "request for addition", "draft encoding",
            "new emoji", "emoji submission",  "emoji submissions"
        ],
        "Technical & Process Proposals": [
            "change request", "change requests", "change order", "change orders",
            "addendum", "addenda", "amendment", "amendments", "extension", "extensions",
            "revision", "revisions", "revised", "revised doc",
            "appendix", "appendices", "modification", "modifications",
            "process proposal", "process proposals", "technical proposal", "technical proposals",
            "update", "updates", "edit", "edits", "refine", "refinement", "synchronize", "synchronization",
            "fix", "fixes", "fixed","alternative", "alternatives", "alternative encoding",
            "line_break", "word_break", "sentence_break", "line break", "line breaks", "word break",
            "word breaks", "sentence break", "sentence breaks","collation", "normalization", "normalizations",
            "compatibility", "data file", "data files", "code chart", "code charts", "nameslist", "alias",
            "aliases", "terminal_punctuation", "namespace", "loose matching"
        ]
    },
    "Standards & Specifications": {
        "Unicode Standard & Annexes (UAX)": [
            "uax", "unicode standard annex", "unicode annex", "annex", "uax#", "annex update", "(snapshot)"
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
            "errata", "erratum", "corrigendum", "corrigenda", "correction", "corrections",
            "issue", "issues", "error", "errors", "bug", "bugs", "typo", "typos",
            "inconsistency", "inconsistencies", "problem", "problems", "wrong",
            "defect", "defects", "suspicious", "fix", "fixes", "missing"
        ]
    },
    "Liaison & External": {
        "ISO/IEC WG2 Documents & Ballots": [
            "iso", "wg2", "iec", "jtc1", "sc2", "ballot", "ballots", "national body", "iso ballot",
            "iso comment", "joint technical committee", "nb comment", "nxxx doc", "irg n", "irg #"
        ],
        "Liaison Reports & Agreements": [
            "liaison", "mou", "memorandum of understanding", "agreement",  "agreements", "contract", "sla", "liaison report"
        ]
    },
    "Administrative & Miscellaneous": {
        "Document Registers & Indexes": [
            "register","registers", "index", "indexes"
        ],
        "Schedules & Planning": [
            "schedule", "schedules", "planning", "timetable", "timetables", "release plan", "release plans", "roadmap", "roadmaps", "calendar", "calendars"
        ],
        "Memo/Circular": [
            "memo", "memos", "memorandum", "memoranda", "circular", "circulars", "internal note", "internal notes", "bulletin", "bulletins"
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

def classify_document_type(subject: str) -> dict:
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

def classify_emoji_relevance(subject: str) -> str:
    """
    Classifies the input subject string into emjoi-relevant and other Unicode categories.
    """
    classification = "Irrelevant"
    subject_lower = subject.lower()
    for category, keywords in unicode_docs_emoji_keywords.items():
        for keyword in keywords:
            if match_keyword(subject_lower, keyword):
                classification = category
                break
    return classification

if __name__ == "__main__":
    
    working_dir = os.getcwd()
    file_name = 'utc_register_all.xlsx'
    file_path = os.path.join(working_dir, file_name)

    df = pd.read_excel(file_path)
    df.dropna(subset=['doc_num', 'source', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df['doc_type'] = df['subject'].apply(classify_document_type)
    df['emoji_relevance'] = df['subject'].apply(classify_emoji_relevance)
    
    df.to_excel('utc_register_all_classified.xlsx', index=False)