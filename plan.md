# Emoji Proposal Metrics: Definitions & Formulas

This document defines all metrics used to analyze the flow, attention, and dynamics of emoji proposals in the Unicode Technical Committee (UTC) process. Each metric includes a short description and its calculation formula.

---

## 1. Proposal Flow & Velocity Metrics

- **Time to Last Reference**: Days from the proposal's first appearance (as a row, not a reference) to its last reference in the UTC registry.  
  _Formula_: `last_reference_date - first_appearance_date`
- **Reference Count**: Number of times the proposal is referenced in other documents (excluding the original).  
  _Formula_: `count(all references) - 1`
- **Velocity**: Rate of references per year/month.  
  _Formula_: `reference_count / (days_span / 365.25)` (per year), `reference_count / (days_span / 30.44)` (per month)
- **Dormancy & Revival**: Longest period (in days) between consecutive references; also, number of years/months with no references.  
  _Formula_: `max(gap between sorted reference dates)`; `years/months in span with no references`
- **First/Last Reference Dates**: Calendar dates of first and last appearance/reference.  
  _Formula_: `min(reference_dates)`, `max(reference_dates)`

---

## 2. Attention Dynamics & Social Metrics

- **Unique People Involved**: Count and list of unique people (from `people` column) across all references.  
  _Formula_: `len(set(all people))`
- **Attention Span**: Number of unique people per year.  
  _Formula_: `len(set(people in year))` for each year
- **Key Contributors**: Top 3-5 people by frequency of mention.  
  _Formula_: `most_common(people)`
- **Entities/Organizations**: Count and list of unique entities involved.  
  _Formula_: `len(set(all entities))`, `set(all entities)`
- **Attention Shifts**: New people/entities appearing each year compared to previous years.  
  _Formula_: `set(people/entities in year) - set(people/entities in previous years)`
- **Attention Drift**: Change in people/entities/emoji from first to last year.  
  _Formula_: `set(last_year) - set(first_year)` (added), `set(first_year) - set(last_year)` (lost)

---

## 3. Emoji & Content Metrics

- **Emoji Count**: Number of unique emoji characters discussed.  
  _Formula_: `len(set(all emoji_chars))`
- **Unicode Points**: Number of unique Unicode points referenced.  
  _Formula_: `len(set(all unicode_points))`
- **Emoji Diversity**: Ratio of unique emoji to total emoji mentions.  
  _Formula_: `len(set(all emoji_chars)) / total_emoji_mentions`
- **Emoji References**: Number and type of emoji references (from `emoji_references`).  
  _Formula_: `count(emoji_references)`

---

## 4. Email Attention & Engagement

- **Email Match Count**: Number of emails matched to the proposal.  
  _Formula_: `count(email matches)`
- **Email Attention Timeline**: First and last email dates, and email volume per year/month.  
  _Formula_: `min(email_dates)`, `max(email_dates)`, `count(emails per year/month)`
- **Email People**: Unique people in emails; overlap with UTC doc people.  
  _Formula_: `set(all email people)`, `set(email people) ∩ set(doc people)`
- **Email Confidence Score**: Average and max confidence score for matches.  
  _Formula_: `mean(confidence_score)`, `max(confidence_score)`
- **Email Subject Diversity**: Number of unique email subjects.  
  _Formula_: `len(set(email subjects))`

---

## 5. Year-wise & Temporal Analysis

- **Yearly Velocity**: References, people, entities, emoji, and emails per year.  
  _Formula_: `count(references/people/entities/emoji/emails in year)`
- **Processing Time**: Days from proposal submission to last reference and to last email.  
  _Formula_: `last_reference_date - submission_date`, `last_email_date - submission_date`
- **Yearly Attention Shifts**: People/entities/emoji involved each year.  
  _Formula_: `set(people/entities/emoji in year)`

---

## 6. Advanced/Creative Metrics

- **Proposal Lifespan**: Number of years between first and last reference/email.  
  _Formula_: `last_reference_year - first_reference_year`, `last_email_year - first_reference_year`
- **Burstiness**: Degree to which references/emails are clustered in time.  
  _Formula_: `stddev(gaps between reference/email dates)`
- **Cross-entity Collaboration**: Number of entities per year; years with >1 entity.  
  _Formula_: `len(set(entities in year))`, `count(years with >1 entity)`
- **Proposal Hotness**: Composite score (velocity + people + email count + emoji diversity), normalized.  
  _Formula_: `normalize(velocity) + normalize(people_count) + normalize(email_count) + normalize(emoji_diversity)`

---

## 7. Change/Trend Analysis

- **Processing Speed Trends**: Change in velocity per year.  
  _Formula_: `velocity_per_year[year]`
- **Email vs. Doc Attention**: Ratio of email matches to document references.  
  _Formula_: `email_count / reference_count`

---

_All metrics are computed per proposal unless otherwise specified. These metrics enable a multi-dimensional analysis of proposal flow, attention, and engagement in the UTC process._
