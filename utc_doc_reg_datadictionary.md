**UTC Proposals — column cheat-sheet**

| Column              | What it contains (short & sweet)                                              |
| ------------------- | ----------------------------------------------------------------------------- |
| **doc\_num**        | The formal UTC/L2 document number for the emoji proposal (e.g., “L2/23-261”). |
| **proposal\_link**  | Clickable Unicode URL that fetches the full proposal PDF/HTML.                |
| **proposal\_title** | The proposal’s headline—usually “Proposal for Emoji: …”.                      |
| **proposer**        | Name(s) of the individual(s) or group(s) submitting the proposal.             |
| **count**           | Simple tally—how many times this exact proposal shows up in your dataset.     |


**UTC Document Register — column cheat-sheet**

| Column                     | What it holds (in one crisp line)                                                 |
| -------------------------- | --------------------------------------------------------------------------------- |
| **doc\_num**               | Official UTC/L2 document number, e.g., “L2/11-001”.                               |
| **doc\_url**               | Relative or absolute path to the file on the Unicode site.                        |
| **subject**                | The document’s title or topic line.                                               |
| **source**                 | Person or organisation that submitted the doc (author, company, WG, etc.).        |
| **date**                   | UTC timestamp when the doc was logged.                                            |
| **doc\_type**              | Hierarchical tags showing which register bucket(s) the doc lives in.              |
| **emoji\_relevance**       | Quick label (“Relevant” / “Irrelevant”) from your classifier.                     |
| **file\_extension**        | File type on disk – pdf, html, txt, htm, etc.                                     |
| **extracted\_doc\_refs**   | List of other L2/UTC numbers the doc cites.                                       |
| **emoji\_chars**           | Raw emoji characters found inside the doc.                                        |
| **unicode\_points**        | Same set, expressed as “U+XXXX” code points.                                      |
| **is\_emoji\_relevant**    | Binary flag (1/0 or TRUE/FALSE) mirroring *emoji\_relevance*.                     |
| **emoji\_keywords\_found** | Emoji-related trigger words matched in the text (e.g., “ZWJ”, “pictograph”).      |
| **emoji\_shortcodes**      | Slack/GitHub-style shortcodes (e.g., `:smile:`) harvested from the file.          |
| **file\_size\_kb**         | Payload size in kilobytes after download.                                         |
| **error\_message**         | Any fetch/parse error captured for this record.                                   |
| **emoji\_relevant**        | Duplicate boolean kept for legacy scripts; same meaning as *is\_emoji\_relevant*. |
| **people**                 | Named-entity list of individual contributors mentioned.                           |
| **emoji\_references**      | Free-text mentions of specific emoji proposals, meetings, code points, etc.       |
| **entities**               | Non-person named entities (Unicode Consortium, WG2, ISO/IEC, …).                  |
| **summary**                | Auto-generated paragraph-length digest of the document.                           |
| **description**            | Concise tagline or blurb (shorter than *summary*).                                |
| **other\_details**         | Miscellaneous notes your pipeline couldn’t slot elsewhere.                        |
| **processing\_error**      | Exception text from any downstream NLP step.                                      |
| **token\_usage**           | Dict of prompt/completion/total tokens consumed by the LLM.                       |
| **api\_cost**              | Dollar cost charged for that LLM call.                                            |
