import re
from pathlib import Path
from openai import OpenAI

client = OpenAI()  # will use OPENAI_API_KEY from env


# ----------------------------------------------------
# Metadata-level prompt (English, schema locked to metaData004.json)
# ----------------------------------------------------
METADATA_PROMPT_TEMPLATE = r"""
You are a metadata-level generation agent for a RAG system.

Your input:
- A single transcript markdown file named like `dataXXX.md`.
- The file may contain:
  - Speaker names and roles (e.g., host, guest, CEO, etc.).
  - References to organizations, brands, podcasts, channels.
  - Sometimes an explicit title, URL, or other metadata in the text.

Your task:
From this transcript ONLY, generate ONE JSON object conceptually named `metaDataXXX.json`
with a structure that is IDENTICAL to the example metaData004.json schema.

------------------------------------------------------------
## ABSOLUTE FORMAT RULES
------------------------------------------------------------

- Your output MUST be valid JSON.
- NO comments, NO trailing commas, NO Markdown code fences.
- Output MUST be a single JSON object only.
- The key order and nesting MUST match the following schema EXACTLY:

{
  "meeting_id": "dataXXX",
  "title": "Exact or best-guess title of the talk/episode, if present; otherwise a concise descriptive title",
  "datetime": "YYYY-MM-DDTHH:MM:SSZ or null if not available",
  "duration_sec": 0,
  "language": "en",
  "source_url": "URL of the original content if clearly provided; otherwise null",
  "meeting_type": "High-level type such as \"Podcast Interview\", \"Keynote\", \"Panel Discussion\", etc.",

  "participants": [
    {
      "name": "Full name of participant 1",
      "role": "Their role or description as inferred from the transcript",
      "aliases": ["Short name or common variants, if available"]
    },
    {
      "name": "Full name of participant 2",
      "role": "Their role or description",
      "aliases": ["Alias A", "Alias B"]
    }
  ],

  "organizations": [
    "Organization or brand 1 mentioned as context for speakers",
    "Organization or brand 2"
  ],

  "topics": [
    "Short phrase for Topic 1 covering an important theme of the conversation",
    "Short phrase for Topic 2",
    "Short phrase for Topic 3"
  ],

  "keywords": [
    "Concise keyword 1 (names, key concepts, frameworks)",
    "Concise keyword 2",
    "Concise keyword 3"
  ],

  "related_files": {
    "transcript": "dataXXX.md",
    "meeting_summary": "meetLevelXXX.md",
    "summary_embedding": "summaryXXX.md"
  },

  "embedding_text_pointer": "summaryXXX.md",

  "summary_brief": "One short paragraph (2–3 sentences) summarizing the entire conversation in neutral, factual style.",

  "actions": [
    {
      "action_id": "MXXX-A01",
      "task": "Concise action/task derived from the key outcomes of the conversation.",
      "assignee": {
        "name": "Name of the person responsible for this action",
        "role": "Their role",
        "aliases": ["Any aliases used in the transcript, if applicable"]
      },
      "description_plain": "Slightly longer plain-language description of the same action, staying fully consistent with the conversation.",
      "due_date": "TBD",
      "status": "open",
      "priority": "medium"
    },
    {
      "action_id": "MXXX-A02",
      "task": "Second action/task, if any. If there is only one clear action in the conversation, you may omit this object.",
      "assignee": {
        "name": "Name of the responsible person",
        "role": "Their role",
        "aliases": ["Alias if available"]
      },
      "description_plain": "Plain-language explanation of this second action.",
      "due_date": "TBD",
      "status": "open",
      "priority": "low"
    }
  ]
}

- You MUST keep all keys in exactly this order.
- You MUST keep the nested structure exactly the same.
- You MAY shorten arrays (e.g., participants, topics, keywords, actions) if the conversation clearly has fewer items.
- If there are no clear items for a list, you MUST return an empty list [] for that field.

------------------------------------------------------------
## FIELD-BY-FIELD RULES
------------------------------------------------------------

1. "meeting_id"
   - MUST be "dataXXX", where XXX is taken from the transcript filename.
   - Do NOT invent another ID.

2. "title"
   - Use the exact title if clearly given.
   - Otherwise, create a concise descriptive title based ONLY on the main theme and speakers,
     e.g. "Discussion on <topic> with <speaker names>".

3. "datetime"
   - Only fill with a precise datetime if clearly specified.
   - If not available, set to null.

4. "duration_sec"
   - Use a numeric value only if the duration can be clearly inferred.
   - Otherwise, set to 0.

5. "language"
   - Set to "en" if the transcript is mainly in English.
   - If clearly another language, use the appropriate ISO code.

6. "source_url"
   - Use the URL only if clearly mentioned.
   - Otherwise, set to null.

7. "meeting_type"
   - Use a short high-level type such as "Podcast Interview", "Internal Meeting",
     "Lecture", "One-on-one supervision meeting", etc., inferred from format.

8. "participants"
   - Include all main speakers.
   - "name": the actual name or label used in the transcript (e.g. "Ankit", "Hongye Qian", "Host").
   - "role": short description inferred ONLY from the transcript (e.g. "Supervisor", "Student").
   - "aliases": list of known variants or nicknames; [] if none.

9. "organizations"
   - List organizations, teams, or companies explicitly mentioned and relevant to context.
   - Do NOT invent organizations that never appear.

10. "topics"
    - 3–10 short phrases summarizing the major themes or sections of the conversation.
    - Each entry must be a short noun phrase, similar to paper section titles.

11. "keywords"
    - 8–15 concise keywords: names, methods, technical terms, frameworks, or recurring concepts.
    - Keep each keyword short (usually 1–3 words).

12. "related_files"
    - MUST be:
      "transcript": "dataXXX.md"
      "meeting_summary": "meetLevelXXX.md"
      "summary_embedding": "summaryXXX.md"
    - Replace XXX with the same meeting number.

13. "embedding_text_pointer"
    - MUST be "summaryXXX.md" with the same XXX.

14. "summary_brief"
    - One short paragraph (2–3 sentences).
    - Neutral, factual, high-level summary of the entire conversation.

15. "actions"
    - Extract 0–3 realistic actions:
      - Only if clearly implied or explicitly stated as tasks / next steps.
    - "action_id":
      - Use "MXXX-A01", "MXXX-A02", ... (XXX is meeting number).
    - "task":
      - Short phrase describing the action.
    - "assignee":
      - The person responsible, consistent with "participants".
    - "description_plain":
      - Slightly longer explanation of the same action.
    - "due_date":
      - Use "TBD" unless an explicit deadline is given.
    - "status":
      - Use "open" by default.
    - "priority":
      - Use "low", "medium", or "high" based on emphasis in the conversation.
    - If no clear actions exist, return "actions": [].

------------------------------------------------------------
## GLOBAL RULES
------------------------------------------------------------

- ZERO hallucination of concrete facts (no fake dates, URLs, company names, or people).
- You MAY infer high-level topics and keywords, but they must be consistent with the transcript.
- You MUST respect the JSON schema strictly.
- You MUST NOT output Markdown, only plain JSON.
- If information is not available:
  - Use null for scalars (e.g. "datetime", "source_url").
  - Use 0 for "duration_sec" if unknown.
  - Use [] for empty lists.
"""


def generate_metadata_from_transcript(transcript_path: str) -> str:
    """
    Given a transcript path like ".../data010.md",
    call gpt-4o-mini to generate the corresponding "metaData010.json"
    in the same directory, with schema identical to metaData004.json.
    """
    path_obj = Path(transcript_path)
    transcript_text = path_obj.read_text(encoding="utf-8")

    # Extract meeting number XXX from filename (dataXXX.md)
    m = re.search(r"data(\d+)\.md", path_obj.name)
    if not m:
        raise ValueError(f"Cannot extract meeting number from filename: {path_obj.name}")
    meeting_no = m.group(1)

    # Fill XXX placeholders in the prompt
    system_prompt = METADATA_PROMPT_TEMPLATE.replace("XXX", meeting_no)

    print(f"[INFO] Generating metaData{meeting_no}.json from {path_obj.name} ...")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_text},
        ],
    )

    content = resp.choices[0].message.content or ""

    # Trim surrounding code fences while keeping inner JSON unchanged.
    content_stripped = content.strip()
    if content_stripped.startswith("```"):
        lines = content_stripped.splitlines()
        fence_line = lines.pop(0) if lines else ""
        if lines and lines[-1].strip() == "```":
            lines.pop()
        content_stripped = "\n".join(lines).strip()

    out_path = path_obj.with_name(f"metaData{meeting_no}.json")
    out_path.write_text(content_stripped, encoding="utf-8")

    print(f"[OK] Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    inp = input("Enter transcript path (e.g. ./datademo/con1/data010.md): ").strip()
    if not inp:
        print("No path provided.")
    else:
        generate_metadata_from_transcript(inp)
