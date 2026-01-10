import os
import re
from openai import OpenAI

# 在导入OpenAI之前取消设置SSL_CERT_FILE
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

client = OpenAI()



MEETLEVEL_PROMPT = """
You are a meeting-level summarization agent.
You will receive a raw transcript markdown file (dataXXX.md) containing numbered paragraphs P001–PXXX.

Your task is to generate ONE output file named meetLevelXXX.md whose structure and writing style must be IDENTICAL to meetLevel004.md.


Every topic MUST begin with EXACTLY the following line:

- **Topic Title:** <short thematic label>

This line MUST appear, MUST be the first bullet of every topic block, and MUST NOT be deleted, replaced, reformatted, merged, or omitted.  
If any topic does NOT contain this line EXACTLY as written ("- **Topic Title:**"), the output is INVALID.

You MUST validate your own output before finishing.  
If even ONE topic is missing the exact substring "- **Topic Title:**",  
you MUST regenerate your answer until ALL topics contain it.

------------------------------------------------------------
# HARD STRUCTURE REQUIREMENTS (NO CHANGES ALLOWED)
------------------------------------------------------------

Your output MUST contain exactly the following structure:

# Meeting Level Summary

## Key Topics

- **Topic Title:** <short thematic label>
  - Topic id: MXXX-T01
  - Reference: Pxxx–Pyyy
  - Summary:
      - [timestamp] Speaker: compressed statement...
      - [timestamp] Speaker: compressed statement...
  - Participants: <comma-separated list of speakers>
  - Duration: [start_timestamp–end_timestamp]
(**Topic Title** <short thematic label, **Topic Title** must be generate in the md, it is an important index)
(Repeat as many topics as needed. You MUST extract ALL topics in chronological order.)

## Action Items

- **Responsible Person:** <name>
  - Topic id: MXXX-A01
  - Reference: Pxxx–Pyyy
  - Task: <explicit task extracted from transcript>
  - Context: <summarized origin of the task>
  - Duration: [timestamp–timestamp]
  - Deadline (if any): TBD

(Repeat only if explicit tasks exist.)

------------------------------------------------------------
# RULES FOR TOPIC GENERATION
------------------------------------------------------------

1. You MUST extract **all** major topics from the transcript.
   A “topic” = a continuous segment of conversation with a coherent theme.

2. Topic titles should be **short, thematic, professional**, like meetLevel004.

3. Topic partitions MUST follow chronological order of P numbers.

4. Each topic's Reference range must use the correct Pxxx–Pyyy
   based ONLY on the transcript’s paragraph numbers.

------------------------------------------------------------
# RULES FOR SUMMARY BULLETS
------------------------------------------------------------

The summary section MUST:
- Contain multiple bullets (not one).
- Each bullet MUST include:
    - Real timestamp
    - Real speaker name
    - **Compressed summarization**, NOT verbatim transcript

VERY IMPORTANT:
❌ DO NOT repeat or rephrase every line.
❌ DO NOT generate line-by-line summaries.
✔ DO write only the **core meaning** of what the speaker said.

Follow the writing style of meetLevel004:  
Short, clean, thematic, highly compressed.

------------------------------------------------------------
# RULES FOR ACTION ITEMS
------------------------------------------------------------

1. Extract **every explicit task** mentioned in the transcript.
2. Each action must have:
   - One responsible person
   - A clear “Task” phrase (compressed)
   - A “Context” explaining which discussion produced the task
   - Correct P-reference range
   - Correct timestamps
3. DO NOT hallucinate tasks.
4. If the transcript contains NO clear tasks,
   you must output an empty Action Items section.

------------------------------------------------------------
# GLOBAL RULES
------------------------------------------------------------

- ZERO hallucination.
- ZERO invented content.
- ZERO invented timestamps.
- ZERO invented references.
- NEVER modify the structure, headings, or indentation.
- Output MUST be in English.
- Output MUST be a complete meetLevelXXX.md file.
- Every topic MUST include the line "- **Topic Title:** ..." as the FIRST line of the topic block. This line is used by a parser; if it is missing or altered, the output is INVALID.


Before producing the final output, you MUST:

1. Verify that every topic block contains EXACTLY the line:
   "- **Topic Title:**"

2. Verify the output begins with "# Meeting Level Summary" and contains the sections exactly as specified.

If any rule is violated, you MUST regenerate until compliance is perfect.

------------------------------------------------------------
# INPUT FORMAT
------------------------------------------------------------

You will be given the entire transcript as plain text. It will contain lines like:

[#P001] [00:00] **Speaker:** text...

Use these EXACT P numbers and timestamps in your output.
"""


def generate_meet_level(transcript_path: str):
    """
    Given a raw transcript path such as ./datademo/conX/data011.md,
    automatically generate meetLevel011.md and save it in the same folder.
    """

    # Read transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    # Extract meeting ID (XXX)
    # Matches any digits inside dataXXX.md
    m = re.search(r"data(\d+)\.md", transcript_path)
    if not m:
        raise ValueError("Cannot extract meeting number from filename.")
    meeting_no = m.group(1)

    # Construct the full prompt for the model
    system_prompt = MEETLEVEL_PROMPT.replace("XXX", meeting_no)

    print(f"Generating meetLevel{meeting_no}.md ...")

    # Call GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_text}
        ]
    )

    output_text = response.choices[0].message.content or ""

    # Build output file path
    folder = os.path.dirname(transcript_path)
    out_path = os.path.join(folder, f"meetLevel{meeting_no}.md")

    # Save result
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"✔ Saved: {out_path}")
    return out_path
  
  
if __name__ == "__main__":
  # Example: generate_meet_level("./datademo/con1/data011.md")
  path = input("Enter transcript path: ")
  generate_meet_level(path)