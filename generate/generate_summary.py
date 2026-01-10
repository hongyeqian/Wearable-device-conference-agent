import os
import re
from pathlib import Path
from openai import OpenAI


# 在导入OpenAI之前取消设置SSL_CERT_FILE
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

client = OpenAI() 



SUMMARY_PROMPT_TEMPLATE = """
You are a summary-level generation agent.
Your input will be a single `meetLevelXXX.md` file.
Your task is to generate the corresponding `summaryXXX.md` file.
Your output must follow the structure and style of `summary004.md` exactly.

------------------------------------------------------------
## TARGET OUTPUT FORMAT (MUST MATCH summary004.md)
------------------------------------------------------------

You MUST produce a Markdown file with this structure:

# Summary Level Summary

## MXXX — {{Episode Title or Speaker Names}}

- **[SXXX-T01] Topic: {{Topic Title from meetingLevel}}**
  {{One concise paragraph summarizing the topic. No timestamps. No dialogues. No new information.}}
  Reference: {{Meeting-level Topic id}}

- **[SXXX-T02] Topic: ...**
  ...

- **[SXXX-A01] Action: {{Action Title}}**
  {{One concise paragraph summarizing the action item, strictly based on meeting-level action content.}}
  Reference: MXXX-AXX

Important:
- The line starting with "- **[SXXX-T01] Topic:" is REQUIRED for every topic.
- The line starting with "- **[SXXX-A01] Action:" is REQUIRED for every action.
- Do NOT change this pattern; my parser depends on it.

------------------------------------------------------------
## STRICT CONSTRAINTS
------------------------------------------------------------

1. Topic order
   - The order of topics in the summary MUST match the order in the meetLevel file.
   - Do NOT merge, drop, or reorder topics.

2. Reference mapping
   - For each topic:
       Meeting-level Topic id = MXXX-T0N
       => Summary-level Reference MUST be "MXXX-T0N".
   - For each action item:
       Meeting-level Action id = MXXX-A0N
       => Summary-level Reference MUST be "MXXX-A0N".

3. Source of content
   - You MUST summarize ONLY from the meeting-level file.
   - Do NOT go back to the original transcript.
   - Do NOT add any idea that is not present in the meeting-level file.
   - NO speculation, NO extra interpretation.

4. Writing style (must match summary004)
   - Single paragraph per topic / per action.
   - Highly condensed and thematic, like an abstract.
   - No timestamps.
   - No dialogue format.
   - No bullet points inside the paragraph.
   - No new viewpoints or examples.
   - Clear logic, explicit theme, concise and slightly academic tone.

5. ID formatting (MXXX, SXXX)
   - Meeting number = XXX (taken from filename).
   - Topic IDs:
       [SXXX-T01], [SXXX-T02], ...
   - Action IDs:
       [SXXX-A01], [SXXX-A02], ...
   - You MUST use exactly this ID pattern.

------------------------------------------------------------
## HOW TO EXTRACT FROM MEET-LEVEL
------------------------------------------------------------

For each Topic in meetLevelXXX.md:
- Use:
    - Topic Title
    - All bullet points under "Summary:"
    - Topic id (e.g. MXXX-T01)
- Transform the multiple summary bullets into ONE compact paragraph
  that captures the main idea of that topic.
- You may lightly paraphrase but must keep the original meaning.

For each Action Item in meetLevelXXX.md:
- Use:
    - Responsible Person
    - Task
    - Context
    - (Duration is optional; you do NOT need to mention it explicitly)
    - Action Topic id (e.g. MXXX-A01)
- Turn Task + Context into a single concise paragraph.
- The Action title can include the responsible person and a short task label.

------------------------------------------------------------
## OUTPUT REQUIREMENTS
------------------------------------------------------------

- Output MUST be a complete Markdown document.
- The heading "# Summary Level Summary" MUST be present.
- The second heading MUST be "## MXXX — {{Title}}".
- Each topic MUST start with a line exactly like:
    - **[SXXX-T0N] Topic: {{...}}**
- Each action MUST start with:
    - **[SXXX-A0N] Action: {{...}}**
- Do NOT include tables or any structure from meet-level.
- Output MUST be in English.

Remember:
- ZERO hallucination.
- ZERO structural changes.
- ONLY summarize what is already present in the meetLevel file.
"""


def generate_summary_from_meetlevel(meetlevel_path: str) -> str:
    """
    Given a meetLevel file path like ".../meetLevel010.md",
    call gpt-4o-mini to generate the corresponding "summary010.md"
    in the same directory, following summary004.md style.
    """
    path_obj = Path(meetlevel_path)
    text = path_obj.read_text(encoding="utf-8")

    # Extract meeting number XXX from filename
    m = re.search(r"meetLevel(\d+)\.md", path_obj.name)
    if not m:
        raise ValueError(f"Cannot extract meeting number from filename: {path_obj.name}")
    meeting_no = m.group(1)

    # Prepare system prompt: replace XXX with actual meeting number
    system_prompt = SUMMARY_PROMPT_TEMPLATE.replace("XXX", meeting_no)

    print(f"[INFO] Generating summary{meeting_no}.md from {path_obj.name} ...")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    content = resp.choices[0].message.content or ""

    # Output path: same folder, summaryXXX.md
    out_path = path_obj.with_name(f"summary{meeting_no}.md")
    out_path.write_text(content, encoding="utf-8")

    print(f"[OK] Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    # 简单命令行用法：手动输入 meetLevel 文件路径
    inp = input("Enter meetLevel file path (e.g. ./datademo/con1/meetLevel010.md): ").strip()
    if not inp:
        print("No path provided.")
    else:
        generate_summary_from_meetlevel(inp)