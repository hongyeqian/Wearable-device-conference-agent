import re
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from openai import OpenAI

client = OpenAI()  # 读取环境变量里的 OPENAI_API_KEY


# ----------------------------------------------------
# Prompt: raw transcript -> dataXXX.md 格式
# ----------------------------------------------------
DATA_TRANSCRIPT_PROMPT_TEMPLATE = r"""
You are a transcript reformatting agent.

You will receive a raw meeting transcript (e.g., from Otter.ai) and your task is to rewrite it into a structured markdown transcript file named conceptually `dataXXX.md`.

You MUST strictly follow this target format and structure:

1. The file MUST start with a single heading line:
   # dataXXX
   where XXX is the meeting number (e.g. 004, 010, 011).

2. After that, every speaker turn MUST be written as ONE paragraph block in the following pattern:

   [#P001] [mm:ss] **Speaker Name:** utterance text...

   - `[#P001]`:
     - The "P" id MUST start at `[#P001]` and increase sequentially:
       [#P001], [#P002], [#P003], ...
     - Always use exactly 3 digits for the index.

   - `[mm:ss]`:
     - Use the real timestamp from the raw transcript whenever it is available.
     - Normalize timestamps to `mm:ss` (e.g. "0:5" -> "00:05", "1:23" -> "01:23").
     - If the raw transcript contains a YouTube-style link wrapper like
       [[00:41](http://...)] you may keep that wrapper exactly.
     - If there is no URL in the original, then use a plain timestamp like [00:41].

   - `**Speaker Name:**`:
     - Use the speaker name from the raw transcript (e.g. "Speaker 1", "Unknown Speaker", "Ankit", "Hongye Qian").
     - Do NOT invent new speakers.
     - Preserve speaker identity consistently.

   - `utterance text...`:
     - Copy the spoken content of that turn, cleaned up minimally
       (fix obvious transcription artifacts or repeated filler words if needed).
     - Do NOT summarize; this is still a transcript, not a summary.
     - You MAY merge multiple consecutive lines from the SAME speaker into one paragraph block,
       as long as you keep their content and chronological order.

3. Paragraph separation:
   - Leave ONE blank line between each [#PXXX] block, exactly like:
     [#P001] ...
     
     [#P002] ...
     
     [#P003] ...
   - Do NOT add extra headings, tables, or other sections.

4. General rules:
   - Preserve the chronological order of the conversation.
   - Do NOT delete important content.
   - Do NOT translate the language; keep it as in the original.
   - Do NOT add analysis or comments.
   - The only heading should be "# dataXXX" at the very top.
   - After that, only [#PXXX] lines + blank lines, following the pattern shown above.

5. Very important for parsing:
   - Every speaker turn MUST start with a line that begins exactly with:
     [#P
     followed by a 3-digit number, a closing bracket, a space, a timestamp in square brackets,
     a space, and then `**Speaker Name:**`.
   - If you omit or alter this pattern, the output will be INVALID for the downstream parser.

Your output MUST be only the final markdown content for dataXXX.md. Do NOT wrap it in code fences. Do NOT explain anything.
"""


def _coerce_raw_paths(raw_paths: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    """Normalize raw transcript inputs into a list of Path objects."""
    if isinstance(raw_paths, (str, Path)):
        paths = [raw_paths]
    else:
        paths = list(raw_paths)

    resolved = []
    for idx, p in enumerate(paths):
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"raw transcript path #{idx + 1} not found: {path}")
        resolved.append(path)
    if not resolved:
        raise ValueError("At least one raw transcript path is required.")
    return resolved


def _load_raw_texts(paths: Sequence[Path]) -> str:
    """Read each transcript file and stitch them together separated by blank lines."""
    texts = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            texts.append(text)
    if not texts:
        raise ValueError("All provided transcript files are empty.")
    return "\n\n".join(texts)


def generate_data_style_transcript(raw_paths: Union[str, Path, Sequence[Union[str, Path]]], meeting_no: str) -> str:
    """
    Convert one or multiple raw transcripts (e.g. Otter.ai txt) into dataXXX.md-style markdown.

    Parameters
    ----------
    raw_paths : Union[str, Path, Sequence[str | Path]]
        Path(s) to the raw transcript file(s), e.g. "./datademo/con1/2025.11.21.txt".
        Multiple files will be concatenated in the given order before sending to the model.
    meeting_no : str
        Meeting number string like "004", "010", "011".
        It will be used to replace XXX in "# dataXXX" and in the system prompt.

    Returns
    -------
    str
        Path to the generated dataXXX.md file.
    """
    raw_path_list = _coerce_raw_paths(raw_paths)
    raw_text = _load_raw_texts(raw_path_list)

    # Ensure meeting_no is 3-digit string
    if not re.fullmatch(r"\d{3}", meeting_no):
        raise ValueError("meeting_no must be a 3-digit string like '004', '010', '011'.")

    system_prompt = DATA_TRANSCRIPT_PROMPT_TEMPLATE.replace("dataXXX", f"data{meeting_no}") \
                                                   .replace("XXX", meeting_no)

    input_names = ", ".join(p.name for p in raw_path_list)
    print(f"[INFO] Converting {input_names} -> data{meeting_no}.md ...")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
    )

    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI response does not contain any content.")

    # 防一手模型给你套 ```markdown``` 代码块
    text = content.strip()
    if text.startswith("```"):
        # 粗暴但常用的去 fence 方式
        lines = text.splitlines()
        # 去掉第一行 ```xxx
        if lines[0].startswith("```"):
            lines = lines[1:]
        # 如果最后一行是 ``` 也去掉
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    out_path = raw_path_list[0].with_name(f"data{meeting_no}.md")
    out_path.write_text(text, encoding="utf-8")

    print(f"[OK] Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    # 简单命令行用法
    raw = input(
        "Enter raw transcript path(s) (comma separated, e.g. ./datademo/con1/2025.11.21.txt): "
    ).strip()
    meeting_no = input("Enter meeting number (3 digits, e.g. 011): ").strip()
    if raw and meeting_no:
        raw_inputs = (
            [part.strip() for part in raw.split(",")] if "," in raw else [raw.strip()]
        )
        generate_data_style_transcript([p for p in raw_inputs if p], meeting_no)
    else:
        print("Path or meeting number missing.")