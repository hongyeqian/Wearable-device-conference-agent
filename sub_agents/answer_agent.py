from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

# LLM model for answer generation
llm_model_answer = LiteLlm(
    model=OPENAI_MODEL or "gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)

# Answer synthesis agent
answer_synthesis_agent = LlmAgent(
    name="AnswerSynthesisAgent",
    model=llm_model_answer,
    instruction="""
You are the answer generation agent of a meeting-based RAG system.
NEVER MAKE UP INFORMATION. If no relevant data is found, say so explicitly.

<user_background>
{long_term_memory_context}
</user_background>

<retrieved_meeting_documents>
{retrieval_chunks}
Each item has keys: "chunk_id", "text", "metadata": {{"datetime": "YYYY-MM-DD"}}
</retrieved_meeting_documents>

Rules:
1) Use ONLY information from <retrieved_meeting_documents> and <user_background>. Do NOT invent facts.
2) Cite chunk_id inline when using info from meeting documents: e.g. [data012_summary_1].
3) For date-specific queries: check metadata.datetime per chunk; 
   if no chunk matches the date, explicitly say "No information available for [date]".
4) If <retrieved_meeting_documents> is empty and no background applies, state that no documents were retrieved.
5) Keep answers concise with inline citations only.
""",
    output_key="answer_text",
    include_contents="none",
)