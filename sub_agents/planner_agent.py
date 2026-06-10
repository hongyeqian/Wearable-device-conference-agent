"""
Planner Agent - Decides whether to use query rewrite and RAG
"""
from pydantic import BaseModel, Field
from typing import List, Literal
from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm
# pyrefly: ignore [missing-import]
from config.settings import OPENAI_API_KEY, OPENAI_MODEL


class Plan(BaseModel):
    """Output schema for planner agent"""
    need_memory: bool = Field(
        description="Whether the query relates to the user's past preferences or personal context."
    )
    need_rewrite: bool = Field(
        description="Whether the resolved_query contains ambiguities (pronouns, relative time, fuzzy names) that require further pipeline resolution."
    )
    need_rag: bool = Field(
        description="Whether the query requires RAG retrieval to answer from knowledge base"
    )
    resolved_query: str = Field(
        description="The contextually independent version of the user's latest query, incorporating history if needed."
    )
    reason: str = Field(
        description="Brief explanation for the decisions made, you should explain why you set need_rewrite and need_rag to True or False."
    )


# Create planner LLM model
planner_llm = LiteLlm(
    model=OPENAI_MODEL or "gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)


# Planner agent - decides if query needs rewrite and/or rag
planner_agent = LlmAgent(
    name="PlannerAgent",
    model=planner_llm,
    instruction="""Analyze the user's latest query ({user_query}) using conversation history.

STEP 1 — resolved_query
Rewrite {user_query} into a standalone sentence.
RESOLVE references that point back to conversation history:
- "this/that meeting" → the specific meeting discussed earlier
- "he/she/they/it" → the actual entity from context
- Implicit topic continuation → make the subject explicit
DO NOT resolve temporal ambiguities — keep expressions like "last meeting", "recent 3 meetings", "yesterday" as-is. Those are handled downstream.
If {user_query} is already standalone, keep unchanged.

STEP 2 — need_rewrite
Scan resolved_query for these keywords (if ANY found → True):
- Pronouns: I, me, my, we, our, us
- Time words: last, recent, previous, past, yesterday, today, this week, this month
- If it appers a name (e.g. "Hongye", "Hongye Qian", "Qian")
If NONE of the above → False.
Example: "What did Hongye discuss in the last meeting?" → True (contains "last", "Hongye" is a name)

STEP 3 — need_rag
RAG searches the meeting records database, NOT conversation history.
True for ANY question about meetings: who attended, what was discussed, decisions, action items, topics, or any meeting content.
False ONLY for: greetings, general knowledge, math, or writing tasks with no meeting connection.
Default: when uncertain, set True.

STEP 4 — need_memory
True if the query asks about the user's personal preferences, background, past experiences, or relationships.
Example: "What do I like to drink?", "Who are my teammates?", "Do I have any background in AI?" -> True
False if the query is strictly about factual meeting contents without personal background implication.

Return valid JSON matching the schema.""",
    description="Decides whether to use query rewrite and RAG retrieval",
    output_schema=Plan,
    output_key="plan",
    include_contents="default"
)
