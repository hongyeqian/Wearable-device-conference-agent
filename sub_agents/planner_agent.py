"""
Planner Agent - Decides whether to use query rewrite and RAG
"""
from pydantic import BaseModel, Field
from typing import List, Literal
from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm
from config.settings import OPENAI_API_KEY, OPENAI_MODEL


class Plan(BaseModel):
    """Output schema for planner agent"""
    need_rewrite: bool = Field(
        description="Whether the query needs rewriting to resolve ambiguities (pronouns, time references, etc.)"
    )
    need_rag: bool = Field(
        description="Whether the query requires RAG retrieval to answer from knowledge base"
    )
    reason: str = Field(
        description="Brief explanation for the decisions made"
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
    instruction="""You are a query planner that analyzes {user_query} and creates execution plans.
            
            System has been initialized. You may safely use meeting_catalog and index_status from state.

Based on the user's query, determine:
1. need_rewrite: True if query is ambiguous, contains pronouns, or needs clarification
2. need_rag: True if query requires knowledge from documents/meetings
3. answer_mode: "rag" or "direct"

Rules:
- Force RAG for: meeting transcripts, action items, minutes, discussions
- Skip RAG for: general knowledge, definitions, writing tasks
- Conservative approach: when in doubt, enable RAG

HARD RULES FOR need_rewrite=True:
- Query contains first-person pronouns (I/we/my/our/us) → need_rewrite=True
- Query contains relative time expressions (last N meetings/last week/yesterday/recent/past N days) → need_rewrite=True  
- Query needs time range inference from meeting_catalog → need_rewrite=True
- If there is a name in user query, and you are not sure it is the full name.

Return ONLY valid JSON that matches the schema.""",
    description="Decides whether to use query rewrite and RAG retrieval",
    output_schema=Plan,
    output_key="plan",
    include_contents="none"
)
