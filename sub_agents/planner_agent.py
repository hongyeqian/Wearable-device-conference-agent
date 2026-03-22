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
    instruction="""You are a smart query planner analyzing the user's latest query ({user_query}) to create execution plans.
            
The conversation history (context) will be automatically provided to you by the system.

Follow these 3 steps:

STEP 1: Generate `resolved_query`
- Look at the user's latest query. If it relies on conversational context (e.g., uses pronouns like "he", "that meeting", or implicitly continues a topic), incorporate the history to rewrite the latest query into a standalone, self-contained sentence.
- If it does not rely on history, return the original {user_query} unmodified. This means orginal user query is `resolved_query`.
- ALWAYS output a `resolved_query`.

STEP 2: Determine `need_rewrite`
- Evaluate the `resolved_query` generated in Step 1.
- HARD RULES for need_rewrite=True:
  - If `resolved_query` contains first-person pronouns (I/we/my/our/us).
  - If `resolved_query` contains relative time expressions (last N meetings, last week, yesterday, recent).
  - If `resolved_query` contains a person's name, but you are not sure it's their full complete name.
- If the `resolved_query` is perfectly explicit (e.g., specific date "2025-11-29", full exact names) and doesn't need external Pandas ID resolution, set need_rewrite=False.

STEP 3: Determine `need_rag`
- True if the `resolved_query` asks for knowledge from documents/meetings.
- Skip RAG for general knowledge, greetings, or writing tasks.

Return ONLY valid JSON that matches the schema.""",
    description="Decides whether to use query rewrite and RAG retrieval",
    output_schema=Plan,
    output_key="plan",
    # By omitting include_contents="none", we default to allowing ADK to ingest ctx.session.events
    include_contents="default"
)
