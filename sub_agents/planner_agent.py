from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm
from pydantic import BaseModel, Field
from typing import Literal
import os

class Plan(BaseModel):
    need_rewrite: bool = Field(description="True if the query is ambiguous or needs clarification")
    need_rag: bool = Field(description="True if the query needs documents/meetings knowledge")
    answer_mode: Literal["rag", "direct"] = Field(description="Either 'rag' or 'direct'")
    reason: str = Field(description="Short reason for the decision")

class PlannerAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            model=LiteLlm(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
            instruction="""You are a query planner that analyzes user questions and creates execution plans.
            
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

Return ONLY valid JSON that matches the schema.""",
            output_schema=Plan,  # Pydantic model class
            output_key="plan"    
        )

planner_agent = PlannerAgent()