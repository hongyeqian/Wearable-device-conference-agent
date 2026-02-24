"""
RAG System Evaluation Script using RAGAS
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

# RAGAS imports 
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, AnswerCorrectness
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

# RAG system imports
from web_app.agent import FullRAGSystemAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from langchain_openai import OpenAIEmbeddings


# set up llm
async_client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=async_client, max_tokens=10000)

# for answer relevancy
embedding_client = AsyncOpenAI()
embeddings = embedding_factory(
    "openai", model="text-embedding-3-small", client=embedding_client
)

# create llm cases
faithfulness = Faithfulness(llm=llm)
answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
context_recall = ContextRecall(llm=llm)
answer_correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)

def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("test_cases", [])


def format_contexts_for_ragas(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into context string for RAGAS"""
    if not chunks:
        return "No relevant context found."
    
    lines = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "N/A")
        text = chunk.get("text", "")
        meta = chunk.get("metadata") or {}
        dt = meta.get("datetime", "N/A")
        
        lines.append(f"[{chunk_id}] (Date: {dt})")
        lines.append(text)
        lines.append("")
    
    return "\n".join(lines)


async def run_evaluation_query(agent: FullRAGSystemAgent, 
                             session_service: InMemorySessionService,
                             question: str) -> Dict[str, Any]:
    
    try:
        runner = Runner(
            agent = agent,
            session_service = session_service,
            app_name = "rag_eval",
        )
        
        # create the session
        session = await session_service.create_session(
            app_name = "rag_eval",
            user_id = "eval_user",
            state = {},
        )
        
        async for event in runner.run_async(
            user_id="eval_user",
            session_id=session.id,
            new_message=Content(role='user', parts=[Part(text=question)]),
        ):
            # Let the system interaction complete
            pass
        
        final_session = await session_service.get_session(
            app_name = "rag_eval",
            user_id = "eval_user",
            session_id = session.id,
        )
        
        # Get answer and chunks from session state
        session_state = final_session.state if final_session else {}
        answer = session_state.get("answer_text", "")
        retrieved_chunks = session_state.get("retrieval_chunks", [])
        
        contexts = format_contexts_for_ragas(retrieved_chunks)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "chunks": retrieved_chunks,
            "success": True,
            "session_state_keys": list(final_session.state.keys()),
        }
    except Exception as e:
        return {
            "answer": "",
            "contexts": "",
            "chunks": [],
            "success": False,
            "error": str(e),
        }

async def run_full_evaluation(
    test_file: str = "test_set/golden_set_demo.json",
    output_file: str = "test_set/evaluation_results.json"
):
    """
    Main evaluation function
    """
    print("=" * 60)
    print("RAG System Evaluation with RAGAS")
    print("=" * 60)
    
    # Load test cases
    test_cases = load_test_cases(test_file)
    print(f"✅ Loaded {len(test_cases)} test cases")
    
    # Create session service
    session_service = InMemorySessionService()
    
    # Create agent (this will initialize the vector store)
    print("\n🔧 Initializing RAG system...")
    agent = FullRAGSystemAgent()
    agent._initialize_components()
    print("✅ RAG system initialized")
    
    # Run evaluation for each test case
    print("\n🚀 Running RAG queries...")
    answers = []
    contexts_list = []
    
    for tc in tqdm(test_cases, desc="Processing"):
        result = await run_evaluation_query(agent, session_service, tc["question"])
        
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        
        # Store for analysis
        tc["retrieved_chunks"] = [
            {"chunk_id": c.get("chunk_id"), "datetime": c.get("metadata", {}).get("datetime")}
            for c in result.get("chunks", [])
        ]
        tc["evaluation_success"] = result["success"]
        if not result["success"]:
            tc["error"] = result.get("error")
        
        # Debug: show first result
        # TODO: remove this after debugging
        if tc["id"] == "q001":
            print(f"\n📝 Sample answer for q001:")
            print(f"   {result['answer'][:200]}...")
    
    # 使用新版 RAGAS API - 直接评估每个样本
    print("\n📈 Running RAGAS metrics with ascore()...")

    # 收集所有评估分数
    faithfulness_scores = []
    relevancy_scores = []
    correctness_scores = []
    recall_scores = []

    # 逐个评估每个样本
    for i, (tc, ans, ctx) in enumerate(zip(test_cases, answers, contexts_list)):
        print(f"   Evaluating sample {i+1}/{len(test_cases)}...")

        # 准备上下文列表
        contexts = [ctx] if ctx else []

        # Faithfulness
        try:
            f_result = await faithfulness.ascore(
                user_input=tc["question"],
                response=ans,
                retrieved_contexts=contexts
            )
            f_score = f_result.value if hasattr(f_result, 'value') else f_result
            faithfulness_scores.append(f_score)
        except Exception as e:
            print(f"      ⚠️ Faithfulness error: {e}")
            faithfulness_scores.append(None)

        # Answer Relevancy
        try:
            ar_result = await answer_relevancy.ascore(
                user_input=tc["question"],
                response=ans,
            )
            ar_score = ar_result.value if hasattr(ar_result, 'value') else ar_result
            relevancy_scores.append(ar_score)
        except Exception as e:
            print(f"      ⚠️ Answer Relevancy error: {e}")
            relevancy_scores.append(None)

        # Answer Correctness
        try:
            ac_result = await answer_correctness.ascore(
                user_input=tc["question"],
                response=ans,
                reference=tc.get("ground_truth", ""),
            )
            ac_score = ac_result.value if hasattr(ac_result, 'value') else ac_result
            correctness_scores.append(ac_score)
        except Exception as e:
            print(f"      ⚠️ Answer Correctness error: {e}")
            correctness_scores.append(None)

        # Context Recall
        try:
            cr_result = await context_recall.ascore(
                user_input=tc["question"],
                reference=tc.get("ground_truth", ""),
                retrieved_contexts=contexts
            )
            cr_score = cr_result.value if hasattr(cr_result, 'value') else cr_result
            recall_scores.append(cr_score)
        except Exception as e:
            print(f"      ⚠️ Context Recall error: {e}")
            recall_scores.append(None)

    # 计算平均分数
    def avg(lst):
        valid = [x for x in lst if x is not None]
        return sum(valid) / len(valid) if valid else None

    results = {
        "faithfulness": avg(faithfulness_scores),
        "answer_relevancy": avg(relevancy_scores),
        "answer_correctness": avg(correctness_scores),
        "context_recall": avg(recall_scores),
    }

    # 打印结果
    if results["faithfulness"] is not None:
        print(f"   ✅ faithfulness: {results['faithfulness']:.4f}")
    if results["answer_relevancy"] is not None:
        print(f"   ✅ answer_relevancy: {results['answer_relevancy']:.4f}")
    if results["answer_correctness"] is not None:
        print(f"   ✅ answer_correctness: {results['answer_correctness']:.4f}")
    if results["context_recall"] is not None:
        print(f"   ✅ context_recall: {results['context_recall']:.4f}")
    
    # Combine results
    combined_results = {
        "evaluation_date": datetime.now().isoformat(),
        "total_test_cases": len(test_cases),
        "successful_queries": sum(1 for tc in test_cases if tc.get("evaluation_success")),
        "metrics": {
            "faithfulness": float(results["faithfulness"]) if results["faithfulness"] is not None else None,
            "answer_relevancy": float(results["answer_relevancy"]) if results["answer_relevancy"] is not None else None,
            "answer_correctness": float(results["answer_correctness"]) if results["answer_correctness"] is not None else None,
            "context_recall": float(results["context_recall"]) if results["context_recall"] is not None else None,
        },
        "test_cases": test_cases,
    }
    
    # Save results
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    
    return combined_results


def run_evaluation(
    test_file: str = "test_set/golden_set_demo.json",
    output_file: str = "test_set/evaluation_results.json"
):
    """Synchronous wrapper"""
    return asyncio.run(run_full_evaluation(test_file, output_file))


if __name__ == "__main__":
    results = run_evaluation()
    
    print("\n📋 FINAL SUMMARY")
    print("-" * 40)
    for metric, value in results['metrics'].items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")