import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir="C:/github/LightRAG/shawntesting",
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    # Insert text
    # rag.insert(
    #     ""
    #            )
    with open('./book.txt') as f:
        rag.insert(f.read())
   

    response=rag.query(
        "Who are the main characters and what are their relationships?",
        param=QueryParam(mode="global")
    )
    print(response)

if __name__ == "__main__":
    main()

print("done")