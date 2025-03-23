from lightrag import LightRAG
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

async def initialize_rag():
    rag = LightRAG(
        working_dir="C:/github/LightRAG/test2",
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag
    


    rag.insert("Artificial Intelligence (AI) is a branch of computer science dedicated to developing systems that can simulate human intelligence.",
    "Machine Learning is a core AI technology that enables computers to learn and improve from data.",
    "Deep Learning is a subset of machine learning that uses multi-layer neural networks to handle complex problems.")

    print(rag.query("What is AI?", param=QueryParam(mode="naive")))

if __name__ == "__main__":
    asyncio.run(main())
