import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# 设置 Neo4j 环境变量
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "shawntesting"  # 替换为你设置的密码

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./shawntesting/dockertesting"

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage"  ,# 覆盖默认的 KG 存储
        # log_level="INFO",
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r") as f:
        rag.insert(f.read())

    # with open('./book.txt') as f:
    #     rag.insert(f.read())
   

    response=rag.query(
        "Who are the main characters and what are their relationships?",
        param=QueryParam(mode="global")
    )
    print(response)

if __name__ == "__main__":
    main()

print("done")