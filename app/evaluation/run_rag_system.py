# evaluation/run_rag_system.py
from pathlib import Path
from datasets import load_from_disk
from loguru import logger
from app.rag.retrievers import hybrid_retriever
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from app.config import config

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "ragas_testset"
OUTPUT_PATH = SCRIPT_DIR / "ragas_results"

def generate_answer(query: str, contexts: list) -> str:
    """调用 LLM 生成答案"""
    if not contexts:
        return "未检索到相关内容，无法回答。"
    
    llm = ChatTongyi(
        model="qwen-max",
        dashscope_api_key=config.dashscope_api_key
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "请基于以下参考资料回答用户的问题。若参考资料不足以回答，请明确说明。\n参考资料：\n{context}"),
        ("user", "{question}")
    ])
    chain = prompt | llm
    context_str = "\n\n".join(contexts[:3])  # 限制上下文长度
    response = chain.invoke({"question": query, "context": context_str})
    return response.content

def run_rag_on_dataset():
    """主流程：加载测试集，调用 RAG 系统，填充答案和上下文，保存结果"""
    
    if not DATASET_PATH.exists():
        logger.error(f"数据集不存在：{DATASET_PATH}")
        logger.info("请先运行 prepare_testset.py 生成测试集")
        return
    
    dataset = load_from_disk(str(DATASET_PATH))
    logger.info(f"加载数据集，共 {len(dataset)} 条")
    
    questions = dataset["question"]
    answers = []
    contexts = []
    
    # 处理 ground_truth：如果不存在则创建
    if "ground_truth" in dataset.column_names:
        ground_truths = dataset["ground_truth"]
    else:
        ground_truths = [""] * len(questions)
        logger.warning("数据集中没有 ground_truth 字段，使用空字符串占位")
    
    for idx, q in enumerate(questions):
        logger.info(f"处理第 {idx+1}/{len(questions)} 条: {q[:30]}...")
        
        try:
            docs = hybrid_retriever.retrieve(q, top_k=5)
            retrieved_contexts = [doc.page_content for doc in docs]
            contexts.append(retrieved_contexts)
            
            if retrieved_contexts:
                answer = generate_answer(q, retrieved_contexts)
            else:
                answer = "未检索到相关内容。"
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            contexts.append([])
            answers.append("")
    
    # 更新数据集（保留原有列，添加 answer 和 contexts）
    updated_dataset = dataset.add_column("answer", answers)
    updated_dataset = updated_dataset.add_column("contexts", contexts)
    
    # 如果原来没有 ground_truth，补上
    if "ground_truth" not in dataset.column_names:
        updated_dataset = updated_dataset.add_column("ground_truth", ground_truths)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    updated_dataset.save_to_disk(str(OUTPUT_PATH))
    logger.success(f"结果已保存至：{OUTPUT_PATH}")
    logger.info(f"共处理 {len(updated_dataset)} 条数据")

def main():
    logger.info("=" * 50)
    logger.info("开始运行 RAG 系统生成答案和上下文...")
    logger.info("=" * 50)
    run_rag_on_dataset()
    logger.info("=" * 50)
    logger.info("RAG 系统运行完成！")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()