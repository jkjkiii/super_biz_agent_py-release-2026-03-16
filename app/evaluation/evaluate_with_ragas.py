# app/evaluation/evaluate_with_ragas.py
import sys
from pathlib import Path
from unittest.mock import MagicMock

# 模拟 vertexai 模块，防止 RAGAS 导入失败
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_google_vertexai'] = MagicMock()
sys.modules['google.cloud.aiplatform'] = MagicMock()

# 现在导入 ragas（使用新的导入方式避免警告）
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import llm_factory
from openai import OpenAI
from datasets import load_from_disk
from app.config import config

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "ragas_testset"  # 这是 prepare_testset.py 保存的目录

def evaluate_rag_system():
    # 使用 llm_factory 替代 LangchainLLMWrapper（避免废弃警告）
    client = OpenAI(
        api_key=config.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    evaluator_llm = llm_factory(
        model="qwen-max",
        client=client,
        temperature=0
    )

    # 加载数据集
    if not DATASET_PATH.exists():
        print(f"错误：数据集目录不存在：{DATASET_PATH}")
        print("请先运行 prepare_testset.py 生成测试集，再运行 run_rag_system.py 填充答案和上下文。")
        return

    dataset = load_from_disk(str(DATASET_PATH))

    # 检查数据集是否包含必要的字段（非空）
    if all(not c for c in dataset["contexts"]) or all(not a for a in dataset["answer"]):
        print("警告：数据集中 contexts 或 answer 字段为空，RAGAS 评估结果可能无效。")
        print("请先运行 run_rag_system.py 填充这些字段。")

    # 执行评估
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm
    )

    print("=" * 50)
    print("RAGAS 评估结果")
    print("=" * 50)
    for metric, score in result.items():
        print(f"{metric}: {score:.4f}")

    # 保存结果
    result.to_pandas().to_csv(SCRIPT_DIR / "ragas_scores.csv", index=False)
    print(f"结果已保存至 {SCRIPT_DIR / 'ragas_scores.csv'}")

    return result

if __name__ == "__main__":
    evaluate_rag_system()