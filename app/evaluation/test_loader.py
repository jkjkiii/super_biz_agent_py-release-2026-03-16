import json
from typing import List, Dict, Any

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载测试集，返回列表，每个元素为：
    {
        "query": str,
        "relevant_ids": List[str],
        "relevance_scores": Dict[str, int]   # 可选
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 兼容不同格式
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    return data