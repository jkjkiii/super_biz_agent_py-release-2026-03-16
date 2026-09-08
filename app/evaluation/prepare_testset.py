# evaluation/prepare_testset.py
from datasets import Dataset
import json
from pathlib import Path

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent
TEST_DATA_PATH = SCRIPT_DIR / "test_data.json"
SAVE_DIR = SCRIPT_DIR / "ragas_testset"

def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["queries"]

def to_ragas_dataset(test_data):
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for item in test_data:
        ragas_data["question"].append(item["query"])
        ragas_data["answer"].append("")  # 暂时留空，后续填充
        ragas_data["contexts"].append([])  # 暂时留空，后续填充
        ragas_data["ground_truth"].append("")  # 暂时留空，后续填充
        
    return Dataset.from_dict(ragas_data)

if __name__ == "__main__":
    if not TEST_DATA_PATH.exists():
        print(f"错误：测试数据文件不存在：{TEST_DATA_PATH}")
        exit(1)
    
    test_data = load_test_data(str(TEST_DATA_PATH))
    dataset = to_ragas_dataset(test_data)
    
    # 保存到本地磁盘
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(SAVE_DIR))
    
    print(f"✅ RAGAS 测试集已保存至：{SAVE_DIR}")
    print(f"📊 共 {len(dataset)} 条数据")