import sys
from pathlib import Path
from loguru import logger

from app.evaluation.test_loader import load_test_data
from app.evaluation.evaluator import RAGEvaluator

# 获取当前脚本所在目录（app/evaluation/）
SCRIPT_DIR = Path(__file__).parent
# 测试数据放在同一目录下，文件名为 test_data.json（根据你实际命名调整）
TEST_DATA_PATH = SCRIPT_DIR / "test_data.json"

def main():
    if not Path(TEST_DATA_PATH).exists():
        logger.error(f"测试数据文件不存在: {TEST_DATA_PATH}")
        sys.exit(1)
    
    test_data = load_test_data(TEST_DATA_PATH)
    logger.info(f"加载测试集，共 {len(test_data)} 条")
    
    evaluator = RAGEvaluator(test_data)
    
    # 1. Baseline（纯向量）
    baseline_metrics = evaluator.evaluate_baseline(k_values=[3, 5, 10])
    
    # 2. Optimized（混合检索 + 重排序）
    optimized_metrics = evaluator.evaluate_optimized(k_values=[3, 5, 10])
    
    # 3. 对比输出
    print("\n" + "=" * 60)
    print("评测对比")
    print("=" * 60)
    print(f"{'指标':<15} {'Baseline':>12} {'Optimized':>12} {'Δ':>12}")
    print("-" * 60)
    
    for key in baseline_metrics:
        delta = optimized_metrics[key] - baseline_metrics[key]
        print(f"{key:<15} {baseline_metrics[key]:>12.4f} {optimized_metrics[key]:>12.4f} {delta:>+12.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()