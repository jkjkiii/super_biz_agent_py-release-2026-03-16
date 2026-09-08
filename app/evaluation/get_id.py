from app.services.vector_store_manager import vector_store_manager

if not vector_store_manager.is_available():
    print("连接失败，请检查配置")
    exit(1)

# 获取 MilvusClient 实例
client = vector_store_manager.vector_store.client
collection_name = vector_store_manager.collection_name

# 使用 client.query() 查询（注意参数名：filter 或 expr）
try:
    results = client.query(
        collection_name=collection_name,
        filter="",                    # 空字符串表示查询所有
        output_fields=["id", "metadata"],
        limit=100
    )
except TypeError:
    # 如果 "filter" 参数不被接受，尝试 "expr"
    results = client.query(
        collection_name=collection_name,
        expr="",
        output_fields=["id", "metadata"],
        limit=100
    )

if not results:
    print("⚠️ 集合为空，没有文档")
else:
    print(f"共查询到 {len(results)} 条文档：")
    for item in results:
        # 从 metadata 中提取来源字段（根据你实际存储的键名调整）
        source = item.get('metadata', {}).get('_source', '未知来源')
        print(f"ID: {item['id']}, 来源: {source}")