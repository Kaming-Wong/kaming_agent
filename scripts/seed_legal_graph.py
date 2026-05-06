"""
Neo4j 知识图谱数据导入脚本
用法: python scripts/seed_legal_graph.py

运行前确保:
  - Neo4j 已启动 (docker compose up -d neo4j)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.neo4j_utils import Neo4jClient


def main():
    print("=" * 50)
    print("法律知识图谱导入")
    print("=" * 50)

    client = Neo4jClient()

    if not client.available:
        print("❌ Neo4j 不可用，请先启动 docker compose up -d neo4j")
        sys.exit(1)

    # 初始化约束
    print("正在创建约束和索引...")
    client.init_schema()

    # 询问是否清空已有数据
    existing = client.query("MATCH (n) RETURN count(n) AS cnt")
    count = existing[0]["cnt"] if existing else 0
    if count > 0:
        confirm = input(f"知识图谱已有 {count} 个节点，是否清空重建? (y/n): ")
        if confirm.lower() == "y":
            client.drop_all()
            print("已清空")
        else:
            print("跳过，不清空")

    # 导入种子数据
    print("\n正在导入法条、概念和关系...")
    client.seed()

    # 验证
    result = client.query("MATCH (n) RETURN count(n) AS cnt, count(DISTINCT n) AS distinct_cnt")
    if result:
        print(f"\n✅ 导入完成！当前知识图谱:")
        print(f"   总节点: {result[0]['cnt']}")

    rel_result = client.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
    if rel_result:
        print(f"   总关系: {rel_result[0]['cnt']}")

    # 打印示例
    print("\n📌 示例查询: 与'试用期'相关的法条")
    examples = client.search_related("试用期", limit=3)
    for ex in examples:
        print(f"   - {ex['source']}")


if __name__ == "__main__":
    main()
