import logging
from typing import Optional, List, Dict, Any

from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import Neo4jError

from app.config import settings

logger = logging.getLogger(__name__)

# ── 约束和索引（Cypher） ──
CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:LawArticle) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
]

# ── 预置法条关系数据 ──
SEED_ARTICLES = [
    {
        "name": "中华人民共和国劳动法 第三十六条",
        "text": "国家实行劳动者每日工作时间不超过八小时、平均每周工作时间不超过四十四小时的工时制度。",
        "category": "劳动工时",
    },
    {
        "name": "中华人民共和国劳动法 第四十四条",
        "text": "有下列情形之一的，用人单位应当按照下列标准支付高于劳动者正常工作时间工资的工资报酬：（一）安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬；（二）休息日安排劳动者工作又不能安排补休的，支付不低于工资的百分之二百的工资报酬；（三）法定休假日安排劳动者工作的，支付不低于工资的百分之三百的工资报酬。",
        "category": "劳动报酬",
    },
    {
        "name": "中华人民共和国劳动合同法 第三十七条",
        "text": "劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。劳动者在试用期内提前三日通知用人单位，可以解除劳动合同。",
        "category": "劳动合同解除",
    },
    {
        "name": "中华人民共和国劳动合同法 第四十七条",
        "text": "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。",
        "category": "经济补偿",
    },
    {
        "name": "中华人民共和国劳动合同法 第八十七条",
        "text": "用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
        "category": "赔偿金",
    },
    {
        "name": "中华人民共和国劳动合同法 第三十九条",
        "text": "劳动者有下列情形之一的，用人单位可以解除劳动合同：（一）在试用期间被证明不符合录用条件的；（二）严重违反用人单位的规章制度的；（三）严重失职，营私舞弊，给用人单位造成重大损害的；（四）劳动者同时与其他用人单位建立劳动关系，对完成本单位的工作任务造成严重影响，或者经用人单位提出，拒不改正的；（五）因本法第二十六条第一款第一项规定的情形致使劳动合同无效的；（六）被依法追究刑事责任的。",
        "category": "劳动合同解除",
    },
    {
        "name": "中华人民共和国消费者权益保护法 第二十五条",
        "text": "经营者采用网络、电视、电话、邮购等方式销售商品，消费者有权自收到商品之日起七日内退货，且无需说明理由。",
        "category": "消费者权益",
    },
    {
        "name": "中华人民共和国民法典 第一千一百七十九条",
        "text": "侵害他人造成人身损害的，应当赔偿医疗费、护理费、交通费、营养费、住院伙食补助费等为治疗和康复支出的合理费用，以及因误工减少的收入。",
        "category": "侵权责任",
    },
]

SEED_CONCEPTS = [
    {"name": "试用期", "description": "劳动合同期限三个月以上不满一年的，试用期不得超过一个月；一年以上不满三年的，试用期不得超过二个月；三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月。"},
    {"name": "经济补偿", "description": "用人单位在特定条件下解除或终止劳动合同时，依法支付给劳动者的补偿金。"},
    {"name": "加班费", "description": "用人单位安排劳动者在法定标准工作时间以外工作的，应当依法支付的工资报酬。"},
    {"name": "违法解除", "description": "用人单位违反劳动合同法规定解除或者终止劳动合同，应当支付赔偿金。"},
    {"name": "消费者权益", "description": "消费者在购买、使用商品和接受服务时享有的合法权益。"},
    {"name": "工伤赔偿", "description": "劳动者在工作中因事故受到伤害，依法获得的医疗救治和经济补偿。"},
]

SEED_RELATIONS = [
    # 法条 → 法条（引用）
    ("中华人民共和国劳动合同法 第八十七条", "REFERENCES", "中华人民共和国劳动合同法 第四十七条"),
    ("中华人民共和国劳动法 第四十四条", "REFERENCES", "中华人民共和国劳动法 第三十六条"),
    ("中华人民共和国劳动合同法 第三十七条", "REFERENCES", "中华人民共和国劳动合同法 第三十九条"),
    # 法条 → 概念（属于）
    ("中华人民共和国劳动法 第四十四条", "BELONGS_TO", "加班费"),
    ("中华人民共和国劳动法 第三十六条", "BELONGS_TO", "加班费"),
    ("中华人民共和国劳动合同法 第四十七条", "BELONGS_TO", "经济补偿"),
    ("中华人民共和国劳动合同法 第八十七条", "BELONGS_TO", "违法解除"),
    ("中华人民共和国劳动合同法 第三十七条", "BELONGS_TO", "试用期"),
    ("中华人民共和国劳动合同法 第三十九条", "BELONGS_TO", "试用期"),
    ("中华人民共和国消费者权益保护法 第二十五条", "BELONGS_TO", "消费者权益"),
    ("中华人民共和国民法典 第一千一百七十九条", "BELONGS_TO", "工伤赔偿"),
    # 概念 → 概念（关联）
    ("试用期", "RELATES_TO", "违法解除"),
    ("试用期", "RELATES_TO", "经济补偿"),
    ("违法解除", "RELATES_TO", "经济补偿"),
    ("加班费", "RELATES_TO", "经济补偿"),
]


class Neo4jClient:
    """Neo4j 知识图谱客户端

    管理法律实体节点和关系，支持 Cypher 查询和多跳推理。
    用于增强 RAG：向量检索 + 图检索双通道。
    """

    def __init__(self):
        self._driver: Optional[Driver] = None

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=5,
                max_connection_lifetime=3600,
            )
            logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
        return self._driver

    @property
    def available(self) -> bool:
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j unavailable: {e}")
            return False

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── 初始化 ──

    def init_schema(self):
        """创建约束和索引（幂等）"""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                for cypher in CONSTRAINTS:
                    session.run(cypher)
            logger.info("Neo4j schema initialized")
        except Neo4jError as e:
            logger.error(f"Neo4j schema init failed: {e}")
            raise

    # ── 种子数据 ──

    def seed(self):
        """导入预置法条、概念和关系"""
        driver = self._get_driver()
        with driver.session() as session:
            # 创建法条节点
            for art in SEED_ARTICLES:
                session.run(
                    "MERGE (a:LawArticle {name: $name}) "
                    "SET a.text = $text, a.category = $category",
                    name=art["name"], text=art["text"], category=art["category"],
                )
            logger.info(f"Seeded {len(SEED_ARTICLES)} law articles")

            # 创建概念节点
            for c in SEED_CONCEPTS:
                session.run(
                    "MERGE (c:Concept {name: $name}) SET c.description = $desc",
                    name=c["name"], desc=c["description"],
                )
            logger.info(f"Seeded {len(SEED_CONCEPTS)} concepts")

            # 创建关系
            for src, rel, tgt in SEED_RELATIONS:
                session.run(
                    f"MATCH (a {{name: $src}}), (b {{name: $tgt}}) "
                    f"MERGE (a)-[:{rel}]->(b)",
                    src=src, tgt=tgt,
                )
            logger.info(f"Seeded {len(SEED_RELATIONS)} relations")

    def drop_all(self):
        """清空所有数据（谨慎使用）"""
        driver = self._get_driver()
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j cleared all data")

    # ── 查询 ──

    def query(self, cypher: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """执行任意 Cypher 查询"""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except Neo4jError as e:
            logger.error(f"Neo4j query failed: {e}")
            return []

    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """关键字搜索法条和概念（模糊匹配）"""
        cypher = """
            MATCH (a:LawArticle)
            WHERE a.name CONTAINS $kw OR a.text CONTAINS $kw
            RETURN a.name AS source, a.text AS text, 'LawArticle' AS label, 1.0 AS score
            LIMIT $limit
        """
        return self.query(cypher, {"kw": keyword, "limit": limit})

    def search_related(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """多跳关联查询：输入关键字 → 法条 → 关联法条/概念

        用例：用户问"试用期被辞退" → 匹配到「试用期」概念
        → 一跳找到「劳动合同法 第三十七条」「第三十九条」
        → 二跳找到「第四十七条」「第八十七条」（经济补偿/赔偿金）
        """
        cypher = """
            // 先匹配概念或法条
            MATCH (start)
            WHERE start.name CONTAINS $kw
            // 一跳：关联的法条和概念
            OPTIONAL MATCH (start)-[:BELONGS_TO|REFERENCES|RELATES_TO*1..2]-(related)
            WHERE related:LawArticle OR related:Concept
            // 返回法条节点
            WITH DISTINCT related AS node
            WHERE node:LawArticle
            RETURN node.name AS source, node.text AS text,
                   node.category AS category, 'graph' AS source_type, 0.9 AS score
            LIMIT $limit
        """
        return self.query(cypher, {"kw": keyword, "limit": limit})

    def get_article_detail(self, name: str) -> Optional[Dict]:
        """获取单个法条详情"""
        results = self.query(
            "MATCH (a:LawArticle {name: $name}) RETURN a.name AS name, a.text AS text, a.category AS category",
            {"name": name},
        )
        return results[0] if results else None

    def get_relations(self, name: str) -> List[Dict]:
        """获取某个节点的所有关系"""
        return self.query(
            """MATCH (n {name: $name})-[r]->(m)
               RETURN type(r) AS relation, m.name AS target, labels(m)[0] AS target_label""",
            {"name": name},
        )

    # ── 文档实体抽取后写入 ──

    def link_document_to_articles(self, doc_name: str, session_id: str, matched_articles: List[str]):
        """将上传文档与图谱中的法条关联

        Args:
            doc_name: 文档名称（唯一标识）
            session_id: 上传者会话 ID
            matched_articles: LLM 从文档中抽取出的相关法条名称列表
        """
        driver = self._get_driver()
        with driver.session() as session:
            # 创建文档节点
            session.run(
                "MERGE (d:Document {name: $name}) SET d.session_id = $sid",
                name=doc_name, sid=session_id,
            )

            # 关联到法条
            for art in matched_articles:
                session.run(
                    "MATCH (d:Document {name: $doc}), (a:LawArticle {name: $art}) "
                    "MERGE (d)-[:MENTIONED_IN]->(a)",
                    doc=doc_name, art=art,
                )
            logger.info(f"Linked document '{doc_name}' to {len(matched_articles)} articles")


# 全局单例
_neo4j: Optional[Neo4jClient] = None


def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j
