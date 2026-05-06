import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.MYSQL_URI, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_db():
    """迁移已有表结构：新增列（create_all 不会改已有表）"""
    migrations = [
        ("ALTER TABLE conversations ADD COLUMN `references` TEXT",
         "conversations.references"),
        ("ALTER TABLE conversations ADD COLUMN minio_url VARCHAR(512) DEFAULT ''",
         "conversations.minio_url"),
    ]
    db = SessionLocal()
    try:
        for sql, check in migrations:
            if check:
                # 先检查列是否已存在
                col = check.split(".")[-1]
                tbl = check.split(".")[0]
                try:
                    result = db.execute(text(
                        f"SELECT COUNT(*) AS cnt FROM information_schema.COLUMNS "
                        f"WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl AND COLUMN_NAME = :col"
                    ), {"db": settings.MYSQL_DB, "tbl": tbl, "col": col})
                    if result.scalar() > 0:
                        continue  # 已存在，跳过
                except Exception:
                    pass  # information_schema 查询失败，直接尝试 ALTER

            try:
                db.execute(text(sql))
                db.commit()
                logger.info(f"DB migration: {sql[:60]}...")
            except Exception as e:
                db.rollback()
                # 列已存在等错误忽略
                if "Duplicate column" in str(e) or "already exists" in str(e).lower():
                    continue
                logger.warning(f"DB migration skipped ({e})")
        logger.info("Database migration completed")
    except Exception as e:
        logger.warning(f"Database migration failed: {e}")
    finally:
        db.close()
