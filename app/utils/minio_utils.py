import logging
import os
from typing import Optional, List, BinaryIO

from minio import Minio
from minio.error import S3Error

from app.config import settings

logger = logging.getLogger(__name__)


class MinioClient:
    """MinIO 对象存储客户端

    用途：存储用户上传的原始文档（PDF/Word/TXT），
    提供文件下载和直接访问链接，用于前端引用展示。

    所有文件按 {session_id}/{uuid}_{filename} 组织。
    """

    def __init__(self):
        self._client: Optional[Minio] = None
        self._bucket = settings.MINIO_BUCKET

    def _get_client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self):
        """确保存储桶存在，不存在则创建"""
        client = self._client
        try:
            if not client.bucket_exists(self._bucket):
                client.make_bucket(self._bucket)
                logger.info(f"Created MinIO bucket: {self._bucket}")
            else:
                logger.info(f"MinIO bucket exists: {self._bucket}")
        except S3Error as e:
            logger.error(f"MinIO bucket error: {e}")
            raise

    def upload_stream(self, object_name: str, data: BinaryIO, length: int,
                      content_type: Optional[str] = None) -> str:
        """流式上传文件到 MinIO

        Args:
            object_name: 对象路径（如 session_id/uuid_filename.pdf）
            data: 二进制文件流
            length: 数据长度（字节）
            content_type: MIME 类型
        Returns:
            object_name（用于后续 URL 拼接）
        """
        client = self._get_client()
        try:
            client.put_object(
                self._bucket,
                object_name,
                data,
                length,
                content_type=content_type,
            )
            logger.info(f"Uploaded stream to MinIO: {object_name} ({length} bytes)")
            return object_name
        except S3Error as e:
            logger.error(f"MinIO stream upload failed: {e}")
            raise

    def download(self, object_name: str, file_path: str):
        """从 MinIO 下载到本地文件"""
        client = self._get_client()
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            client.fget_object(self._bucket, object_name, file_path)
            logger.info(f"Downloaded from MinIO: {object_name} → {file_path}")
        except S3Error as e:
            logger.error(f"MinIO download failed: {e}")
            raise

    def get_file(self, object_name: str) -> bytes:
        """直接读取文件内容（小文件用）"""
        client = self._get_client()
        try:
            response = client.get_object(self._bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"MinIO get object failed: {e}")
            raise

    def list_files(self, prefix: str = "") -> List[dict]:
        """列出指定前缀下的所有文件

        prefix 通常是 session_id，用于查看某会话上传的所有文件。
        """
        client = self._get_client()
        try:
            objects = client.list_objects(self._bucket, prefix=prefix, recursive=True)
            result = []
            for obj in objects:
                result.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else "",
                    "content_type": obj.content_type or "",
                })
            return result
        except S3Error as e:
            logger.error(f"MinIO list failed: {e}")
            return []

    def delete_prefix(self, prefix: str):
        """删除指定前缀的所有文件（删除文档时同时清理 MinIO）"""
        client = self._get_client()
        try:
            objects = client.list_objects(self._bucket, prefix=prefix, recursive=True)
            names = [obj.object_name for obj in objects]
            if names:
                errors = client.remove_objects(self._bucket, names)
                for err in errors:
                    logger.error(f"MinIO delete error: {err}")
            logger.info(f"Deleted {len(names)} files from MinIO prefix: {prefix}")
        except S3Error as e:
            logger.error(f"MinIO prefix delete failed: {e}")

    def get_url(self, object_name: str) -> str:
        """生成 MinIO 文件的 HTTP 访问链接（用于前端展示）"""
        endpoint = settings.MINIO_ENDPOINT
        return f"http://{endpoint}/{self._bucket}/{object_name}"
