from typing import Optional, Any
from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    status: int = Field(..., description="200=成功")
    message: str = Field(..., description="响应信息")
    data: Optional[Any] = Field(None, description="业务数据")
    page: Optional[Any] = Field(None, description="分页信息")
    model_config = {"from_attributes": True, "arbitrary_types_allowed": True}
