import functools
import logging
from abc import abstractmethod
from typing import Callable
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)


class BaseRouter:
    def __init__(self):
        self.router = APIRouter()

    @abstractmethod
    def _register_routes(self) -> APIRouter:
        pass

    def get_router(self) -> APIRouter:
        return self.router

    def base_endpoint(self, func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                func_result = await func(*args, **kwargs)
                if isinstance(func_result, tuple) and len(func_result) == 2:
                    results, outer_kwargs = func_result
                else:
                    results, outer_kwargs = func_result, {}
                if isinstance(results, (StreamingResponse, FileResponse, Response)):
                    return results
                return {"results": results, **outer_kwargs}
            except ValueError as e:
                logger.error(f"Error in {func.__name__}() - {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"Error in {func.__name__}() - {str(e)}")
                raise HTTPException(status_code=500, detail={
                    "message": f"An error '{e}' occurred during {func.__name__}",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }) from e
        return wrapper
