import asyncio
import json
import subprocess
from typing import Any, Optional

import requests

from verl.tools.base_tool import BaseTool


class ShoppingBenchHTTPTool(BaseTool):
    endpoint: str = ""

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        try:
            result = await asyncio.to_thread(self._request, parameters)
            return json.dumps(result, ensure_ascii=False), 0.0, {"ok": True}
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False), 0.0, {"ok": False, "error": str(exc)}

    def _request(self, parameters: dict[str, Any]):
        base_url = self.config.get("base_url", "http://127.0.0.1:5631").rstrip("/")
        session = requests.Session()
        session.trust_env = False
        response = session.get(f"{base_url}/{self.endpoint}", params=parameters, timeout=self.config.get("timeout", 60))
        response.raise_for_status()
        return response.json()


class ShoppingBenchFindProductTool(ShoppingBenchHTTPTool):
    endpoint = "find_product"


class ShoppingBenchViewProductInformationTool(ShoppingBenchHTTPTool):
    endpoint = "view_product_information"


class ShoppingBenchRecommendProductTool(BaseTool):
    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self._state = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        instance_id = await super().create(instance_id)
        self._state.setdefault(instance_id, [])
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        product_ids = parameters.get("product_ids", "")
        self._state.setdefault(instance_id, []).append(product_ids)
        return f"Having recommended the products to the user: {product_ids}.", 0.0, {"ok": True}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._state.pop(instance_id, None)


class ShoppingBenchTerminateTool(BaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        status = parameters.get("status", "")
        return f"The interaction has been completed with status: {status}", 0.0, {"ok": status == "success"}


class ShoppingBenchPythonExecuteTool(BaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters.get("code", "")
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout", 20),
            )
            payload = {
                "success": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            return json.dumps(payload, ensure_ascii=False), 0.0, {"ok": proc.returncode == 0}
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), 0.0, {"ok": False}
