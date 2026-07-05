import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TOOL_SPECS = [
    {
        "name": "find_product",
        "description": "Search for products and return up to 10 products, with each product including a product_id, shop_id, title, price, service, and sold_count.",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": 'The query used to search for products. e.g. "nike shoes", "backpack for college student".',
                },
                "page": {
                    "type": "integer",
                    "description": "Modify the parameter, ranging from 1 to 5, to get additional products.",
                },
                "shop_id": {
                    "type": "string",
                    "description": "Specify a shop using the shop_id, then search for products within that shop.",
                },
                "price": {
                    "type": "string",
                    "description": 'The price range for the products. e.g. "0-100", "100-1000", "1000-".',
                },
                "sort": {
                    "type": "string",
                    "description": "Choose one from the options listed below:\n- priceasc: Search for the products and sort by price in ascending order.\n- pricedesc: Search for the products and sort by price in descending order.\n- order: Search for the products and sort by sales volume in descending order.\n- default: Search for the products and sort by the relevance between query and product.",
                    "enum": ["priceasc", "pricedesc", "order", "default"],
                },
                "service": {
                    "type": "string",
                    "description": 'Choose one or more from the options listed below and join them with a comma (","):\n- official: Search for products and only select those that are offered with LazMall service. LazMall offers a 100% authenticity guarantee, 15-day unconditional returns, 7-day delivery, and other services.\n- freeShipping: Search for products and only select those that are offered with free shipping service.\n- COD: Search for products and only select those that are offered with cash on delivery service.\n- flashsale: Search for products and only select those that are offered with LazFlash service. LazFlash offers products with limited-time promotions, and its discounts are often significant.\n- default: Search for products without applying any other selection criteria.',
                },
            },
            "required": ["q", "page"],
        },
    },
    {
        "name": "view_product_information",
        "description": "Given a list of product_ids (unique product identifiers), fetch their corresponding information, including product descriptions, SKU options, and SPU attributes.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_ids": {
                    "type": "string",
                    "description": "A comma-separated list of product_ids (unique product identifiers).",
                },
            },
            "required": ["product_ids"],
        },
    },
    {
        "name": "recommend_product",
        "description": "Recommend the products to the user. You can use the tool only once.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_ids": {
                    "type": "string",
                    "description": "A comma-separated list of product_ids:\n1. If the user finds a single product, provide the product_id that best matches the user's requirements.\n2. If the user finds `N` products, provide `N` product_ids in the order specified by the user's requirements.\n3. If the user finds a shop selling `N` products, provide `N` product_ids in the specified order, ensuring they all come from the same shop.",
                },
            },
            "required": ["product_ids"],
        },
    },
    {
        "name": "python_execute",
        "description": "Executes Python code string.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Provide Python code to execute. The code must use print() calls to output results.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "terminate",
        "description": "Terminate the dialogue and declare the task completion status.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "The finish status of the task.",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["status"],
        },
    },
    {
        "name": "web_search",
        "description": "Search for information using the web search engine and return the search results",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": 'The query used to search for information. e.g. "PopMart latest news".',
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of results to return, ranging from 1 to 20. Default is 10.",
                },
            },
            "required": ["q"],
        },
    },
]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def tool_schema_text(
    exclude_tools: list[str] | set[str] | tuple[str, ...] | None = None,
    include_tools: list[str] | set[str] | tuple[str, ...] | None = None,
) -> str:
    excluded = set(exclude_tools or [])
    included = set(include_tools or [])
    lines = []
    enabled_tools = [
        tool
        for tool in TOOL_SPECS
        if tool["name"] not in excluded and (not included or tool["name"] in included)
    ]
    for idx, tool in enumerate(enabled_tools, 1):
        lines.append(f"{idx}. Name: {tool['name']}")
        lines.append(f"Description: {tool['description']}")
        lines.append(f"Parameters: {json.dumps(tool['parameters'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).strip()


def build_system_prompt(
    prompt_file: str | Path,
    exclude_tools: list[str] | set[str] | tuple[str, ...] | None = None,
    include_tools: list[str] | set[str] | tuple[str, ...] | None = None,
) -> str:
    prompt = resolve_project_path(prompt_file).read_text(encoding="utf-8").strip()
    return prompt.replace(
        "<|toolkit_description|>",
        tool_schema_text(exclude_tools=exclude_tools, include_tools=include_tools),
    )
