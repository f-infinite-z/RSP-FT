"""
三裁判工厂：统一封装 DeepSeek / Doubao / Qwen

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

# ============================================================
# 人工批注（作者）
# ============================================================
# 如标题，只做一件事：把三个不同 API（DeepSeek/豆包/千问）封装成统一的 Judge 列表。配置环境和接口的
# 下面都是我项目阶段用的模型，结构完全可以采用，模型根据需求可以调整，基本上都是openAI范式，但是注意我这边豆包用的是单独的适配，换openAI范式的其他模型记得把豆包那段调整一致
# 然后如果用中转站的话就类似豆包那款范式，基本应该都懂，就不废话了，总之这个模块就是做模型API接口统一调用，保障项目正常运行的
# ============================================================
  - DeepSeek       : https://api.deepseek.com                              model=deepseek-v4-pro
  - Doubao-Seed-Lite: https://ark.cn-beijing.volces.com/api/v3              model=<推理接入点 endpoint_id>
  - Qwen3.7-Plus   : https://dashscope.aliyuncs.com/compatible-mode/v1     model=qwen3.7-plus

裁判超参数（在 llm_judge.py Judge._call 中统一设置）：
  temperature=0.05  top_p=0.2  max_tokens=300  response_format=json_object

用法：
  from src.judge.judge_factory import build_judges
  judges = build_judges(domain="poetry", cfg_path="configs/judges.yaml")

配置文件 configs/judges.yaml 示例见本仓库；缺省字段可用环境变量覆盖：
  DEEPSEEK_API_KEY / ARK_API_KEY / DASHSCOPE_API_KEY
"""
import os
from typing import Dict, List, Optional

from src.judge.llm_judge import Judge

# 各家 OpenAI 兼容 endpoint
ENDPOINTS = {
    "deepseek": "https://api.deepseek.com",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3/responses",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

# 环境变量兜底
ENV_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "doubao": "ARK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}

# 默认 model 名（豆包必须由用户填 endpoint_id，无通用默认）
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "doubao": "",                      # 必须填推理接入点 ID
    "qwen": "qwen3.7-plus",
}


def _make_client(base_url: str, api_key: str, name: str = ""):
    if name == "doubao":
        return _DoubaoClient(api_key, base_url)
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


class _DoubaoClient:
    """火山方舟 doubao 适配器：将 OpenAI 兼容调用转译为 Responses API 格式。"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, *, model: str, messages: list, temperature: float = 0.05,
               max_tokens: int = 300, top_p: float = 0.2,
               response_format: dict = None, **kwargs):
        import requests

        # doubao 推理层消耗 token，实际输出需额外 buffer
        max_output = max(max_tokens + 200, 500)

        api_input = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                api_input.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": content}]
                })

        body = {"model": model, "input": api_input, "max_output_tokens": max_output,
                "thinking": {"type": "disabled"}}  # 关闭深度思考加速推理
        if response_format and response_format.get("type") == "json_object":
            body["text"] = {"format": {"type": "json_object"}}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        resp = requests.post(self.base_url, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # doubao Responses API 输出顺序：reasoning → message(assistant)
        output_text = ""
        for item in data.get("output", []):
            if item.get("role") == "assistant":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        output_text += c.get("text", "")

        class FakeChoice:
            def __init__(self, text):
                self.message = type("Msg", (), {"content": text})()

        class FakeResp:
            def __init__(self, text):
                self.choices = [FakeChoice(text)]

        return FakeResp(output_text)


def build_judges(domain: str, cfg_path: Optional[str] = None,
                 cfg: Optional[Dict] = None) -> List[Judge]:
    """
    根据配置构建三裁判列表。配置结构：
      {"deepseek": {"api_key": "...", "model": "deepseek-v4-pro"},
       "doubao":   {"api_key": "...", "model": "ep-xxxxxxxx"},
       "qwen":     {"api_key": "...", "model": "qwen3.7-plus"}}
    未提供的 api_key 回退到环境变量；未提供 model 用默认（豆包除外）。
    只构建 api_key 齐全的裁判，其余跳过并告警。
    """
    if cfg is None:
        cfg = {}
        if cfg_path:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

    judges = []
    scoring_mode = cfg.get("scoring_mode", "strict")
    for name in ("deepseek", "doubao", "qwen"):
        entry = cfg.get(name, {}) or {}
        api_key = entry.get("api_key") or os.getenv(ENV_KEYS[name], "")
        model = entry.get("model") or DEFAULT_MODELS[name]
        if not api_key:
            print(f"[judge_factory] 跳过 {name}：缺少 api_key（配置或环境变量 {ENV_KEYS[name]}）")
            continue
        if not model:
            print(f"[judge_factory] 跳过 {name}：缺少 model（豆包需填推理接入点 endpoint_id）")
            continue
        client = _make_client(ENDPOINTS[name], api_key, name=name)
        judge = Judge(client, model, domain, name=name, scoring_mode=scoring_mode)
        judges.append(judge)
        print(f"[judge_factory] 已启用裁判 {name} (model={model}, mode={scoring_mode})")

    if len(judges) < 2:
        print(f"[judge_factory] 警告：仅 {len(judges)} 个裁判可用，无法做一致性检验（Kendall's W 需 >=2）")
    return judges
