from typing import Dict, Any, List
import json 
from litellm import completion
from tau_bench.envs.airline.tools.tool_templates import TOOL_TEMPLATES
from tau_bench.utils import format_diaglogue, clean_json_content

class ExtractionAgent:
    def __init__(
        self, 
        tool_templates: Dict[str, str] = TOOL_TEMPLATES,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        wiki: str = "",
        wiki_aware: bool = False,
    ):
        self.tool_templates = tool_templates
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.wiki = wiki
        self.wiki_aware = wiki_aware

    def _build_prompt(self) -> str:
        tool_descriptions = json.dumps(self.tool_templates, indent=2)

        # add wiki info if wiki_aware is True
        wiki_section = ""
        if self.wiki_aware and self.wiki:
            wiki_section = f"""
    WORKFLOW KNOWLEDGE:
    Use this information to understand when prerequisite tools are needed:
    {self.wiki}
"""
        return f"""Extract all relevant tool calls that can help solve the user's request from the dialogue and tool output. 

    {wiki_section}Available tools:
    {tool_descriptions}

    TBC"""


    def extract_memory(self, dialogue: List[Dict[str, Any]]) -> Dict[str, Any]:
        formatted_dialogue = format_diaglogue(dialogue)
        prompt = TOOL_TEMPLATES["memory_extraction"].format(dialogue=formatted_dialogue)

        response = completion(
            model=self.model_name,
            prompt=prompt,
            max_tokens=500,
            temperature=0.0,
        )

        content = clean_json_content(response.choices[0].text)
        memory_data = json.loads(content)
        return memory_data