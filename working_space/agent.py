from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from tool import search_arxiv
from models import ResearchReport, NotablePaper
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

import re
import os 

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

class ResearchProgressCallback(BaseCallbackHandler):
    """Hooks into LangChain's event stream to show what the agent is doing."""

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Extract the string input query for the clean display"""
        print(f"\n Searching: {input_str[:120]}")

    def on_tool_end(self, output, **kwargs):
        # Count the retrieved papers 
        match = re.search(r"Found (\d+) papers", str(output))
        if match:
            print(f"Retrieved {match.group(1)} papers")
        else:
            print(f"No results for this query!")

    def on_agent_action(self, action, **kwarge):
        pass # Suppress raw action logs so that the tool hook can be cleaner 



SYSTEM_PROMPT = """You are a deep research assistant specializing in academic literature on ArXiv.

When given a research topic, follow this process strictly:

1. PLAN: Identify 3-4 distinct angles or sub-topics worth searching
2. SEARCH: Use search_arxiv for each angle separately. Use short keyword queries.
3. EVALUATE: Read abstracts. If results are weak, rephrase and search again.
4. SYNTHESIZE: Only after at least 3 searches, write your full report.

Your final report MUST use exactly this structure:

## Summary
[3-5 sentence overview of what the literature says]

## Key Findings
- [one insight per bullet]

## Notable Papers
For each important paper:
**[Title]** | [Authors] | [Year] | [One sentence on why it matters] | [URL]

## Research Gaps
- [unanswered questions in this field]

## Suggested Next Steps
- [what to read or do next]

## Stats
Total papers reviewed: [number]

Rules:
- Do NOT write the report until you have completed at least 3 searches
- Prioritize papers from the last 3 years unless older ones are foundational
- Be specific — name actual techniques, model names, benchmark results from abstracts
"""

root = "."

deep_agent = create_deep_agent(
    model = "google_genai:gemini-3.5-flash",
    tools = [search_arxiv],
    system_prompt = SYSTEM_PROMPT,
    backend = FilesystemBackend(root_dir = root, virtual_mode = True),
)

if __name__ == "__main__":
    query = input("Enter research topic:")

    for chunk in deep_agent.stream({"message": [{"role": "user", "content": "{query}"}]}):
        # chunk keys: 'agent' (model output) or tool name (tool result)
        if "deep_agent" in chunk:
            for msg in chunk["agent"].get("messages", []):
                # Tool calls to show what the agent is searching
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = tc.get("args", {})
                        query = next(iter(args.values()), "")
                        print(f"\n Searching: {str(query)[:120]}")
                # Final text response
                elif hasattr(msg, "content") and msg.content:
                    print(msg.content, end = "", flush = True)
        # show paper count
        else: 
            for tool_name, tool_state in chunk.items():
                if isinstance(tool_state, dict):
                    msgs = tool_state.get("messages", [])
                elif isinstance(tool_state, list):
                    msgs = tool_state
                else: 
                    msgs = [tool_state]

                for msg in msgs:
                    content = str(getattr(msg, "content", ""))
                    if "Found" in content: 
                        match = re.search(r"Found (\d+) paper", content)
                        if match:
                            print(f"Research {match.group(1)} papers")
                        else:
                            print("No result for this query! \n Please try again")
