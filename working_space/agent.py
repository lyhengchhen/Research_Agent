from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from tool import search_arxiv
from models import ResearchReport, NotablePaper
from deepagents import create_deep_agent

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

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


agent = create_deep_agent(
    model = "google_genai:gemini-3.5-flash",
    backend = Filesystem
)