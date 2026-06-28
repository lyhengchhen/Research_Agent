from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal

# User Query → [ArxivSearchInput] → ArXiv API → [Paper] → [ArxivSearchResult] → Claude → [ResearchReport]

class ArxivSearchInput(BaseModel):
    """Input schema for the ArXiv search tool.
    LangChain reads this to auto-generate the tool's argument spec."""

    query: str = Field(
        ...,
        description="Short keyword query. E.g. 'retrieval augmented generation'. Not a full sentence.",
        min_length=2,
        max_length=200
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of papers to return. Keep between 3-8."
    )
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(
        default="relevance",
        description="Sort order for results."
    )


class Paper(BaseModel):
    """Retrieve the metadata about a paper from the Arxiv"""
    title: str 
    year_publication: str # ArXiv give "2024-02-02"
    authors: list[str]
    url : str 
    abstract: str
    catergories: list[str]

    @field_validator("url")
    @classmethod
    def ensure_correct_url(cls, v:str) -> str: 
        if not v.startswith("http"): # if it does not start with http, the system raise error
            raise ValueError(f"Invalid URL: {v}")
        return v 

    def truncate_long_abstract(cls, v: str) -> str:
        return v[:1200] + "..." if len(v) > 1200 else v # avoid burning the tokens for nothing

    def to_summary(self) -> str:
        authors_short = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_short += "et al."
        return (
                f"📄 {self.title}\n"
                f"   {authors_short} ({self.year_publication})\n"
                f"   {self.url}"
                )

class ArXivSearchResult(BaseModel):
    """Wrap a list of papers return from one search call.""" #Great for context engineering

    query: str
    total_papers: int 
    papers: list[Paper]


    def is_empty(self) -> bool:
        return len(self.papers) == 0 

    def to_tool_string(self):
        """Serialize the result into string so that the model can understand and read"""
        if self.is_empty:
            return (f"No paper found! Please try using different keywords")
            
        lines = [f"Found {self.total_papers} papers for '{self.query}:\n'" ]

        for i, paper in enumerate(self.papers, 1):
            lines.append(
                f"{i}. Title: {paper.title}\n"
                f"   Authors: {paper.authors_short}\n"
                f"   Published: {paper.year_publication}\n"
                f"   Categories: {', '.join(paper.categories)}\n"
                f"   URL: {paper.url}\n"
                f"   Abstract: {paper.abstract}\n"
            )

            return "\n".join(lines)

class NotablePaper(BaseModel):
    title: str
    authors: list[str]
    year: str
    significance: str = Field(description = "Provide one sentence on why it is significane")
    url: str 


class ResearchReport(BaseModel):
    """The structured final output of the agent"""
    query: str
    summary: str = Field(..., description = "3 to 5 sentence overview of the paper")
    key_findigs: list[str] = Field(description = "List down the 3 to 5 insights in bulletpoint")
    notable_paper: list[NotablePaper]
    research_gaps: list[str]
    suggested_next_steps: list[str]
    total_paper_reviewed: int

    def to_markdown(self) -> str: 
        """Generate the report as cleaned markdown"""

        lines = [
            f"# Research report: {self.query}\n",
            f"# Total paper reviewed: {self.total_paper_reviewed}",
            f"## Summary\n",
            self.summary + "\n",
        ]

        lines.append(f"## Key finding:\n")
        for finding in self.key_findigs:
            lines.append(f"-{finding}")

        lines.append(f"## Notable papers:\n")
        for notable_paper in self.notable_papers: 
            lines.append(f"- Title: {notable_paper.title}\n"
                         f"- Author: {notable_paper.authors}\n"
                         f"- Year: {notable_paper.year}\n"
                         f"- Significance level: {notable_paper.significance}\n"
                         f"- URL: {notable_paper.url}")

        lines.append(f"## Research gaps:\n")
        for research_gap in self.research_gaps: 
            lines.append(f"-{research_gap}")

        lines.append(f"## Next Step:\n")
        for next_step in self.suggested_next_steps:
            lines.append(f"-{next_step}")

        return "\n".join(lines)


if __name__ == "__main__": 

    user_input = {"query": "What is a transformer",
                  "max_results": 5,
                  "sort_by": "relevance"}
    try:
        test = ArxivSearchInput.model_validate(user_input)
        print(test.query)

    except ValidationError as e:
        print(e) 

