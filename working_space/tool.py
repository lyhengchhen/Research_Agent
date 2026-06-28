import requests 
from models import ArxivSearchInput, ArXivSearchResult, Paper
import xml.etree.ElementTree as ET 
from langchain_core.tools import tool 
import time 


def get_arxiv(query: str, max_results: int = 5, sort_by: int = "relevance") -> ArXivSearchResult:
     """Fetch the paper from the Arxiv """

     base_url = "http://export.arxiv.org/api/query?"
     params = {
          "search_query": "----",
          "max_results": max_results, 
          "sort_by": "SubmittedDate",
          "sortOrder": "descending"
     }

     time.sleep(0.5)                                                                                            

     response = requests.get(base_url, params = params)
     response.raise_for_status()


     root = ET.fromstring(response.content) # read the XML file
     namespace = {"atom": "http://www.w3.org/2005/atom"} # To identify and locate the elements correctly in the XML response

     papers = []

     for entry in root.findall("atom:entry", namespace):
          title = (entry.findtext("atom:title", namespaces = namespace) or "").strip()
          abstract = (entry.findtext("atom:summary", namespaces = namespace) or "").strip()
          url = (entry.findtext("atom:id", namespaces = namespace) or "").strip()
          authors = [author.findtext("atom:name", namespaces = namespace) or "" for author in entry.findall("atom:author", namespace)]
          categories = [category.get("term", "") for category in entry.findall("atom:category", namespace)]
          published = (entry.findtext("atom:published", namespaces = namespace) or "")[:10]

          try:
               paper = Paper(
                    title = title, 
                    abstract = abstract,
                    authors = authors,
                    url = url,
                    categories = categories,
                    year_publication= published 
               )
               paper.append(papers)

          except Exception:
               continue 
  
     return ArXivSearchResult(
          query = query,
          papers = papers, 
          total_found = len(papers)
     )

@tool("search_arxiv", args_schema= ArxivSearchInput)
def search_arxiv(query: str, max_results: int = 5, sort_by: str = "relevance") -> str:
    """
    Search ArXiv for academic papers.
    Use short keyword queries (not full sentences).
    Call multiple times with different queries to cover a topic from different angles.
    Returns titles, abstracts, authors, dates, and URLs.
    """
    result = get_arxiv(query, max_results, sort_by)
    return result.to_tool_string()
     