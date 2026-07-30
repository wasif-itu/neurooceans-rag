"""Retrievers for external knowledge sources: web search, YouTube, PubMed, Arxiv, Wikipedia."""

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

# Wikipedia requires a User-Agent header – set BEFORE importing the library
os.environ.setdefault(
    "USER_AGENT",
    "NeuroOceansRAG/1.0 (https://github.com/wasif-itu/neurooceans-rag)",
)

import arxiv
import wikipedia
from ddgs import DDGS
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from youtube_transcript_api import YouTubeTranscriptApi

from utils.config import Settings

wikipedia.set_lang("en")


# ── Web Search ──────────────────────────────────────────────────────────────


def search_web(query: str, settings: Settings, k: int = 4) -> list[Document]:
    """Search the web using Tavily (primary) or DuckDuckGo (fallback)."""
    if settings.tavily_api_key:
        return _search_tavily(query, settings.tavily_api_key, k)
    return _search_duckduckgo(query, k)


def _search_tavily(query: str, api_key: str, k: int) -> list[Document]:
    client = TavilySearch(api_key=api_key, k=k)
    results = client.invoke(query)
    documents = []
    for result in results:
        documents.append(Document(
            page_content=result.get("content", ""),
            metadata={
                "source": result.get("url", "tavily"),
                "title": result.get("title", ""),
                "retriever": "tavily",
            },
        ))
    return documents


def _search_duckduckgo(query: str, k: int) -> list[Document]:
    client = DDGS()
    results = client.text(query, max_results=k)
    documents = []
    for result in results:
        documents.append(Document(
            page_content=result.get("body", ""),
            metadata={
                "source": result.get("href", "duckduckgo"),
                "title": result.get("title", ""),
                "retriever": "duckduckgo",
            },
        ))
    return documents


# ── YouTube Transcripts ─────────────────────────────────────────────────────


def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    import re

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def search_youtube(query_or_url: str, k: int = 3) -> list[Document]:
    """Fetch transcripts from YouTube video URLs or search for videos by query.

    If the input is a YouTube URL, fetches its transcript directly.
    Otherwise, searches DuckDuckGo for YouTube videos matching the query.
    """
    video_id = _extract_video_id(query_or_url)
    if video_id:
        return _fetch_transcript(video_id)

    # Search for YouTube videos via DuckDuckGo
    client = DDGS()
    search_query = f"site:youtube.com {query_or_url}"
    results = client.text(search_query, max_results=k)
    documents = []
    for result in results:
        vid = _extract_video_id(result.get("href", ""))
        if vid:
            docs = _fetch_transcript(vid)
            documents.extend(docs)
    return documents


def _fetch_transcript(video_id: str) -> list[Document]:
    """Fetch transcript for a single video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join(entry["text"] for entry in transcript_list)
        return [Document(
            page_content=full_text,
            metadata={
                "source": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
                "retriever": "youtube",
            },
        )]
    except Exception as error:
        print(f"Could not fetch transcript for {video_id}: {error}")
        return []


# ── PubMed ──────────────────────────────────────────────────────────────────


PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def search_pubmed(query: str, k: int = 5) -> list[Document]:
    """Search PubMed articles and return title + abstract for each result."""
    # Step 1: ESearch – get PMIDs
    search_url = (
        f"{PUBMED_BASE}esearch.fcgi?db=pubmed&retmax={k}&retmode=json&term="
        f"{urllib.parse.quote(query)}"
    )
    try:
        with urllib.request.urlopen(search_url, timeout=15) as response:
            search_data = json.loads(response.read().decode())
    except Exception as error:
        print(f"PubMed search failed: {error}")
        return []

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    # Step 2: EFetch – get article details
    fetch_url = (
        f"{PUBMED_BASE}efetch.fcgi?db=pubmed&id={','.join(id_list)}"
        f"&retmode=xml&rettype=abstract"
    )
    try:
        with urllib.request.urlopen(fetch_url, timeout=15) as response:
            xml_data = response.read().decode()
    except Exception as error:
        print(f"PubMed fetch failed: {error}")
        return []

    return _parse_pubmed_xml(xml_data)


def _parse_pubmed_xml(xml_data: str) -> list[Document]:
    """Parse PubMed XML response into Documents."""
    documents = []
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as error:
        print(f"PubMed XML parse error: {error}")
        return []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        title_elem = article_elem.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()) if title_elem is not None else ""

        abstract_elems = article_elem.findall(".//AbstractText")
        abstract_parts = []
        for elem in abstract_elems:
            label = elem.get("Label", "")
            text = "".join(elem.itertext())
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Journal info
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Authors
        authors = []
        for author_elem in article_elem.findall(".//Author"):
            last = author_elem.find("LastName")
            fore = author_elem.find("ForeName")
            if last is not None and fore is not None:
                authors.append(f"{last.text} {fore.text}")
        author_str = ", ".join(authors)

        content = f"Title: {title}\n\nAbstract: {abstract}"
        documents.append(Document(
            page_content=content,
            metadata={
                "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "authors": author_str,
                "retriever": "pubmed",
            },
        ))

    return documents


# ── Arxiv ───────────────────────────────────────────────────────────────────


def search_arxiv(query: str, k: int = 5) -> list[Document]:
    """Search Arxiv papers and return title + summary for each result."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=k, sort_by=arxiv.SortCriterion.Relevance
    )
    documents = []
    try:
        for result in client.results(search):
            content = f"Title: {result.title}\n\nSummary: {result.summary}"
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": result.entry_id,
                    "title": result.title,
                    "published": (
                        str(result.published.date()) if result.published else ""
                    ),
                    "authors": ", ".join(a.name for a in result.authors),
                    "retriever": "arxiv",
                },
            ))
    except Exception as error:
        print(f"Arxiv search failed: {error}")
    return documents


# ── Wikipedia ───────────────────────────────────────────────────────────────


def search_wikipedia(query: str, k: int = 3) -> list[Document]:
    """Search Wikipedia and return page summaries."""
    try:
        search_results = wikipedia.search(query, results=k)
    except Exception as error:
        print(f"Wikipedia search failed: {error}")
        return []

    documents = []
    for title in search_results:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            documents.append(Document(
                page_content=page.summary,
                metadata={
                    "source": page.url,
                    "title": page.title,
                    "retriever": "wikipedia",
                },
            ))
        except (wikipedia.DisambiguationError, wikipedia.PageError) as error:
            print(f"Wikipedia page '{title}' error: {error}")
            continue
        except Exception as error:
            print(f"Wikipedia page '{title}' unexpected error: {error}")
            continue

    return documents


# ── Dispatcher ──────────────────────────────────────────────────────────────


RETRIEVERS = {
    "web": search_web,
    "youtube": search_youtube,
    "pubmed": search_pubmed,
    "arxiv": search_arxiv,
    "wikipedia": search_wikipedia,
}


def retrieve_from_sources(
    query: str,
    settings: Settings,
    sources: list[str] | None = None,
    k: int = 4,
) -> list[Document]:
    """Run the selected retrievers and return merged results.

    Args:
        query: The search query.
        settings: Application settings (for API keys).
        sources: List of source names to query. Defaults to all sources.
        k: Number of results per source.

    Returns:
        A flat list of Documents from all selected sources.
    """
    if sources is None:
        sources = list(RETRIEVERS.keys())

    all_documents: list[Document] = []
    for source in sources:
        retriever = RETRIEVERS.get(source)
        if retriever is None:
            print(f"Unknown retriever: {source}")
            continue
        try:
            if source == "web":
                docs = retriever(query, settings, k=k)
            else:
                docs = retriever(query, k=k)
            all_documents.extend(docs)
            print(f"Retrieved {len(docs)} document(s) from {source}")
        except Exception as error:
            print(f"Retriever '{source}' failed: {error}")

    return all_documents