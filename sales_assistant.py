#!/usr/bin/env python3
"""
Task 9: Streamlit Research AI Assistant
Interactive web interface for the LangGraph Research Assistant
"""

import streamlit as st
import os
import time
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langchain_aws import ChatBedrock

from ddgs import DDGS
from datetime import datetime
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Page config
st.set_page_config(
    page_title="Product Pitch Research Assistant",
    page_icon="📣",
    layout="wide"
)

# Initialize LLM
@st.cache_resource
def get_llm():
    """Use Amazon Bedrock-hosted Claude model."""
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        temperature=0.7,
        max_tokens=2048
    )

# State definition
class ResearchState(TypedDict):
    topic: str
    company_name: str
    country: str
    website_url: str
    research_questions: List[str]
    search_queries: List[str]
    search_results: List[str]
    iteration: int
    max_iterations: int
    quality_score: float
    quality_threshold: float
    final_report: str
    status: str
    current_node: str
    company_background: str
    products_services: str
    primary_decision_makers: str
    points_of_contact: str
    recent_news: str

# Node functions
def input_processor_node(state: ResearchState):
    """Process and validate input topic"""
    llm = get_llm()
    
    # Update UI status
    state["current_node"] = "input_processor"
    
    company = state.get("company_name") or state.get("topic", "").strip()
    country = state.get("country", "").strip()
    subject = f"{company} ({country})" if country else company
    
    prompt = f"""Normalize this company name to the official form (no extra words): '{company}'. Return only the name."""
    response = llm.invoke(prompt)
    normalized = response.content.strip() if response and hasattr(response, 'content') else company
    
    def find_website(name: str, country_hint: str) -> str:
        queries = [
            f"{name} official site {country_hint}",
            f"{name} website {country_hint}",
            f"{name} homepage {country_hint}"
        ]
        seen = set()
        candidates = []
        name_key = re.sub(r"[^a-z0-9]", "", (name or "").lower())
        exclude_hosts = (
            "linkedin.com", "wikipedia.org", "facebook.com", "twitter.com", "x.com",
            "crunchbase.com", "bloomberg.com", "news.yahoo.com", "medium.com",
            "github.com", "gitlab.com", "angel.co", "glassdoor.com", "g2.com",
            "indeed.com", "yahoo.com", "google.com"
        )
        ddgs = DDGS()
        for q in queries:
            try:
                for r in ddgs.text(q, max_results=8):
                    href = r.get("href") or r.get("url") or ""
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    try:
                        u = urlparse(href)
                    except Exception:
                        continue
                    host = (u.netloc or "").lower()
                    if not host or any(bad in host for bad in exclude_hosts):
                        continue
                    host_key = re.sub(r"[^a-z0-9]", "", host.replace("www.", ""))
                    score = 0
                    if name_key and name_key in host_key:
                        score += 3
                    if country_hint:
                        score += 1
                    candidates.append((score, host, u.scheme or "https"))
            except Exception:
                continue
        if not candidates:
            return ""
        candidates.sort(reverse=True)
        best = candidates[0]
        scheme = best[2]
        host = best[1]
        return f"{scheme}://{host}"
    
    website = find_website(normalized, country)
    
    return {
        "topic": subject,
        "company_name": normalized or company,
        "country": country,
        "website_url": website,
        "status": "topic_processed",
        "current_node": "input_processor"
    }

def question_generator_node(state: ResearchState):
    """Generate research questions"""
    llm = get_llm()
    state["current_node"] = "question_generator"
    
    # Target only missing fields to focus subsequent searches
    def missing(v: str) -> bool:
        return not v or v.strip().lower() == "not found"
    categories = []
    if missing(state.get("company_background", "")): categories.append("Company background")
    if missing(state.get("products_services", "")): categories.append("Key products or services")
    if missing(state.get("primary_decision_makers", "")): categories.append("Primary decision makers (with titles)")
    if missing(state.get("points_of_contact", "")): categories.append("Points of contact (emails, LinkedIn pages, or roles)")
    if missing(state.get("recent_news", "")): categories.append("Recent news or strategic moves")
    if not categories:
        categories = ["Recent news or strategic moves"]
    
    subject = (state.get("company_name") or state.get("topic", "")).strip()
    subject_full = f"{subject} in {state.get('country','').strip()}".strip()
    prompt = f"""Generate 1-2 specific, web-searchable queries about '{subject_full}' for each of these categories:
{chr(10).join(['- ' + c for c in categories])}
Return only the queries, one per line, maximum 10 lines, no bullets or numbering."""
    
    response = llm.invoke(prompt)
    questions = response.content.strip().split('\n')
    questions = [q.strip('-• ').strip() for q in questions if q.strip()][:10]
    existing = set(state.get("research_questions", []) + state.get("search_queries", []))
    questions = [q for q in questions if q not in existing]
    
    all_questions = state.get("research_questions", []) + questions
    
    return {
        "research_questions": all_questions,
        "status": "questions_generated",
        "current_node": "question_generator"
    }

def search_tool_node(state: ResearchState):
    """Search for information using DuckDuckGo"""
    state["current_node"] = "search_tool"
    
    search_results = state.get("search_results", [])
    search_queries = state.get("search_queries", [])
    
    for question in state["research_questions"]:
        if question not in search_queries:
            try:
                ddgs = DDGS()
                results = ddgs.text(question, max_results=5)
                
                for result in results:
                    title = result.get('title', '')
                    body = result.get('body', '')
                    search_results.append(f"{title}: {body}")
                
                search_queries.append(question)
                
            except Exception as e:
                st.error(f"Search error: {e}")
    
    site = state.get("website_url", "").strip()
    if site:
        try:
            to_scrape = [
                site,
                urljoin(site + '/', 'about'),
                urljoin(site + '/', 'company'),
                urljoin(site + '/', 'team'),
                urljoin(site + '/', 'leadership'),
                urljoin(site + '/', 'contact'),
                urljoin(site + '/', 'news'),
                urljoin(site + '/', 'press'),
                urljoin(site + '/', 'products'),
                urljoin(site + '/', 'solutions'),
                urljoin(site + '/', 'services'),
            ]
            scraped_seen = set(q.replace('SCRAPE:', '') for q in search_queries if q.startswith('SCRAPE:'))
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"}
            for u in to_scrape:
                if u in scraped_seen:
                    continue
                try:
                    resp = requests.get(u, headers=headers, timeout=10)
                    ctype = resp.headers.get('Content-Type', '')
                    if resp.status_code != 200 or 'text/html' not in ctype:
                        search_queries.append(f"SCRAPE:{u}")
                        continue
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for tag in soup(['script', 'style', 'noscript']):
                        tag.extract()
                    text = ' '.join(soup.stripped_strings)
                    text = re.sub(r"\s+", " ", text)
                    if text:
                        snippet = text[:1500]
                        path = urlparse(u).path or '/'
                        search_results.append(f"{site} [{path}]: {snippet}")
                    search_queries.append(f"SCRAPE:{u}")
                except Exception:
                    search_queries.append(f"SCRAPE:{u}")
                    continue
        except Exception:
            pass
    
    return {
        "search_results": search_results,
        "search_queries": search_queries,
        "iteration": state["iteration"] + 1,
        "status": "search_completed",
        "current_node": "search_tool"
    }

def analyzer_node(state: ResearchState):
    """Analyze search results and extract key findings"""
    llm = get_llm()
    state["current_node"] = "analyzer"
    
    if not state["search_results"]:
        return {
            "company_background": "Not found",
            "products_services": "Not found",
            "primary_decision_makers": "Not found",
            "points_of_contact": "Not found",
            "recent_news": "Not found",
            "quality_score": 0.0,
        }
    
    # Use the most recent results so new iterations improve coverage
    results_text = "\n".join(state["search_results"][-20:])
    
    subj = (state.get("company_name") or state.get("topic", "")).strip()
    loc = state.get("country", "").strip()
    prompt = f"""Using the following snippets about '{subj}' in '{loc}', extract concise information for a product pitch.

Snippets:
{results_text}

Return a single JSON object with EXACTLY these keys:
- company_background
- products_services
- primary_decision_makers
- points_of_contact
- recent_news

Guidelines:
- Keep each value 1-3 sentences or short bullet-like lines separated by '; '.
- Include titles with names when possible.
- Prefer official sources and up-to-date info.
- If unknown, use 'Not found'.
"""
    
    response = llm.invoke(prompt)
    raw = response.content.strip()
    
    # Attempt to clean code fences and extract JSON
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1]
        else:
            raw = raw.replace("```json", "").replace("```", "")
        raw = raw.strip()
    if '{' in raw and '}' in raw and not raw.strip().startswith('{'):
        raw = raw[raw.find('{'): raw.rfind('}') + 1]

    data = {}
    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: naive parsing
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        def pull(prefix):
            for ln in lines:
                if ln.lower().startswith(prefix):
                    return ln.split(':', 1)[-1].strip()
            return "Not found"
        data = {
            "company_background": pull("company background"),
            "products_services": pull("products") if pull("products") != "Not found" else pull("key products or services"),
            "primary_decision_makers": pull("primary decision makers"),
            "points_of_contact": pull("points of contact"),
            "recent_news": pull("recent news"),
        }
    
    cb = data.get("company_background", "Not found") or "Not found"
    ps = data.get("products_services", "Not found") or "Not found"
    dm = data.get("primary_decision_makers", "Not found") or "Not found"
    poc = data.get("points_of_contact", "Not found") or "Not found"
    rn = data.get("recent_news", "Not found") or "Not found"
    
    def avail(v: str) -> bool:
        return bool(v) and v.strip() and v.strip().lower() != "not found"
    cb_prev = state.get("company_background", "")
    ps_prev = state.get("products_services", "")
    dm_prev = state.get("primary_decision_makers", "")
    poc_prev = state.get("points_of_contact", "")
    rn_prev = state.get("recent_news", "")
    
    cb_final = cb_prev if avail(cb_prev) else (cb if avail(cb) else "Not found")
    ps_final = ps_prev if avail(ps_prev) else (ps if avail(ps) else "Not found")
    dm_final = dm_prev if avail(dm_prev) else (dm if avail(dm) else "Not found")
    poc_final = poc_prev if avail(poc_prev) else (poc if avail(poc) else "Not found")
    rn_final = rn_prev if avail(rn_prev) else (rn if avail(rn) else "Not found")
    
    filled = sum(1 for v in [cb_final, ps_final, dm_final, poc_final, rn_final] if avail(v))
    quality = filled / 5.0
    
    return {
        "company_background": cb_final,
        "products_services": ps_final,
        "primary_decision_makers": dm_final,
        "points_of_contact": poc_final,
        "recent_news": rn_final,
        "quality_score": quality,
        "status": "analysis_completed",
        "current_node": "analyzer"
    }

def report_generator_node(state: ResearchState):
    """Generate final research report"""
    state["current_node"] = "report_generator"
    report = f"""Company Background
{state.get('company_background', '')}

Key Products or Services
{state.get('products_services', '')}

Primary Decision Makers (with titles if possible)
{state.get('primary_decision_makers', '')}

Point(s) of Contact (emails, LinkedIn, or roles)
{state.get('points_of_contact', '')}

Recent News or Strategic Moves
{state.get('recent_news', '')}
"""

    return {
        "final_report": report,
        "status": "report_completed",
        "current_node": "report_generator"
    }

def should_continue_research(state: ResearchState) -> Literal["search", "report"]:
    """Router: Decide whether to continue searching or generate report"""
    th = state.get("quality_threshold", 0.8)
    if state["quality_score"] >= th:
        return "report"
    if state["iteration"] >= state["max_iterations"]:
        return "report"
    return "search"

# Build workflow
@st.cache_resource
def build_workflow():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("input", input_processor_node)
    workflow.add_node("questions", question_generator_node)
    workflow.add_node("search", search_tool_node)
    workflow.add_node("analyze", analyzer_node)
    workflow.add_node("report", report_generator_node)
    
    # Define flow
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "questions")
    workflow.add_edge("questions", "search")
    workflow.add_edge("search", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "search": "questions",
            "report": "report"
        }
    )
    workflow.add_edge("report", END)
    
    return workflow.compile()

# Streamlit UI
def main():
    st.title("📣 Product Pitch Research Assistant")
    st.markdown("**Extract company intel to prepare a product pitch: background, products/services, decision makers, contacts, and recent news.**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum number of search iterations"
        )
        
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Minimum quality score to stop research"
        )
        
        st.divider()
        
        st.subheader("🏢 Example Companies")
        example_companies = [
            ("OpenAI", "United States"),
            ("NVIDIA", "United States"),
            ("Stripe", "United States"),
            ("Salesforce", "United States"),
            ("Siemens", "Germany")
        ]
        
        if st.button("Load Example"):
            choice = st.selectbox(
                "Choose a company:",
                [f"{n} • {c}" for n, c in example_companies]
            )
            if " • " in choice:
                n, c = choice.split(" • ", 1)
                st.session_state.example_company = n
                st.session_state.example_country = c
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏢 Company")
        
        # Input form
        with st.form("research_form"):
            company_name_input = st.text_input(
                "Company name:",
                value=st.session_state.get("example_company", ""),
                placeholder="e.g., OpenAI"
            )
            country_input = st.text_input(
                "Country:",
                value=st.session_state.get("example_country", ""),
                placeholder="e.g., United States"
            )
            
            submitted = st.form_submit_button("🚀 Start Research", type="primary")
        
        if submitted and company_name_input:
            # Initialize state
            initial_state = {
                "topic": company_name_input,
                "company_name": company_name_input,
                "country": country_input,
                "website_url": "",
                "research_questions": [],
                "search_queries": [],
                "search_results": [],
                "iteration": 0,
                "max_iterations": max_iterations,
                "quality_score": 0.0,
                "quality_threshold": quality_threshold,
                "final_report": "",
                "status": "initialized",
                "current_node": "",
                "company_background": "",
                "products_services": "",
                "primary_decision_makers": "",
                "points_of_contact": "",
                "recent_news": ""
            }
            
            # Progress tracking
            progress_container = st.container()
            
            with progress_container:
                st.subheader("🔄 Research Progress")
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Node status cards
                node_cols = st.columns(5)
                node_status = {}
                
                nodes = ["input", "questions", "search", "analyze", "report"]
                node_labels = ["📥 Input", "🔎 Query Generator", "🔍 Search", "🧭 Extract", "📝 Format"]
                
                for i, (node, label) in enumerate(zip(nodes, node_labels)):
                    with node_cols[i]:
                        node_status[node] = st.container()
                        with node_status[node]:
                            st.metric(label, "⏳ Waiting")
                
                # Run workflow with status updates
                assistant = build_workflow()
                
                with st.status("Research in progress...", expanded=True) as status:
                    try:
                        # Process with streaming updates
                        for i, (node, label) in enumerate(zip(nodes, node_labels)):
                            status.write(f"Processing: {label}")
                            
                            # Update node status
                            with node_status[node]:
                                st.metric(label, "🔄 Active", delta="Processing")
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(nodes))
                            status_text.text(f"Step {i+1}/{len(nodes)}: {label}")
                            
                            # Small delay for visual effect
                            time.sleep(0.5)
                            
                            # Mark as complete
                            with node_status[node]:
                                st.metric(label, "✅ Done", delta="Complete")
                        
                        # Execute workflow
                        result = assistant.invoke(initial_state)
                        status.update(label="Research complete!", state="complete")
                        
                    except Exception as e:
                        status.update(label="Research failed", state="error")
                        st.error(f"Error: {e}")
                        return
                
                # Display results
                st.divider()
                st.subheader("📊 Extracted Fields")
                
                # Create tabs for different sections
                tabs = st.tabs(["🧾 Extracted Fields", "🔎 Search Queries", "🔍 Search Results", "📈 Metadata"])
                
                with tabs[0]:
                    st.text(result["final_report"])
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Fields (txt)",
                        data=result["final_report"],
                        file_name=f"pitch_research_{(company_name_input or 'company').replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                
                with tabs[1]:
                    st.subheader("Search Queries")
                    for i, q in enumerate(result["research_questions"], 1):
                        st.write(f"{i}. {q}")
                
                with tabs[2]:
                    st.subheader("Search Results")
                    with st.expander(f"View {len(result['search_results'])} results"):
                        for i, r in enumerate(result["search_results"], 1):
                            st.write(f"**Result {i}:**")
                            st.write(r[:300] + "..." if len(r) > 300 else r)
                            st.divider()
                
                with tabs[3]:
                    st.subheader("Research Metadata")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Iterations", result["iteration"])
                        st.metric("Queries", len(result["research_questions"]))
                    
                    with col2:
                        st.metric("Sources", len(result["search_results"]))
                    
                    with col3:
                        st.metric("Quality Score", f"{result['quality_score']:.2f}")
                        st.metric("Status", result["status"])
                    
                    # Full state viewer
                    with st.expander("View Complete State"):
                        st.json({k: v for k, v in result.items() if k != "final_report"})
    
    with col2:
        st.subheader("🗺️ Workflow Visualization")
        
        # Workflow diagram
        st.markdown("""
        ```mermaid
        graph TD
            Start([Start]) --> Input[📥 Input Processor]
            Input --> Queries[🔎 Query Generator]
            Queries --> Search[🔍 Search Tool]
            Search --> Extract[🧭 Extractor]
            Extract --> Router{🚦 Continue?}
            Router -->|Search more| Queries
            Router -->|Done| Report[📝 Report Formatter]
            Report --> End([End])
            
            style Start fill:#48bb78
            style End fill:#fc8181
            style Router fill:#f6ad55
        ```
        """)
        
        # Info boxes
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Input Processing**: Normalizes the company name
            2. **Query Generation**: Builds targeted queries for the 5 fields
            3. **Search Tool**: Uses DuckDuckGo to find information
            4. **Extraction**: LLM consolidates the 5 fields from results
            5. **Report Formatting**: Outputs the fields in plain text
            """)
        
        with st.expander("🎓 LangGraph Concepts"):
            st.markdown("""
            This app demonstrates:
            - **StateGraph**: Managing workflow state
            - **Nodes**: Processing functions
            - **Edges**: Workflow connections
            - **Conditional Routing**: Dynamic paths
            - **Loops**: Iterative refinement
            - **Tool Integration**: External APIs
            - **State Accumulation**: Building knowledge
            """)
        
        with st.expander("💻 View Code"):
            st.code("""
# Simplified workflow definition
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("input", input_processor_node)
workflow.add_node("questions", question_generator_node)
workflow.add_node("search", search_tool_node)
workflow.add_node("analyze", analyzer_node)
workflow.add_node("report", report_generator_node)

# Define flow with loop
workflow.add_edge(START, "input")
workflow.add_edge("input", "questions")
workflow.add_edge("questions", "search")
workflow.add_edge("search", "analyze")

# Conditional routing for loop
workflow.add_conditional_edges(
    "analyze",
    should_continue_research,
    {
        "search": "questions",
        "report": "report"
    }
)

workflow.add_edge("report", END)

# Compile and run
assistant = workflow.compile()
result = assistant.invoke(initial_state)
            """, language="python")

if __name__ == "__main__":
    main()
