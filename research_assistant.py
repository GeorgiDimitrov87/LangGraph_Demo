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
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock

from ddgs import DDGS
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="LangGraph Research Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_llm():
    """Use Amazon Bedrock-hosted Claude model instead of OpenAI."""
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        temperature=0.7,
        max_tokens=2048
    )

# State definition
class ResearchState(TypedDict):
    topic: str
    research_questions: List[str]
    search_queries: List[str]
    search_results: List[str]
    key_findings: List[str]
    iteration: int
    max_iterations: int
    quality_score: float
    final_report: str
    status: str
    current_node: str

# Node functions
def input_processor_node(state: ResearchState):
    """Process and validate input topic"""
    llm = get_llm()
    
    # Update UI status
    state["current_node"] = "input_processor"
    
    prompt = f"""Given this research topic: '{state['topic']}'
    Suggest a more specific research focus (one line):"""
    
    response = llm.invoke(prompt)
    enhanced_topic = response.content.strip()
    
    return {
        "topic": enhanced_topic,
        "status": "topic_processed",
        "current_node": "input_processor"
    }

def question_generator_node(state: ResearchState):
    """Generate research questions"""
    llm = get_llm()
    state["current_node"] = "question_generator"
    
    prompt = f"""Generate 3 specific research questions about: {state['topic']}
    Questions should be searchable and factual.
    Return only the questions, one per line:"""
    
    response = llm.invoke(prompt)
    questions = response.content.strip().split('\n')
    questions = [q.strip() for q in questions if q.strip()][:3]
    
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
                results = ddgs.text(question, max_results=2)
                
                for result in results:
                    title = result.get('title', '')
                    body = result.get('body', '')
                    search_results.append(f"{title}: {body}")
                
                search_queries.append(question)
                
            except Exception as e:
                st.error(f"Search error: {e}")
    
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
        return {"key_findings": ["No search results to analyze"], "quality_score": 0.0}
    
    results_text = "\n".join(state["search_results"][:10])
    
    prompt = f"""Analyze these search results about '{state['topic']}':

{results_text}

Extract 5 key findings. Return only the findings, one per line:"""
    
    response = llm.invoke(prompt)
    findings = response.content.strip().split('\n')
    findings = [f.strip() for f in findings if f.strip()][:5]
    
    all_findings = state.get("key_findings", []) + findings
    quality = min(len(all_findings) * 0.2, 1.0)
    
    return {
        "key_findings": all_findings,
        "quality_score": quality,
        "status": "analysis_completed",
        "current_node": "analyzer"
    }

def report_generator_node(state: ResearchState):
    """Generate final research report"""
    llm = get_llm()
    state["current_node"] = "report_generator"
    
    prompt = f"""Create a comprehensive research report based on this information:

Topic: {state['topic']}

Research Questions:
{chr(10).join(state['research_questions'])}

Key Findings:
{chr(10).join(state['key_findings'])}

Number of sources consulted: {len(state['search_results'])}

Generate a well-structured report with:
1. Executive Summary
2. Key Findings
3. Conclusion

Keep it concise but informative:"""
    
    response = llm.invoke(prompt)
    report = response.content
    
    # Add metadata
    report += f"\n\n---\n📊 Research Metadata:\n"
    report += f"• Topic: {state['topic']}\n"
    report += f"• Questions Asked: {len(state['research_questions'])}\n"
    report += f"• Sources Consulted: {len(state['search_results'])}\n"
    report += f"• Key Findings: {len(state['key_findings'])}\n"
    report += f"• Research Iterations: {state['iteration']}\n"
    report += f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    
    return {
        "final_report": report,
        "status": "report_completed",
        "current_node": "report_generator"
    }

def should_continue_research(state: ResearchState) -> Literal["search", "report"]:
    """Router: Decide whether to continue searching or generate report"""
    if state["iteration"] >= state["max_iterations"]:
        return "report"
    
    if state["quality_score"] >= 0.8:
        return "report"
    
    if len(state.get("key_findings", [])) >= 10:
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
    
    # Conditional routing
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
    st.title("🤖 LangGraph Research Assistant")
    st.markdown("**Interactive AI-powered research using LangGraph workflow orchestration**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=5,
            value=2,
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
        
        st.subheader("📚 Example Topics")
        example_topics = [
            "Benefits of LangGraph for AI agents",
            "State management in workflow systems",
            "Building production AI applications",
            "Graph-based vs linear AI workflows",
            "Tool integration in language models"
        ]
        
        if st.button("Load Example"):
            st.session_state.example_topic = st.selectbox(
                "Choose a topic:",
                example_topics
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Research Topic")
        
        # Input form
        with st.form("research_form"):
            topic = st.text_input(
                "Enter your research topic:",
                value=st.session_state.get("example_topic", ""),
                placeholder="e.g., Benefits of LangGraph for AI agents"
            )
            
            submitted = st.form_submit_button("🚀 Start Research", type="primary")
        
        if submitted and topic:
            # Initialize state
            initial_state = {
                "topic": topic,
                "research_questions": [],
                "search_queries": [],
                "search_results": [],
                "key_findings": [],
                "iteration": 0,
                "max_iterations": max_iterations,
                "quality_score": 0.0,
                "final_report": "",
                "status": "initialized",
                "current_node": ""
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
                node_labels = ["📥 Input", "❓ Questions", "🔍 Search", "🔬 Analyze", "📝 Report"]
                
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
                st.subheader("📊 Research Results")
                
                # Create tabs for different sections
                tabs = st.tabs(["📝 Report", "❓ Questions", "🔍 Searches", "💡 Findings", "📈 Metadata"])
                
                with tabs[0]:
                    st.markdown(result["final_report"])
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=result["final_report"],
                        file_name=f"research_{topic.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                
                with tabs[1]:
                    st.subheader("Generated Questions")
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
                    st.subheader("Key Findings")
                    for i, f in enumerate(result["key_findings"], 1):
                        st.info(f"**Finding {i}:** {f}")
                
                with tabs[4]:
                    st.subheader("Research Metadata")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Iterations", result["iteration"])
                        st.metric("Questions", len(result["research_questions"]))
                    
                    with col2:
                        st.metric("Sources", len(result["search_results"]))
                        st.metric("Findings", len(result["key_findings"]))
                    
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
            Input --> Questions[❓ Question Generator]
            Questions --> Search[🔍 Search Tool]
            Search --> Analyze[🔬 Analyzer]
            Analyze --> Router{🚦 Continue?}
            Router -->|Yes| Questions
            Router -->|No| Report[📝 Report Generator]
            Report --> End([End])
            
            style Start fill:#48bb78
            style End fill:#fc8181
            style Router fill:#f6ad55
        ```
        """)
        
        # Info boxes
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Input Processing**: Enhances your topic for better research
            2. **Question Generation**: Creates specific research questions
            3. **Search Tool**: Uses DuckDuckGo to find information
            4. **Analysis**: Extracts key findings from search results
            5. **Routing**: Decides whether to continue or generate report
            6. **Report Generation**: Compiles findings into a structured report
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
