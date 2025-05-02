import streamlit as st
import os
import json
from typing import Dict, TypedDict, Annotated, Sequence, List, Any, Optional, Union
from langchain_groq import ChatGroq
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel
import operator
from langgraph.graph import END, StateGraph
import traceback

# Import utility functions from error_handler.py
from error_handler import (
    safe_execute, 
    validate_list, 
    validate_dict, 
    is_valid_subtask,
    get_nested_value
)

# Set page config
st.set_page_config(
    page_title="AI Task Planner",
    page_icon="📋",
    layout="wide"
)

st.title("🤖 AI Task Planning Assistant")
st.markdown("""
This app helps you break down complex tasks into actionable subtasks, 
incorporating best practices from web search results.
""")

# API key inputs with password masking
with st.sidebar:
    st.header("API Keys")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
    tavily_api_key = st.text_input("Tavily API Key", type="password", help="Enter your Tavily API key")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses LangGraph to:
    1. Split your task into subtasks
    2. Search for best practices online
    3. Refine subtasks based on search results
    """)

# Define the state schema
class Subtask(BaseModel):
    id: int
    description: str

class SubtaskList(BaseModel):
    subtasks: List[Subtask] = Field(description="The list of subtasks")

class AgentState(TypedDict):
    original_task: str
    draft_subtasks: List[Dict]
    search_results: List[Dict]
    final_subtasks: List[Dict]
    messages: Sequence[HumanMessage | AIMessage]

# Task splitting function
def split_task(state: AgentState) -> AgentState:
    """Split the original task into draft subtasks"""
    prompt = ChatPromptTemplate.from_template(
        """Break the following high-level task into 5–7 clear, actionable subtasks.
        
        Task: {task}
        
        Format each subtask as a JSON object with 'id' and 'description' fields.
        Return a JSON array of these objects.
        """
    )
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    with st.status("Breaking down task into subtasks...", expanded=True) as status:
        try:
            subtasks = chain.invoke({"task": state["original_task"]})
            
            # Validate subtasks
            if not isinstance(subtasks, list):
                st.error("Error: Expected a list of subtasks but received a different format.")
                subtasks = []
                
            status.update(label="✅ Task breakdown complete!", state="complete")
        except Exception as e:
            st.error(f"Error in task breakdown: {str(e)}")
            subtasks = []
            status.update(label="❌ Task breakdown failed", state="error")
    
    return {
        **state, 
        "draft_subtasks": subtasks,
        "messages": list(state["messages"]) + [AIMessage(content=f"I've broken down your task into {len(subtasks)} subtasks.")]
    }

# Web search function
def search_best_practices(state: AgentState) -> AgentState:
    """Search for best practices related to the task"""
    web_tool = TavilySearchResults(k=3)
    
    search_query = f"best practices for {state['original_task']}"
    
    with st.status("Searching for best practices...", expanded=True) as status:
        try:
            search_results = web_tool.invoke(search_query)
            
            # Validate search results
            if not isinstance(search_results, list):
                st.error("Error: Expected a list of search results but received a different format.")
                search_results = []
                
            status.update(label="✅ Search complete!", state="complete")
        except Exception as e:
            st.error(f"Error in web search: {str(e)}")
            search_results = []
            status.update(label="❌ Search failed", state="error")
    
    return {
        **state,
        "search_results": search_results,
        "messages": list(state["messages"]) + [AIMessage(content="I've gathered some recent best practices from the web.")]
    }

# Refine subtasks with search results
def refine_subtasks(state: AgentState) -> AgentState:
    """Refine subtasks based on search results"""
    prompt = ChatPromptTemplate.from_template(
        """Given the following draft subtasks and search results on best practices, 
        refine the subtasks to incorporate current best practices.
        
        Original Task: {original_task}
        
        Draft Subtasks:
        {draft_subtasks}
        
        Search Results on Best Practices:
        {search_results}
        
        Please update the subtasks incorporating these best practices.
        Format each subtask as a JSON object with 'id' and 'description' fields.
        Return a JSON array of these objects.
        """
    )
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    with st.status("Refining subtasks based on best practices...", expanded=True) as status:
        try:
            # Ensure draft_subtasks is a valid list
            if not state.get("draft_subtasks") or not isinstance(state["draft_subtasks"], list):
                state["draft_subtasks"] = []
                
            # Ensure search_results is a valid list
            if not state.get("search_results") or not isinstance(state["search_results"], list):
                state["search_results"] = []
                
            refined_subtasks = chain.invoke({
                "original_task": state["original_task"],
                "draft_subtasks": state["draft_subtasks"],
                "search_results": state["search_results"]
            })
            
            # Validate refined_subtasks
            if not isinstance(refined_subtasks, list):
                st.error("Error: Expected a list of refined subtasks but received a different format.")
                refined_subtasks = state.get("draft_subtasks", [])  # Fall back to draft subtasks
                
            status.update(label="✅ Refinement complete!", state="complete")
        except Exception as e:
            st.error(f"Error in refining subtasks: {str(e)}")
            refined_subtasks = state.get("draft_subtasks", [])  # Fall back to draft subtasks
            status.update(label="❌ Refinement failed", state="error")
    
    # Safe string generation for the final message
    subtask_descriptions = []
    if refined_subtasks and isinstance(refined_subtasks, list):
        for task in refined_subtasks:
            if isinstance(task, dict) and "id" in task and "description" in task:
                subtask_descriptions.append(f"{task['id']}. {task['description']}")
    
    final_message = f"""
Here are your refined subtasks for: {state['original_task']}

{', '.join(subtask_descriptions) if subtask_descriptions else "No subtasks were generated."}

These subtasks incorporate current best practices found through web search.
"""
    
    return {
        **state,
        "final_subtasks": refined_subtasks,
        "messages": list(state["messages"]) + [AIMessage(content=final_message)]
    }

# Define the workflow graph
def create_agent_graph():
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("split_task", split_task)
    workflow.add_node("search_best_practices", search_best_practices)
    workflow.add_node("refine_subtasks", refine_subtasks)
    
    # Define edges
    workflow.add_edge("split_task", "search_best_practices")
    workflow.add_edge("search_best_practices", "refine_subtasks")
    workflow.add_edge("refine_subtasks", END)
    
    # Set entry point
    workflow.set_entry_point("split_task")
    
    # Compile the graph
    return workflow.compile()

# Create the main execution function
def plan_task(task: str):
    try:
        # Initialize the agent graph
        agent_graph = create_agent_graph()
        
        # Set initial state
        initial_state = {
            "original_task": task,
            "draft_subtasks": [],
            "search_results": [],
            "final_subtasks": [],
            "messages": [HumanMessage(content=f"Please help me plan: {task}")]
        }
        
        # Execute the graph
        result = agent_graph.invoke(initial_state)
        
        # Validate the result
        if not isinstance(result, dict):
            st.error("Error: Graph execution did not return a dictionary.")
            return {
                "task": task,
                "subtasks": [],
                "conversation": [AIMessage(content="There was an error processing your task.")],
                "search_results": []
            }
        
        # Return the final result with validated data
        return {
            "task": task,
            "subtasks": validate_list(get_nested_value(result, ["final_subtasks"], []), is_valid_subtask),
            "conversation": validate_list(get_nested_value(result, ["messages"], [])),
            "search_results": validate_list(get_nested_value(result, ["search_results"], []))
        }
    
    except Exception as e:
        st.error(f"Error planning task: {str(e)}")
        traceback.print_exc()
        
        # Return a fallback result
        return {
            "task": task,
            "subtasks": [],
            "conversation": [AIMessage(content=f"There was an error processing your task: {str(e)}")],
            "search_results": []
        }

# Main app flow
task_input = st.text_area("Enter your task:", placeholder="Example: Organize an online conference on AI and ethics", height=100)

# Initialize LLM and execute when user submits task
if st.button("Plan My Task", type="primary"):
    if not task_input:
        st.error("Please enter a task!")
    elif not groq_api_key or not tavily_api_key:
        st.error("Please enter both API keys in the sidebar!")
    else:
        # Set API keys
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key

        # Initialize Groq LLM
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0
        )
        
        try:
            # Create tabs for displaying results
            results_tab, search_tab, conversation_tab = st.tabs(["📋 Final Plan", "🔍 Search Results", "💬 Agent Reasoning"])
            
            # Execute planning
            with st.spinner("Planning your task..."):
                result = plan_task(task_input)
                
            # Ensure all expected keys exist in the result
            result = result or {}  # Ensure result is at least an empty dict if None
            if "subtasks" not in result:
                result["subtasks"] = []
            if "conversation" not in result:
                result["conversation"] = []
            if "search_results" not in result:
                result["search_results"] = []
            
            # Display task subtasks in the first tab
            with results_tab:
                st.header(f"Task Plan: {task_input}")
                
                if result["subtasks"] and isinstance(result["subtasks"], list):
                    for task in result["subtasks"]:
                        if isinstance(task, dict) and "id" in task and "description" in task:
                            st.markdown(f"**{task['id']}. {task['description']}**")
                        else:
                            st.warning("Invalid subtask format encountered")
                else:
                    st.warning("No subtasks were generated. There might be an issue with the planning process.")
                
                # Add download button for the plan only if we have valid subtasks
                if result["subtasks"] and isinstance(result["subtasks"], list):
                    import json
                    result_json = json.dumps({"task": task_input, "subtasks": result["subtasks"]}, indent=2)
                    st.download_button(
                        label="Download Plan as JSON",
                        data=result_json,
                        file_name="task_plan.json",
                        mime="application/json",
                    )
            
            # Display search results in the second tab
            with search_tab:
                st.header("Best Practices Search Results")
                
                # Check if search_results exists and is not None
                if result.get("search_results") and isinstance(result["search_results"], list):
                    for idx, result_item in enumerate(result["search_results"]):
                        if isinstance(result_item, dict) and "title" in result_item:
                            with st.expander(f"Source {idx+1}: {result_item['title']}"):
                                st.markdown(f"**URL:** [{result_item.get('url', '#')}]({result_item.get('url', '#')})")
                                st.markdown(result_item.get('content', 'No content available'))
                else:
                    st.info("No search results available or search was unsuccessful.")
            
            # Display agent conversation in the third tab
            with conversation_tab:
                st.header("Agent Reasoning Process")
                
                if result["conversation"] and isinstance(result["conversation"], list):
                    for message in result["conversation"]:
                        if isinstance(message, HumanMessage):
                            st.markdown(f"**User:** {message.content}")
                        elif isinstance(message, AIMessage):
                            st.markdown(f"**Agent:** {message.content}")
                        elif isinstance(message, dict) and "role" in message and "content" in message:
                            # Handle dictionary-like messages if they come in that format
                            role = "User" if message["role"].lower() == "human" else "Agent"
                            st.markdown(f"**{role}:** {message['content']}")
                        else:
                            st.markdown("**Message:** Invalid format")
                else:
                    st.info("No conversation history available.")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Check if your API keys are correct and try again.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit, LangGraph, LangChain, and Groq LLM")