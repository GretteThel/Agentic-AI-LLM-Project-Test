"""
Basic Chatbot with LangGraph

A simple conversational chatbot that responds to your messages naturally.
This is your starting point - the solution will show you how to add intelligent state management!
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Streamlit: Framework for building web apps
import streamlit as st

# ChatOpenAI: Connects to OpenAI's GPT models
from langchain_openai import ChatOpenAI

# Message types for conversation
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Pydantic: For defining data structures
from pydantic import BaseModel
from typing import List, Literal

# LangGraph: For building AI workflows
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Basic Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Basic Chatbot")
st.caption("A friendly AI assistant that chats with you")


# =============================================================================
# SESSION STATE
# =============================================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

#if "chatbot" not in st.session_state:
    #st.session_state.chatbot = None
    
if "prompt_generator" not in st.session_state:
    st.session_state.prompt_generator = None


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
        
        if st.button("Change API Keys"):
            st.session_state.openai_key = ""
            st.session_state.llm = None
           # st.session_state.chatbot = None
            st.session_state.prompt_generator = None
            st.rerun()
    else:
        st.warning("⚠️ Not Connected")


# =============================================================================
# API KEY INPUT
# =============================================================================

if not st.session_state.openai_key:
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-..."
    )
    
    if st.button("Connect"):
        if api_key and api_key.startswith("sk-"):
            st.session_state.openai_key = api_key
            st.rerun()
        else:
            st.error("❌ Invalid API key format")
    
    st.stop()


# =============================================================================
# INITIALIZE AI
# =============================================================================

if not st.session_state.llm:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=st.session_state.openai_key
    )


# =============================================================================
# CREATE SIMPLE CHATBOT
# =============================================================================

# if st.session_state.llm and not st.session_state.chatbot:
if st.session_state.llm and not st.session_state.prompt_generator:
    
    # Define requirements structure
    class PromptInstructions(BaseModel):
        """Structure for collecting prompt requirements"""
        objective: str
        variables: List[str]
        constraints: List[str]
        requirements: List[str]
    
    # Bind tool to LLM
    llm_with_tool = st.session_state.llm.bind_tools([PromptInstructions])
    
    # Define conversation state
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    # STATE 1: Gather requirements through conversation
    def gather_requirements(state: State):
        """Ask questions to understand what prompt the user needs"""
        system_prompt = """Help the user create a custom AI prompt through friendly conversation.

        You need to understand:
        1. Purpose: What do they want the AI to help with?
        2. Information needed: What details will they provide each time?
        3. Things to avoid: What should the AI NOT do?
        4. Must include: What should the AI always do?

        RULES:
        - Ask ONE question at a time in plain language
        - No technical terms like variables or parameters
        - Be conversational and friendly

        When you have all information, call the tool."""
        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}
    
    # Transition node between states
    def add_tool_message(state: State):
        """Add confirmation message after requirements are collected"""
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content="Requirements collected! Generating prompt...",
                    tool_call_id=tool_call_id
                )
            ]
        }
    
    # STATE 2: Generate the actual prompt
    def generate_prompt(state: State):
        """Create a professional prompt based on collected requirements"""
        tool_args = None
        post_tool_messages = []
        
        # Extract requirements from tool call
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_args = msg.tool_calls[0]["args"]
            elif isinstance(msg, ToolMessage):
                continue
            elif tool_args:
                post_tool_messages.append(msg)
        
        if tool_args:
            requirements_text = f"""
            Objective: {tool_args.get('objective', 'Not specified')}
            Variables: {', '.join(tool_args.get('variables', []))}
            Constraints: {', '.join(tool_args.get('constraints', []))}
            Requirements: {', '.join(tool_args.get('requirements', []))}
            """
            
            system_msg = SystemMessage(content=f"""Create a prompt template based on:

            {requirements_text}

            Guidelines:
            - Make it clear and specific
            - Use {{{{variable_name}}}} format for variables
            - Address all constraints and requirements
            - Use professional prompt engineering techniques""")
            
            messages = [system_msg] + post_tool_messages
        else:
            messages = post_tool_messages
        
        response = st.session_state.llm.invoke(messages)
        return {"messages": [response]}
    
        
    # Router to decide which state to go to
    def route_conversation(state: State) -> Literal["add_tool_message", "gather", "__end__"]:
        """Decide what to do next based on current state"""
        last_msg = state["messages"][-1]
        
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "add_tool_message"  # Requirements collected, transition to generate
        elif not isinstance(last_msg, HumanMessage):
            return "__end__"  # Done
        else:
            return "gather"  # Keep gathering requirements

    # Build workflow
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("gather", gather_requirements)
    workflow.add_node("add_tool_message", add_tool_message)
    workflow.add_node("generate", generate_prompt)
    
    # Add edges
    workflow.add_edge(START, "gather")
    workflow.add_conditional_edges(
        "gather",
        route_conversation,
        {
            "add_tool_message": "add_tool_message",
            "gather": "gather",
            "__end__": END
        }
    )
    workflow.add_edge("add_tool_message", "generate")
    workflow.add_edge("generate", END)
    
    # Compile and save
    st.session_state.prompt_generator = workflow.compile()


# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =============================================================================
# HANDLE USER INPUT
# =============================================================================

user_input = st.chat_input("Type your message...")

if user_input:
    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build message history
            messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Run prompt generator
            result = st.session_state.prompt_generator.invoke({"messages": messages})
            response = result["messages"][-1].content
            
            # Display and save response
            st.write(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
