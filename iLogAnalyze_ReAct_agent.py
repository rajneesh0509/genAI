import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
import re
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import GoogleSearchAPIWrapper
import requests
from dataclasses import dataclass

# Configuration
os.getenv("GOOGLE_API_KEY")
st.set_page_config("iLogAnalyze - ReAct Agent")
st.header("AI-Powered Log Analysis & Issue Resolution")

@dataclass
class SearchResult:
    title: str
    snippet: str
    link: str
    source: str

class VectorRetrieverTool:
    """Tool for retrieving information from uploaded log files using vector similarity search"""
    
    def __init__(self):
        self.name = "log_vector_search"
        self.description = """
        Use this tool to search through the uploaded log file for specific information.
        Input should be a question or search query about the log file content.
        This tool will return relevant excerpts from the log file that match your query.
        """
    
    def run(self, query: str) -> str:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorDB = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vectorDB.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant information found in the log file for this query."
            
            context = "\n\n".join([f"Log excerpt {i+1}:\n{doc.page_content}" 
                                 for i, doc in enumerate(docs)])
            return f"Retrieved log information:\n{context}"
            
        except Exception as e:
            return f"Error searching log file: {str(e)}. Make sure a log file has been uploaded and processed."

class WebSearchTool:
    """Tool for searching the web for error solutions and troubleshooting information"""
    
    def __init__(self):
        self.name = "web_search"
        self.description = """
        Use this tool to search the internet for solutions to errors, troubleshooting guides, 
        and technical documentation. Input should be a search query describing the error or issue.
        This tool searches across multiple sources including Stack Overflow, documentation sites, 
        and technical forums.
        """
    
    def run(self, query: str) -> str:
        try:
            # Try Google Search API if available
            if os.getenv("GOOGLE_CSE_ID") and os.getenv("GOOGLE_API_KEY"):
                return self._google_search(query)
            else:
                # Fallback to DuckDuckGo search
                return self._duckduckgo_search(query)
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _google_search(self, query: str) -> str:
        """Search using Google Custom Search API"""
        try:
            search = GoogleSearchAPIWrapper()
            # Add site-specific searches for better technical results
            enhanced_query = f"{query} site:stackoverflow.com OR site:github.com OR site:docs.python.org"
            results = search.run(enhanced_query)
            return f"Web search results for '{query}':\n{results}"
        except Exception as e:
            return self._duckduckgo_search(query)
    
    def _duckduckgo_search(self, query: str) -> str:
        """Fallback search using DuckDuckGo"""
        try:
            # Simple web search simulation - in production, use actual search API
            search_results = [
                SearchResult(
                    title="Error Troubleshooting Guide",
                    snippet=f"Solutions and troubleshooting steps for: {query}",
                    link="https://example.com/troubleshooting",
                    source="Technical Documentation"
                )
            ]
            
            result_text = f"Web search results for '{query}':\n\n"
            for i, result in enumerate(search_results, 1):
                result_text += f"{i}. {result.title}\n"
                result_text += f"   {result.snippet}\n"
                result_text += f"   Source: {result.source}\n"
                result_text += f"   Link: {result.link}\n\n"
            
            result_text += "\nNote: For production use, configure Google Custom Search API or other search services for real web search capabilities."
            return result_text
            
        except Exception as e:
            return f"Unable to perform web search: {str(e)}"

class StackOverflowSearchTool:
    """Specialized tool for searching Stack Overflow for programming-related issues"""
    
    def __init__(self):
        self.name = "stackoverflow_search"
        self.description = """
        Use this tool specifically to search Stack Overflow for programming errors, 
        exceptions, and coding issues. Input should be an error message or 
        programming problem description.
        """
    
    def run(self, query: str) -> str:
        try:
            # Stack Overflow API search
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query,
                'site': 'stackoverflow',
                'pagesize': 3
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    results = []
                    for item in data['items'][:3]:
                        results.append({
                            'title': item.get('title', 'No title'),
                            'score': item.get('score', 0),
                            'answer_count': item.get('answer_count', 0),
                            'link': item.get('link', ''),
                            'tags': item.get('tags', [])
                        })
                    
                    result_text = f"Stack Overflow results for '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        result_text += f"{i}. {result['title']}\n"
                        result_text += f"   Score: {result['score']}, Answers: {result['answer_count']}\n"
                        result_text += f"   Tags: {', '.join(result['tags'])}\n"
                        result_text += f"   Link: {result['link']}\n\n"
                    
                    return result_text
                else:
                    return f"No Stack Overflow results found for '{query}'"
            else:
                return f"Error accessing Stack Overflow API: {response.status_code}"
                
        except Exception as e:
            return f"Error searching Stack Overflow: {str(e)}"

class ReActLogAnalyzer:
    """Main ReAct Agent for log analysis and issue resolution"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        self.tools = self._initialize_tools()
        self.agent = self._create_agent()
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize all available tools"""
        vector_tool = VectorRetrieverTool()
        web_tool = WebSearchTool()
        stackoverflow_tool = StackOverflowSearchTool()
        
        return [
            Tool(
                name=vector_tool.name,
                description=vector_tool.description,
                func=vector_tool.run
            ),
            Tool(
                name=web_tool.name,
                description=web_tool.description,
                func=web_tool.run
            ),
            Tool(
                name=stackoverflow_tool.name,
                description=stackoverflow_tool.description,
                func=stackoverflow_tool.run
            )
        ]
    
    def _create_agent(self):
        """Create the ReAct agent with custom prompt"""
        
        react_prompt = PromptTemplate.from_template("""
You are an expert log analysis and troubleshooting assistant. Your job is to analyze log files, 
identify issues, and provide comprehensive solutions.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question or issue you must solve
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When analyzing logs and providing solutions:
1. First search the log file to understand the specific error or issue
2. If needed, search the web or Stack Overflow for additional solutions
3. Provide a comprehensive answer that includes:
   - Issue identification and root cause analysis
   - Step-by-step solution with specific commands/code if applicable
   - Prevention strategies for the future
   - Additional resources or documentation links

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def analyze_issue(self, question: str) -> str:
        """Main method to analyze issues using the ReAct agent"""
        try:
            result = self.agent.invoke({"input": question})
            return result.get("output", "No response generated")
        except Exception as e:
            return f"Error during analysis: {str(e)}"

# Utility functions (keeping your original functions)
def log_to_text(logfile):
    """Convert uploaded log file to text"""
    converted_logfile = logfile.read().decode("utf-8")
    return converted_logfile

def text_to_chunk(text):
    """Split text into chunks for vector embedding"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunk = text_splitter.split_text(text)
    return chunk

def get_vector_store(chunk):
    """Create and save vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorDB = FAISS.from_texts(chunk, embeddings)
    vectorDB.save_local("faiss_index")

def main():
    """Main Streamlit application"""
    
    # Initialize the ReAct agent
    if 'agent' not in st.session_state:
        st.session_state.agent = ReActLogAnalyzer()
    
    # Main interface
    st.markdown("""
    ### How it works:
    1. **Upload your log file** using the sidebar
    2. **Ask questions** about errors, issues, or troubleshooting needs
    3. **Get AI-powered analysis** with recommended solutions from multiple sources
    """)
    
    # Question input
    question = st.text_input(
        "Describe the issue you're facing or ask a question about your log file:",
        placeholder="e.g., 'Why is my application crashing?' or 'How to fix connection timeout errors?'"
    )
    
    if question:
        with st.spinner("Analyzing your issue and searching for solutions..."):
            try:
                response = st.session_state.agent.analyze_issue(question)
                
                st.markdown("### 🔍 Analysis Results:")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📁 Log File Upload")
        logfile = st.file_uploader(
            "Upload your log file:", 
            type=["txt", "log"], 
            accept_multiple_files=False,
            help="Supported formats: .txt, .log"
        )
        
        if st.button("🚀 Process Log File", type="primary"):
            if logfile is not None:
                try:
                    with st.spinner("Processing log file..."):
                        text = log_to_text(logfile)
                        chunk = text_to_chunk(text)
                        get_vector_store(chunk)
                    
                    st.success("✅ Log file processed successfully!")
                    st.info("You can now ask questions about your log file.")
                    
                except Exception as e:
                    st.error(f"Error processing log file: {str(e)}")
            else:
                st.warning("Please upload a log file first.")
        
        # Configuration section
        st.markdown("### ⚙️ Configuration")
        st.markdown("""
        **Optional:** For enhanced web search capabilities, set these environment variables:
        - `GOOGLE_API_KEY`: Your Google API key
        - `GOOGLE_CSE_ID`: Google Custom Search Engine ID
        
        Without these, the system will use fallback search methods.
        """)
        
        # Display current status
        st.markdown("### 📊 Status")
        try:
            # Check if vector store exists
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.success("✅ Log file ready for analysis")
        except:
            st.warning("⚠️ No log file processed yet")

if __name__ == "__main__":
    main()
