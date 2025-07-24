# main.py
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import GoogleSearchAPIWrapper
import requests
from dataclasses import dataclass

# Initialize FastAPI app
app = FastAPI(
    title="iLogAnalyze API",
    description="AI-Powered Log Analysis & Issue Resolution using ReAct Agent",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for session management (use Redis/Database in production)
sessions = {}

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    timestamp: str
    tools_used: List[str] = []

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    message: str
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

@dataclass
class SearchResult:
    title: str
    snippet: str
    link: str
    source: str

class VectorRetrieverTool:
    """Tool for retrieving information from uploaded log files using vector similarity search"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.name = "log_vector_search"
        self.description = """
        Use this tool to search through the uploaded log file for specific information.
        Input should be a question or search query about the log file content.
        This tool will return relevant excerpts from the log file that match your query.
        """
    
    def run(self, query: str) -> str:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            index_path = f"./vector_stores/{self.session_id}/faiss_index"
            
            if not os.path.exists(index_path):
                return "No log file has been uploaded for this session. Please upload a log file first."
            
            vectorDB = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
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
                # Fallback to simulated search
                return self._fallback_search(query)
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _google_search(self, query: str) -> str:
        """Search using Google Custom Search API"""
        try:
            search = GoogleSearchAPIWrapper()
            enhanced_query = f"{query} site:stackoverflow.com OR site:github.com OR site:docs.python.org"
            results = search.run(enhanced_query)
            return f"Web search results for '{query}':\n{results}"
        except Exception as e:
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> str:
        """Fallback search with common solutions"""
        try:
            # Common error patterns and solutions
            error_solutions = {
                "connection timeout": "1. Check network connectivity\n2. Increase timeout values\n3. Verify firewall settings\n4. Check DNS resolution",
                "memory leak": "1. Use memory profilers\n2. Check for unclosed resources\n3. Review object lifecycle\n4. Monitor garbage collection",
                "null pointer": "1. Add null checks\n2. Initialize variables properly\n3. Validate input parameters\n4. Use defensive programming",
                "authentication": "1. Verify credentials\n2. Check token expiration\n3. Review permission settings\n4. Validate authentication flow",
                "database": "1. Check connection strings\n2. Verify database status\n3. Review query performance\n4. Check transaction handling"
            }
            
            result_text = f"Troubleshooting suggestions for '{query}':\n\n"
            
            # Find matching solutions
            for pattern, solution in error_solutions.items():
                if pattern.lower() in query.lower():
                    result_text += f"Common solutions for {pattern} issues:\n{solution}\n\n"
            
            result_text += "For more specific solutions, configure Google Custom Search API or search manually on:\n"
            result_text += "- Stack Overflow: https://stackoverflow.com\n"
            result_text += "- GitHub Issues: https://github.com\n"
            result_text += "- Official Documentation\n"
            
            return result_text
            
        except Exception as e:
            return f"Unable to provide troubleshooting suggestions: {str(e)}"

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
            
            response = requests.get(url, params=params, timeout=10)
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
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        self.tools = self._initialize_tools()
        self.agent = self._create_agent()
        self.tools_used = []
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize all available tools"""
        vector_tool = VectorRetrieverTool(self.session_id)
        web_tool = WebSearchTool()
        stackoverflow_tool = StackOverflowSearchTool()
        
        return [
            Tool(
                name=vector_tool.name,
                description=vector_tool.description,
                func=lambda query: self._track_tool_usage("log_vector_search", vector_tool.run(query))
            ),
            Tool(
                name=web_tool.name,
                description=web_tool.description,
                func=lambda query: self._track_tool_usage("web_search", web_tool.run(query))
            ),
            Tool(
                name=stackoverflow_tool.name,
                description=stackoverflow_tool.description,
                func=lambda query: self._track_tool_usage("stackoverflow_search", stackoverflow_tool.run(query))
            )
        ]
    
    def _track_tool_usage(self, tool_name: str, result: str) -> str:
        """Track which tools are being used"""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        return result
    
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
    
    def analyze_issue(self, question: str) -> tuple[str, List[str]]:
        """Main method to analyze issues using the ReAct agent"""
        self.tools_used = []  # Reset tools tracking
        try:
            result = self.agent.invoke({"input": question})
            return result.get("output", "No response generated"), self.tools_used
        except Exception as e:
            return f"Error during analysis: {str(e)}", self.tools_used

# Utility functions
def log_to_text(logfile_content: bytes) -> str:
    """Convert uploaded log file content to text"""
    return logfile_content.decode("utf-8")

def text_to_chunk(text: str) -> List[str]:
    """Split text into chunks for vector embedding"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks: List[str], session_id: str) -> int:
    """Create and save vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorDB = FAISS.from_texts(chunks, embeddings)
    
    # Create session-specific directory
    session_dir = f"./vector_stores/{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    
    vectorDB.save_local(f"{session_dir}/faiss_index")
    return len(chunks)

# Background task for cleanup
def cleanup_old_sessions():
    """Clean up old session data (run this periodically)"""
    try:
        vector_stores_dir = Path("./vector_stores")
        if vector_stores_dir.exists():
            for session_dir in vector_stores_dir.iterdir():
                if session_dir.is_dir():
                    # Remove sessions older than 24 hours
                    creation_time = session_dir.stat().st_ctime
                    if datetime.now().timestamp() - creation_time > 86400:  # 24 hours
                        shutil.rmtree(session_dir)
    except Exception as e:
        print(f"Cleanup error: {e}")

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_log_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process log file"""
    
    # Validate file type
    if not file.filename.endswith(('.txt', '.log')):
        raise HTTPException(status_code=400, detail="Only .txt and .log files are supported")
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Read file content
        content = await file.read()
        
        # Convert to text and create chunks
        text = log_to_text(content)
        chunks = text_to_chunk(text)
        
        # Create vector store
        chunks_count = create_vector_store(chunks, session_id)
        
        # Store session info
        sessions[session_id] = {
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "chunks_count": chunks_count
        }
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_old_sessions)
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            message="Log file processed successfully",
            chunks_created=chunks_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/analyze", response_model=QueryResponse)
async def analyze_issue(query: QueryRequest):
    """Analyze log issue and provide solutions"""
    
    # Validate session if provided
    if query.session_id and query.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Use provided session_id or create a temporary one
    session_id = query.session_id or "temp_session"
    
    try:
        # Create analyzer instance
        analyzer = ReActLogAnalyzer(session_id)
        
        # Analyze the issue
        answer, tools_used = analyzer.analyze_issue(query.question)
        
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            tools_used=tools_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its data"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Remove vector store
        session_dir = f"./vector_stores/{session_id}"
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        
        # Remove from sessions
        del sessions[session_id]
        
        return {"message": "Session deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "iLogAnalyze API - AI-Powered Log Analysis & Issue Resolution",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "analyze": "/analyze",
            "sessions": "/sessions/{session_id}",
            "docs": "/docs"
        }
    }

# Run the application
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./vector_stores", exist_ok=True)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
