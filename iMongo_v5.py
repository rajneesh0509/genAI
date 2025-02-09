import os
from ast import literal_eval
from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
import json
import pandas as pd

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# LLM initialization
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)
#llm = ChatGroq(model="DeepSeek-R1-Distill-Llama-70b", api_key=groq_api_key)

def connect_mongodb(uri, db):
    """Establish MongoDB connection"""
    try:
        client = MongoClient(uri)
        # Test the connection
        client.server_info()
        return client[db]
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

def text_to_query(input_text, collection_name):
    """Convert natural language to MongoDB query"""
    prompt = PromptTemplate(
        template="""You are an expert in MongoDB. Convert the given input text into a valid pymongo query operation.
        The query should run on collection: {collection}

        Important instructions:
        1. Return ONLY the Python dictionary/code that represents the MongoDB operation
        2. For find operations, return query as: {{"operation": "find", "query": {{your_query}}, "projection": {{optional}}, "sort": {{optional}}}}
        3. For insert operations, return as: {{"operation": "insert", "documents": [{{doc1}}, {{doc2}}]}}
        4. For update operations, return as: {{"operation": "update", "filter": {{your_filter}}, "update": {{your_update}}, "many": true/false}}
        5. For delete operations, return as: {{"operation": "delete", "filter": {{your_filter}}, "many": true/false}}
        6. For aggregate operations, return as: {{"operation": "aggregate", "pipeline": [{{stage1}}, {{stage2}}]}}
        7. For distinct/unique operations, return as: {{"operation": "distinct", "field": "field_name", "query": {{optional_filter}}}}
        
        Do not include any code formatting markers or explanations.
        
        Input text: {query}
        """,
        input_variables=["query", "collection"]
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    mongo_query = chain.invoke({"query": input_text, "collection": collection_name})
    return mongo_query['text']

def run_mongo_query(db, collection_name, query_dict):
    """Execute MongoDB operations based on the query type"""
    collection = db[collection_name]
    try:
        operation = query_dict.get("operation", "find")
        
        if operation == "find":
            query = query_dict.get("query", {})
            projection = query_dict.get("projection", None)
            sort = query_dict.get("sort", None)
            
            cursor = collection.find(query, projection)
            if sort:
                cursor = cursor.sort(list(sort.items()))
            return list(cursor)
            
        elif operation == "insert":
            documents = query_dict.get("documents", [])
            if not isinstance(documents, list):
                documents = [documents]
            result = collection.insert_many(documents)
            return {"inserted_ids": [str(id) for id in result.inserted_ids]}
            
        elif operation == "update":
            filter_dict = query_dict.get("filter", {})
            update_dict = query_dict.get("update", {})
            many = query_dict.get("many", False)
            
            if many:
                result = collection.update_many(filter_dict, update_dict)
                return {
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count
                }
            else:
                result = collection.update_one(filter_dict, update_dict)
                return {
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count
                }
                
        elif operation == "delete":
            filter_dict = query_dict.get("filter", {})
            many = query_dict.get("many", False)
            
            if many:
                result = collection.delete_many(filter_dict)
            else:
                result = collection.delete_one(filter_dict)
            return {"deleted_count": result.deleted_count}
            
        elif operation == "aggregate":
            pipeline = query_dict.get("pipeline", [])
            return list(collection.aggregate(pipeline))
        
        elif operation == "distinct":
            field = query_dict.get("field")
            filter_query = query_dict.get("query", {})
            
            if not field:
                raise ValueError("Field name is required for distinct operation")
            
            # Use distinct method
            distinct_results = collection.distinct(field, filter=filter_query)
            
            # Convert to list of dictionaries for consistent return type
            return [{"value": value} for value in distinct_results]
            
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
    except Exception as e:
        raise Exception(f"Error executing MongoDB operation: {str(e)}")

def convert_to_dataframe(results):
    """Convert MongoDB results to pandas DataFrame"""
    if isinstance(results, list) and results:
        # Handle distinct results specifically
        if all('value' in item for item in results):
            return pd.DataFrame(results)
        # For find and aggregate operations that return lists
        return pd.json_normalize(results)
    elif isinstance(results, dict):
        # For insert, update, and delete operations that return status
        return pd.DataFrame([results])
    else:
        return pd.DataFrame()

# Streamlit UI setup
st.set_page_config(page_title="iMongo", page_icon="🦜")
st.title("🦜 Chat with MongoDB")

# Sidebar for connection details
uri = st.sidebar.text_input("Enter MongoDB URI", type="password")
db_name = st.sidebar.text_input("Enter database name")
collection_name = st.sidebar.text_input("Enter collection name")

# Initialize session state for database connection
if 'db' not in st.session_state:
    st.session_state.db = None

# Connect to MongoDB
if uri and db_name:
    try:
        if st.session_state.db is None:
            st.session_state.db = connect_mongodb(uri, db_name)
            st.sidebar.success("Successfully connected to MongoDB")
    except Exception as e:
        st.sidebar.error(f"Failed to connect: {str(e)}")

# Main query interface
st.markdown("""
### Example queries you can try:
- "Find all documents where age is greater than 25"
- "Insert a new document with name John and age 30"
- "Update the age to 31 for all documents where name is John"
- "Delete documents where status is inactive"
- "Find average age grouped by department"
""")

input_query = st.text_input("Enter your query in natural language")

if input_query: #and st.session_state.db:
    try:
        # Convert natural language to MongoDB query
        mongo_query = text_to_query(input_query, collection_name)
        
        # Display the translated query
        with st.expander("View MongoDB Query"):
            st.code(mongo_query, language="python")
        
        # Execute the query
        try:
            query_dict = literal_eval(mongo_query)
            results = run_mongo_query(st.session_state.db, collection_name, query_dict)
            
            # Display options
            st.subheader("Query Results")
            display_format = st.radio("Select Display Format:", ["Table", "JSON"], horizontal=True)
            
            if results:
                if display_format == "JSON":
                    if isinstance(results, list):
                        for result in results:
                            st.json(result)
                    else:
                        st.json(results)
                else:  # Table format
                    df = convert_to_dataframe(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button for CSV
                    if not df.empty:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
            else:
                st.info("No results found for the query.")
                
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
elif input_query:
    st.warning("Please connect to MongoDB first")