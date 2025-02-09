import os
from dotenv import load_dotenv
load_dotenv()
import pyodbc
import base64
import pandas as pd
import streamlit as st
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import StreamlitCallbackHandler

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="iSQL", page_icon="🦜")
st.title("🦜 Chat with SQL DB")

#Function to test SQL server connectivity
def test_sql_server():
    #server = SQL_SERVER+'.database.windows.net'  # for Azure cloud based SQL server
    #server = SQL_SERVER+'\SQLEXPRESS'
    server = SQL_SERVER
    database = SQL_DB
    #Note: Username & Password are not required in case of SSO (windows credential) based login to SQL server
    #username = SQL_USERNAME
    #password = SQL_PWD
    driver = '{ODBC Driver 17 for SQL Server}'

    cursor = pyodbc.connect(f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;").cursor()
    return cursor


#Providing option to choose SQL or file upload
LOCALDB="USE_LOCALDB"
SQLSERVER="USE_SQLSERVER"
FILEUPLOAD = "USE_CSV"
radio_opt=["Use SQLLite 3 Database: Student.db", "Connect to SQL server Database", "File upload"]
selected_opt=st.sidebar.radio(label="Choose the DB to which you want to chat", options=radio_opt)

#Test connectivity based on chosen option
if radio_opt.index(selected_opt)==0:
    db_uri = LOCALDB
elif radio_opt.index(selected_opt)==1:
    db_uri = SQLSERVER
    #SQL_SERVER = st.sidebar.text_input("Enter SQL server name")
    #server = os.getenv("SQL_SERVER")+'\SQLEXPRESS'
    #SQL_DB = st.sidebar.text_input("Enter database name")
    #SQL_USERNAME = st.sidebar.text_input("Enter Username")
    #SQL_PWD = st.sidebar.text_input("Enter Password")
    server = 'LP1-AP-52203667\SQLEXPRESS'
    database = 'Sales'

    #conn_status = st.sidebar.button("Connect", on_click=test_sql_server)
    conn_status = st.sidebar.button("Connect")
    try:
        if conn_status:
            driver = '{ODBC Driver 17 for SQL Server}'
            connection = pyodbc.connect(
                f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;")
            cursor = connection.cursor()
            st.sidebar.success("Connection is successful")
    except pyodbc.Error as e:
        st.exception(e)
elif radio_opt.index(selected_opt)==2:
    db_uri = FILEUPLOAD
    uploaded_file = st.file_uploader("Please upload file", type={"csv"})
    if uploaded_file is not None:
        #Read csv as dataframe
        df = pd.read_csv(uploaded_file)
        """
        if not all(key in st.secrets for key in ["SQL_SERVER", "SQL_DB", "SQL_USERNAME", "SQL_PWD"]):
                st.error("Missing SQL connection details in `st.secrets or .env`.")
                st.stop()
        """

        #Connection details
                      
        #server = st.secrets["SQL_SERVER"]+'.database.windows.net'
        SQL_SERVER = os.getenv("SQL_SERVER")   
        #server = os.getenv("SQL_SERVER")+'\SQLEXPRESS'
        SQL_DB = os.getenv("SQL_DB")
        #SQL_USERNAME = os.getenv("SQL_USERNAME")
        #SQL_PWD = os.getenv("SQL_PWD")
        driver = 'ODBC Driver 17 for SQL Server'
        try:
            """
            connection = create_engine(
                'mssql+pyodbc://' + SQL_USERNAME + ':' + SQL_PWD + '@' + server + '/' + SQL_DB 
                + '?driver=ODBC+Driver+17+for+SQL+Server', 
                echo=False)
            """
            connection = create_engine((f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}?driver={driver.replace(' ', '+')}&trusted_connection=yes"))
            up_status = df.to_sql('uploaded_data', con=connection, if_exists='replace', index=False)
            st.success("File upload is successful")
        except Exception as e:
            st.exception(e)
        


#Function to configure database
@st.cache_resource(ttl="2h")
def db_configure(db_uri, SQL_SERVER='LP1-AP-52203667\SQLEXPRESS', SQL_USERNAME=None, SQL_PWD=None, SQL_DB='Sales'):
    if db_uri==LOCALDB:
        dbfilepath = (Path(__file__).parent/"student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    if db_uri==SQLSERVER:
        driver = 'ODBC Driver 17 for SQL Server'
        server = SQL_SERVER
        database = SQL_DB
        odbc_str = (f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes")
        return SQLDatabase(create_engine(odbc_str))
    if db_uri==FILEUPLOAD:
        driver = 'ODBC Driver 17 for SQL Server'
        #odbc_str = 'mssql+pyodbc:///?odbc_connect=' + 'Driver='+driver+ ';Server=' + SQL_SERVER +'\\SQLEXPRESS' +';PORT=1433' + ';DATABASE=' + SQL_DB + ';Uid=' + SQL_USERNAME+ ';Pwd=' + SQL_PWD + ';Encrypt=yes;TrustServerCertificate=no;'
        odbc_str = (f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}?driver={driver.replace(' ', '+')}&trusted_connection=yes")
        return SQLDatabase(create_engine(odbc_str))
    
if db_uri==SQLSERVER:
    server = 'LP1-AP-52203667\SQLEXPRESS'
    database = 'Sales'
    db = db_configure(db_uri, server, database)
elif db_uri==FILEUPLOAD:
    SQL_SERVER = os.getenv("SQL_SERVER")   
    #server = os.getenv("SQL_SERVER")+'\SQLEXPRESS'
    SQL_DB = os.getenv("SQL_DB")
    db = db_configure(db_uri, SQL_SERVER, SQL_DB)
else:
    db = db_configure(db_uri)


# LLM initialization
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Creating SQL toolkit
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_toolkit.get_tools()

# Initialize SQL agent
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Get user query and provide response
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=sql_agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
