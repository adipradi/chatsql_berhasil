import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
import os
import pandas as pd

# === Load environment ===
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")

if not api_key or not base_url:
    raise ValueError("❌ OPENROUTER_API_KEY dan OPENROUTER_BASE_URL belum diset!")

# Konfigurasi ke LangChain
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url

# === Streamlit UI ===
st.set_page_config(page_title="Chat MySQL + OpenRouter", page_icon="🧠")
st.title("🧠 Chatbot Analitik + SQL (OpenRouter)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Halo! Saya siap bantu jawab pertanyaanmu, baik tentang database maupun lainnya."),
    ]
if "db" not in st.session_state:
    st.session_state.db = None

# === Sidebar: Database connection ===
with st.sidebar:
    st.subheader("🔌 Koneksi Database")
    host = st.text_input("Host", value="localhost")
    port = st.text_input("Port", value="3306")
    user = st.text_input("User", value="root")
    password = st.text_input("Password", type="password", value="admin")
    database = st.text_input("Database", value="Chinook")

    if st.button("Connect"):
        with st.spinner("Menghubungkan ke database..."):
            try:
                db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
                db = SQLDatabase.from_uri(db_uri)
                st.session_state.db = db
                st.success("✅ Terhubung ke database!")
            except Exception as e:
                st.error(f"❌ Koneksi gagal: {e}")

# === Classify query intent ===
def classify_query_type(question: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
Classify the user question into one of:
- sql
- recommendation
- logic
- general
- unknown

Respond ONLY with the label."""),
            ("human", "{question}")
        ])
        llm = ChatOpenAI(model="mistralai/mistral-7b-instruct", temperature=0.2)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question}).strip().lower()
    except Exception:
        return "unknown"

# === SQL generation chain ===
def get_sql_chain(db):
    prompt = ChatPromptTemplate.from_template("""
You are a SQL expert. Based on the schema and chat history, generate a SQL query only.

<SCHEMA>{schema}</SCHEMA>
Conversation History: {chat_history}
Question: {question}
SQL Query:
""")
    llm = ChatOpenAI(model="meta-llama/llama-3.3-8b-instruct:free", temperature=0)
    return (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | prompt
        | llm
        | StrOutputParser()
    )

# === Response logic ===
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    query_type = classify_query_type(user_query)
    llm = ChatOpenAI(model="deepseek/deepseek-chat:free", temperature=0.4)

    if query_type == "sql":
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        sql_result = db.run(sql_query)

        # Tampilkan tabel jika memungkinkan
        try:
            df = pd.DataFrame(sql_result)
            st.dataframe(df)
        except:
            pass

        explain_prompt = ChatPromptTemplate.from_template("""
You're a helpful assistant. Explain the SQL result.

Schema: {schema}
Question: {question}
SQL Query: {query}
SQL Result: {response}
""")
        return (
            explain_prompt | llm | StrOutputParser()
        ).invoke({
            "question": user_query,
            "query": sql_query,
            "response": sql_result,
            "schema": db.get_table_info()
        })

    elif query_type in ["recommendation", "logic"]:
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        sql_result = db.run(sql_query)

        insight_prompt = ChatPromptTemplate.from_template("""
You're a strategic assistant. Give analysis or advice from:

- Schema: {schema}
- Question: {question}
- SQL: {query}
- Result: {response}
""")
        return (
            insight_prompt | llm | StrOutputParser()
        ).invoke({
            "question": user_query,
            "query": sql_query,
            "response": sql_result,
            "schema": db.get_table_info()
        })

    elif query_type == "general":
        return llm.invoke(user_query).content

    else:
        return "🤖 Saya belum yakin bagaimana menjawabnya. Bisa coba ditanyakan ulang?"

# === Display chat history ===
for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)

# === Chat input ===
user_query = st.chat_input("Tulis pertanyaan...")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if st.session_state.db is None:
            response = "❗ Silakan hubungkan ke database dulu."
        else:
            try:
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            except Exception as e:
                response = f"⚠️ Error: {e}"

        st.markdown(response)
        st.session_state.chat_history.append(AIMessage(content=response))
