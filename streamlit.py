import streamlit as st
from streamlit_option_menu import option_menu
import requests
import os
import logging
from groq import Groq
from langchain_community.vectorstores import FAISS
from urllib.parse import quote
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings.laser import LaserEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from downloader_service.gcp_handler import GCPStorageManager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
import pyrebase



load_dotenv()


from downloader_service.downloader import PaperDownloader 
SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
FIREBASE_CREDENTIALS=os.getenv('FIREBASE_CREDENTIALS')
FIREBASE_API_KEY=os.getenv('FIREBASE_WEB_API_KEY')




cred = credentials.Certificate(FIREBASE_CREDENTIALS)


groq_model = ChatGroq(
    temperature=0.3,
    model="llama3-70b-8192",
    api_key= GROQ_API_KEY,
)



embedding_huggingface = HuggingFaceBgeEmbeddings(model_name="nomic-ai/nomic-embed-text-v1")












def sign_in_with_email_and_password(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error(f"Sign-in failed: {response.json().get('error', {}).get('message', 'Unknown error')}")
        return None

def create_user_with_email_and_password(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error(f"Sign-up failed: {response.json().get('error', {}).get('message', 'Unknown error')}")
        return None

def welcome_page():
    st.markdown(
        """
        <style>
        .centered-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #c24a4a;
            margin-bottom: 30px;
        }
        </style>
        <h1 class="centered-title">Login/Signup</h1>
        """,
        unsafe_allow_html=True
    )

    choice = st.sidebar.selectbox('Login or Signup', ['Login', 'Signup'])

    if choice == 'Signup':
        email = st.text_input('Enter your email', key='signup_email')
        password = st.text_input('Enter your password', type='password', key='signup_password')
        username = st.text_input('Enter your username', key='signup_username')
        if st.button('Create Account'):
            user_data = create_user_with_email_and_password(email, password)
            if user_data:
                st.success('Account created successfully')
                st.balloons()

    if choice == 'Login':
        email = st.text_input('Enter your email', key='login_email')
        password = st.text_input('Enter your password', type='password', key='login_password')
        if st.button('Login'):
            user_data = sign_in_with_email_and_password(email, password)
            if user_data:
                st.session_state['user'] = user_data
                st.success('Logged in successfully')


























def find_academic_papers_page():
    if 'user' not in st.session_state:
        st.error("Please login first")
        return


    st.title("Download Academic Papers")
    st.write("This page allows you to download academic papers based on a search query and upload them to a specific folder in GCP.")

    # Initialize the GCPStorageManager
    gcp_manager = GCPStorageManager(GOOGLE_APPLICATION_CREDENTIALS)

    # Fetch existing directories
    existing_files = gcp_manager.list_files(GCS_BUCKET_NAME)
    directories = set()
    for file in existing_files:
        directories.add(os.path.dirname(file))

    directories = list(directories)
    directories.append("Create new directory")

    selected_directory = st.selectbox("Choose or create a directory", directories)

    if selected_directory == "Create new directory":
        new_directory = st.text_input("Enter new directory name")
        if new_directory:
            selected_directory = new_directory
            # Create the new directory in GCP
            gcp_manager.create_folder(GCS_BUCKET_NAME, f"{new_directory}/")

    query = st.text_input("Enter search query")
    n = st.number_input("Number of papers to download", min_value=1, value=20)

    if st.button("Download Papers"):
        if query and selected_directory:
            # Instantiate the PaperDownloader
            paper_downloader = PaperDownloader(SEMANTIC_SCHOLAR_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, GCS_BUCKET_NAME)
            try:
                # Download the papers
                downloaded_papers = paper_downloader.download_top_papers(query=query, n=1000, limit=n, directory=selected_directory)
                st.success("Downloaded papers successfully.")
                for paper in downloaded_papers:
                    st.markdown(f"[{paper['title']}]({paper['file_path']})")
            except PermissionError as pe:
                st.error(f"Request failed: {pe}")
            except Exception as e:
                st.error(f"Request failed: {e}")
        else:
            st.error("Please fill in all fields")

    # File uploader to allow users to upload new papers to GCS
    uploaded_file = st.file_uploader("Upload a new paper", type=["pdf"])
    if uploaded_file and selected_directory:
        try:
            destination_file_name = f"{selected_directory}/{uploaded_file.name}"
            content = uploaded_file.read()
            public_url = gcp_manager.upload_content(GCS_BUCKET_NAME, content, destination_file_name)
            st.success(f"Uploaded {uploaded_file.name} successfully to {selected_directory}! [View file]({public_url})")
        except Exception as e:
            st.error(f"Failed to upload file to GCS: {e}")



















def extract_details():

    if 'user' not in st.session_state:
        st.error("Please login first")
        return
    

    gcp_manager = GCPStorageManager(GOOGLE_APPLICATION_CREDENTIALS)

    st.title("Extract Experimental Details of Academic Papers")
    st.write("This page allows you to extract experimental details; including population size with characteristics, experiment type, Effect sizes with metrics, p values from academic papers in the selected directory.")

    try:
        files = gcp_manager.list_files(GCS_BUCKET_NAME)
    except Exception as e:
        st.error(f"Failed to list files in the GCS bucket: {e}")
        return

    if not files:
        st.info("No papers found in the bucket.")
        return

    # Extract directories from the list of files
    directories = set()
    for file in files:
        directory = os.path.dirname(file)
        if directory:
            directories.add(directory)
    
    directories = list(directories)
    selected_directory = st.selectbox("Select a directory", directories)

    if not selected_directory:
        st.info("No directory selected.")
        return

    filtered_files = [file for file in files if file.startswith(selected_directory) and file.endswith(".pdf")]
    if not filtered_files:
        st.info("No PDF files found in the selected directory.")
        return

    for file in filtered_files:
        file_name = os.path.basename(file)
        file_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{quote(file)}"
        st.write(f"File: [{file_name}]({file_url})")
        
        if st.button(f"Extract Details from {file_name}"):
            with st.spinner(f"Processing file: {file_name}"):
                try:
                    pdf_loader = PyMuPDFLoader(file_url)
                    documents = pdf_loader.load()

                    if documents:
                        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=200)
                        split_documents = text_splitter.split_documents(documents)

                        # Using FAISS for vector storage
                        vectorstore = FAISS.from_documents(
                            documents=split_documents,
                            embedding=embedding_huggingface,
                        )
                        # create a Chroma vectorstore



                        retriever_X = vectorstore.as_retriever()
                        # model_local = ChatOpenAI(base_url="http://localhost:1234/v1", api_key='lm-studio', streaming=True, temperature=0.3, model_kwargs={"top_p": 0.9})

                        after_rag_template = """
                        provide experimental results in given JSON format for the questions below based on ONLY the below context:
                        Important: DO NOT CHANGE THE JSON FORMAT

                        {context}

                        JSON output:
                        "
                        {{
                        "population size": (Sample size of the experiment),
                        "population_characteristics": (Ethnicity, Age, country of the population, etc),
                        "study_design": (Randomized Controlled Trial, Observational Study, etc),
                        "research_objective": (What is the research objective of the study?),
                        "treatment_used": (Main treatment/intervention used in the study),
                        "outcome_variable": [
                            {{
                            "variable name": (Main outcome variable),
                            "effect_size": (Numenric value of the effect size with the metric used; example metrics: Odds Ratio, Relative Risk, etc.),
                            "p_value": (p value of the effect size),
                            "confidence_interval": (95 percent confidence interval of the effect size),
                            }}
                        ]
                        }}
                        "
                        If you are unable to find the information for a given key of JSON format, please respond with "Information not available" for that key.
                        DONT SKIP OR CHANGE ANY KEY IN THE JSON FORMAT.

                        """

                        after_rag_prompt_X = ChatPromptTemplate.from_template(after_rag_template)

                        after_rag_chain = (
                            {"context": retriever_X, "question": RunnablePassthrough()}
                            | after_rag_prompt_X
                            | groq_model
                            | StrOutputParser()
                        )

                        question = "Provide experimental details IN JSON FORMAT."
                        summary = after_rag_chain.stream(question)
                        with st.chat_message("assistant"):
                            st.write_stream(summary)
                    else:
                        st.info("No documents found to process.")
                except Exception as e:
                    st.error(f"Failed to process file {file_name}: {e}")









def summary_page():
    if 'user' not in st.session_state:
        st.error("Please login first")
        return

    gcp_manager = GCPStorageManager(GOOGLE_APPLICATION_CREDENTIALS)

    st.title("Summary of Academic Papers")
    st.write("This page provides a summary of all academic papers in the selected directory.")

    try:
        files = gcp_manager.list_files(GCS_BUCKET_NAME)
    except Exception as e:
        st.error(f"Failed to list files in the GCS bucket: {e}")
        return

    if not files:
        st.info("No papers found in the bucket.")
        return

    # Extract directories from the list of files
    directories = set()
    for file in files:
        directory = os.path.dirname(file)
        if directory:
            directories.add(directory)
    
    directories = list(directories)
    selected_directory = st.selectbox("Select a directory", directories)

    if not selected_directory:
        st.info("No directory selected.")
        return

    filtered_files = [file for file in files if file.startswith(selected_directory) and file.endswith(".pdf")]
    if not filtered_files:
        st.info("No PDF files found in the selected directory.")
        return

    for file in filtered_files:
        file_name = os.path.basename(file)
        file_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{quote(file)}"
        st.write(f"File: [{file_name}]({file_url})")
        
        if st.button(f"Generate Summary for {file_name}"):
            with st.spinner(f"Processing file: {file_name}"):
                try:
                    pdf_loader = PyMuPDFLoader(file_url)
                    documents = pdf_loader.load()

                    if documents:
                        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=200)
                        split_documents = text_splitter.split_documents(documents)

                        # Using FAISS for vector storage
                        vectorstore = FAISS.from_documents(
                            documents=split_documents,
                            embedding=embedding_huggingface,
                        )

                        retriever = vectorstore.as_retriever()
                        # model_local_X = ChatOpenAI(base_url="http://localhost:1234/v1", api_key='lm-studio', streaming=True, temperature=0.7)

                        after_rag_template = """Provide a summary for the research paper with words between less than 150 based only on the following context, ALWAYS break down the summary into important topics and provide a concise summary for each topic.:
                            {context}
                            Add some links with further readings at the end if needed.
                            Question: {question}
                            """

                        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

                        after_rag_chain = (
                            {"context": retriever, "question": RunnablePassthrough()}
                            | after_rag_prompt
                            | groq_model
                            | StrOutputParser()
                        )

                        question = "Provide a summary of the research paper."
                        details = after_rag_chain.stream(question)
                        with st.chat_message("assistant"):
                            st.write_stream(details)
                    else:
                        st.info("No documents found to process.")
                except Exception as e:
                    st.error(f"Failed to process file {file_name}: {e}")











































# Function to create the Chatbot page
def chatbot_page():


    st.markdown(
        """
        <style>
        .centered-chat-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: white;
            margin-bottom: 20px;
        }
        </style>
        <h1 class="centered-chat-title">Academic Pal</h1>
        """,
        unsafe_allow_html=True
    )

    if 'user' not in st.session_state:
        st.error("Please login first")
        return

    # llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0.3, model_kwargs={"top_p": 0.9})

    gcp_manager = GCPStorageManager(GOOGLE_APPLICATION_CREDENTIALS)

    # Session state for messages and OpenAI model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Select directory
    try:
        files = gcp_manager.list_files(GCS_BUCKET_NAME)
    except Exception as e:
        st.error(f"Failed to list files in the GCS bucket: {e}")
        return


    if not files:
        st.info("No papers found in the bucket.")
        return

    directories = set(os.path.dirname(file) for file in files if file.endswith('.pdf'))
    selected_directory = st.selectbox("Select a directory", list(directories))
        


    


        
    filtered_files = [file for file in files if file.startswith(selected_directory) and file.endswith(".pdf")]

    all_documents = []
    for file in filtered_files:
        file_path = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{quote(file)}"
        pdf_loader = PyMuPDFLoader(file_path)
        documents = pdf_loader.load()
        all_documents.extend(documents)
        
    if all_documents:
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            embedding = embedding_huggingface
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            doc_chunks = text_splitter.split_documents(all_documents)

            vectorstore = FAISS.from_documents(
                documents=doc_chunks,
                embedding=embedding,
            )
            retriever = vectorstore.as_retriever()

            # langchain_client = ChatOpenAI(base_url="http://localhost:1234/v1", api_key='lm-studio', temperature=0.3, model_kwargs={"top_p": 0.9})

            template = """Answer the question based only on the following context: Always answer pointwise and in a concise manner. If needed provide any links for further readings at the end.
            {context}
            Question: {question}
            """
            prompt_template = ChatPromptTemplate.from_template(template)

            memory = ConversationSummaryMemory(
                llm=groq_model, memory_key="chat_history", return_messages=True
            )

            final_rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | groq_model
                | StrOutputParser()
            )

            


            details = final_rag_chain.stream(prompt)

            with st.chat_message("assistant"):
                response = st.write_stream(details)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("No documents found to process.")
    





























# Main function to set up the Streamlit app
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Welcome"
    
    # Session state for messages and OpenAI model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user" not in st.session_state:
        welcome_page()
    else:
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=["Welcome", "Find Academic Papers", "Extract Experimental Details", "Generate Summaries", "Chatbot"],
                icons=["house", "search", "list-task", "book", "robot"],
                menu_icon="cast",
                default_index=0,
            )

        if selected == "Welcome":
            welcome_page()
        elif selected == "Find Academic Papers":
            find_academic_papers_page()
        elif selected == "Extract Experimental Details":
            extract_details()
        elif selected == "Generate Summaries":
            summary_page()
        elif selected == "Chatbot":
            chatbot_page()

if __name__ == "__main__":
    main()
