import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# OpenAI removed — use Hugging Face or Groq providers
import requests


from typing import Optional
try:
    from transformers import pipeline
except Exception:
    pipeline = None



st.set_page_config(
    page_title="Document Analysis App",
    page_icon="📊", 
    layout="wide"
)

# Provider selection: allow user to pick 'openai', 'huggingface' or 'groq' (your Groq API)
PROVIDER = st.sidebar.selectbox(
    "Model provider",
    options=["huggingface", "groq"],
    index=0,
    help="Choose 'huggingface' to use a free HF model, or 'groq' to use your Groq endpoint.")

# Initialize clients based on provider
hf_pipe = None
groq_client = None
if PROVIDER == 'groq':
    # Load Groq API settings from Streamlit secrets
    groq_key = st.secrets.get("GROQ_API_KEY")
    groq_url = st.secrets.get("GROQ_API_URL")
    # Optional: allow specifying the target model name in secrets (Groq model names differ)
    groq_model = st.secrets.get("GROQ_MODEL")
    if groq_key and groq_url:
        groq_client = {"url": groq_url, "key": groq_key, "model": groq_model}
    else:
        groq_client = None
else:
    # Try to create a Hugging Face text2text pipeline (requires transformers installed)
    if pipeline is not None:
        try:
            hf_pipe = pipeline("text2text-generation", model="google/flan-t5-small")
        except Exception:
            hf_pipe = None


def send_groq_request(prompt: str = None, messages: list = None, groq_client: dict = None, max_tokens: int = 500) -> str:
    """Send a request to a generic Groq endpoint. The exact payload shape varies between Groq setups,
    so this helper tries a couple of common response shapes and returns the best-effort text.

    Requires `groq_client` to be a dict with keys `url` and `key` (both strings) taken from Streamlit secrets.
    """
    if groq_client is None:
        raise RuntimeError("Groq client not configured. Set GROQ_API_URL and GROQ_API_KEY in Streamlit secrets.")

    headers = {"Authorization": f"Bearer {groq_client['key']}", "Content-Type": "application/json"}
    url = groq_client['url']

    # Choose payload shape depending on whether the Groq endpoint is OpenAI-compatible
    lower_url = (url or "").lower()
    try:
        if 'openai' in lower_url or 'chat' in lower_url or 'completions' in lower_url:
            # Send OpenAI-style chat/completions payload. Use groq_client model if available.
            groq_model = groq_client.get('model') if isinstance(groq_client, dict) else None
            base = {"messages": messages, "max_tokens": max_tokens} if messages is not None else {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
            if groq_model:
                base["model"] = groq_model
            payload = base
        else:
            # Generic Groq payloads
            if messages is not None:
                payload = {"input": messages}
            else:
                payload = {"prompt": prompt, "max_tokens": max_tokens}

        
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse OpenAI-style response
        if isinstance(data, dict):
            if 'choices' in data:
                choices = data['choices']
                if isinstance(choices, list) and len(choices) > 0:
                    c0 = choices[0]
                    # chat-style response
                    if isinstance(c0, dict) and 'message' in c0 and isinstance(c0['message'], dict):
                        return c0['message'].get('content') or str(c0)
                    # older/completions style
                    return c0.get('text') or str(c0)

            # Generic Groq responses
            if 'output' in data:
                out = data['output']
                if isinstance(out, list) and len(out) > 0:
                    first = out[0]
                    if isinstance(first, dict):
                        return first.get('content') or first.get('result') or str(first)
                    return str(first)

            if 'result' in data:
                return str(data['result'])

        return str(data)
    except requests.HTTPError as http_err:
        # include server response body if available to help debugging
        body = ''
        try:
            body = resp.text
        except Exception:
            pass
        raise RuntimeError(f"Groq HTTP error: {http_err}; response body: {body}")
    except Exception:
        raise


def _prepare_hf_prompt(prompt: str, hf_pipeline, max_new_tokens: int = 128) -> str:
    """Truncate `prompt` so its tokenized length fits the model's input size when reserving
    space for `max_new_tokens` for generation.

    Uses the pipeline's tokenizer when available; falls back to a simple character trim.
    """
    if hf_pipeline is None:
        return prompt

    # Try to get tokenizer and model max length
    tokenizer = getattr(hf_pipeline, 'tokenizer', None)
    model_max_len = None
    try:
        if tokenizer is not None:
            # tokenizer.model_max_length is usually present
            model_max_len = int(getattr(tokenizer, 'model_max_length', None) or getattr(hf_pipeline.model.config, 'n_positions', None) or 512)
    except Exception:
        model_max_len = 512

    if not model_max_len:
        model_max_len = 512

    # Reserve tokens for generation
    reserved = max_new_tokens + 10
    allowed_input_tokens = max(16, model_max_len - reserved)

    # If tokenizer is available, use it to truncate by tokens
    try:
        if tokenizer is not None:
            tokenized = tokenizer(prompt, truncation=True, max_length=allowed_input_tokens)
            input_ids = tokenized.get('input_ids')
            # tokenizer may return list of lists for batch; pick first if necessary
            if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if input_ids:
                truncated = tokenizer.decode(input_ids, skip_special_tokens=True)
                return truncated
    except Exception:
        pass

    # As a safe fallback, truncate by characters (not ideal but prevents token overflow)
    # Estimate average 4 chars per token and trim accordingly
    approx_chars = allowed_input_tokens * 4
    if len(prompt) > approx_chars:
        return prompt[-approx_chars:]
    return prompt

# Session state to store uploaded data
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = None
if 'data_summary' not in st.session_state:
    st.session_state['data_summary'] = None


st.title("📊 Welcome to Data Analysis Tool!")


# Sidebar for file upload
st.sidebar.header("📁 Upload your data")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "json"])
st.sidebar.info("👆 Upload a file to get started")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else \
         pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else \
         pd.read_json(uploaded_file)
        
        # session_state to store dataframe
        st.session_state['dataframe'] = df

        st.sidebar.success(f"✅ File uploaded successfully! Data shape: {df.shape[0]} rows and {df.shape[1]} columns")

        # Display the uploaded file name
        st.sidebar.write(f"Uploaded file: {uploaded_file.name}")

        # Data preview options
        with st.sidebar.expander("Data Preview"):
            st.dataframe(df.head())
        
        # Basic statistics
        with st.sidebar.expander("Basic statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of rows", df.shape[0])
                st.metric("Number of columns", df.shape[1])
            with col2:
                st.metric("Missing values", df.isnull().sum().sum())
                st.metric("Duplicate rows", df.duplicated().sum())
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")


    

# Main chat interface
if st.session_state['dataframe'] is not None:
    
        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                #Re-display any saved figures
                if 'figure' in message:
                    st.pyplot(message['figure'])

        # Chat input
        user_question = st.chat_input("Ask a question about your data...")

        if user_question:
            st.session_state['messages'].append({"role": "user", "content": user_question})
            # Placeholder for app response
            response_placeholder = st.empty()
            response_placeholder.markdown("**App:** Thinking...")
            
            # Here you would integrate your data processing and response generation logic
            # For demonstration, we will just echo the question
            app_response = f"You asked: {user_question}."

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Update the messages with the app response
            st.session_state['messages'].append({"role": "app", "content": app_response})
            response_placeholder.markdown(f"**App:** {app_response}")

            # prepare data context with token optimization
            df = st.session_state['dataframe']
            if len(df) > 100:
                data_context = f"""
                Dataset Shape: {st.session_state.data_summary['shape']}
                Columns:{', '.join(st.session_state.data_summary['columns'])}
                Data types:{st.session_state.data_summary['dtypes']}
                Sample rows: {st.session_state.data_summary['Samples']}
                Basic statistics: {st.session_state.data_summary['Statistics']}
                """
            else:
                data_context = f"""
                Full Dataset:
                {df.to_string()}
                """
            # System prompt for LLM
            system_prompt = f""" You are a data analysis assistant. 
                The user has uploaded a file with the following information:{data_context}
                
                The data is loaded in pandas dataframe called 'df'.
                
                Guidelines:
                - Answer the user's questions clearly and concisely.
                - If the question requires analysis, write Python code using pandas, matplotlib or seaborn to perform the analysis.
                - For visualizations, always use plt.figure() before plotting and include plt.tight_layout() and plt.show() to display the plot.
                - Always validate data before operations ( check for nulls, data types, etc.)
                - If you cant answe due to data limitations explain why.
                - Keep responses focussed on the data and question asked.

                when writing code:
                - Import statements are already done.(Pandas as pd, matplotlib.pyplot as plt, seaborn as sns)
                - The dataframe is already loaded as 'df'.
                - For plots, use plt.figure(figsize=(10,6)) for better display.
                - Always add titles ad labels to plots.
            """
            # Use Hugging Face pipeline if selected and available (free model)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Analyzing your data..."):
                    try:
                        # detect if user explicitly requests code
                        wants_code = any(tok in user_question.lower() for tok in ["code", "python", "pandas", "plot", "visualize", "script", "function", "snippet"]) 
                        if PROVIDER == 'groq':
                            # Use Groq endpoint for code or text
                            try:
                                
                                if wants_code:
                                    code_prompt = (
                                        "Generate a Python code snippet that answers the user's question. "
                                        "The dataframe is available as a pandas DataFrame named 'df'. "
                                        "Use pandas, matplotlib or seaborn as needed, and do not include explanatory text — only the code. "
                                        f"Question: {user_question}\nCode:"
                                    )
                                    groq_out = send_groq_request(prompt=code_prompt, groq_client=groq_client, max_tokens=512)
                                    reply = groq_out
                                    display_reply = f"```python\n{reply}\n```"
                                    #st.markdown(f"**App:** {display_reply}")
                                    message_placeholder.markdown(f"**App:** {display_reply}")
                                    st.session_state['messages'].append({"role": "app", "content": reply})

                                    
                                else:
                                    groq_out = send_groq_request(prompt=system_prompt + "\nQuestion:" + user_question, groq_client=groq_client, max_tokens=512)
                                    reply = groq_out
                                    st.markdown(f"**App:** {reply}")

                                    # trying to execute code if any in the response
                                    if "```python" in reply:
                                        code_blocks = reply.split("```python")
                                        for i in range(1, len(code_blocks)):
                                            code_block = code_blocks[i].split("```")[0]
                                            try:
                                                # create figure for potential plots
                                                plt.figure(figsize=(10,6))

                                                #Execute the code block in a safe namespace
                                                exec_globals = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns}
                                                # Ensure we execute a proper string (not a list) and strip surrounding whitespace
                                                code_to_exec = code_block.strip()
                                                # Execute the code block in a restricted namespace and handle syntax/errors
                                                try:
                                                    exec(code_to_exec, exec_globals)
                                                except SyntaxError as se:
                                                    st.error(f"Syntax error executing code block: {se}")
                                                    st.code(code_to_exec, language='python')
                                                    st.info("You can copy and modify this code to fix the error.")
                                                    # move to next code block
                                                    continue

                                                # Display the plot if created
                                                fig = plt.gcf()
                                                if fig.get_axes():
                                                    st.pyplot(fig)
                                                    # save figure in message for persistence
                                                    st.session_state['messages'].append({"role": "app", "content": reply, "figure": fig})
                                                else:
                                                    st.session_state['messages'].append({"role": "app", "content": reply})
                                                    plt.close()

                                            except Exception as e:
                                                error_type = type(e).__name__
                                                st.error(f"Code execution failed: {error_type}")
                                                
                                                # Provide helpful context based on error type
                                                if "NameError" in str(e):
                                                    st.info("This might mean a column name is misspelled or doesn't exist.")
                                                elif "TypeError" in str(e):
                                                    st.info("This often happens when trying to plot non-numeric data.")
                                                elif "KeyError" in str(e):
                                                    st.info("The specified column might not exist in your dataset.")
                                                else:
                                                    st.info("Try rephrasing your question or check your data format.")
                                                
                                                st.code(code_to_exec, language="python")
                                                st.error(f"Error executing code block: {str(e)}")
                                                st.code(code_block, language='python')
                                                st.info("You can copy and modify this code to fix the error.")
                                            
                                            # Save assistant response to history
                                            st.session_state['messages'].append({"role": "app", "content": reply})
                            
                            except Exception as e:
                                response_placeholder.markdown(f"**App:** Groq error: {str(e)}")
                                st.error(f"Groq error: {str(e)}")
                        elif PROVIDER == 'huggingface' and hf_pipe is not None:
                            # Create a short prompt including the data context to send to the text2text model
                            prompt = f"You are a helpful data assistant. Data context:\n{data_context}\nQuestion: {user_question}\nAnswer:" 
                            # Prepare/truncate prompt to fit model input and reserve space for generation
                            max_new_tokens = 128
                            safe_prompt = _prepare_hf_prompt(prompt, hf_pipe, max_new_tokens=max_new_tokens)
                            out = hf_pipe(safe_prompt, max_new_tokens=max_new_tokens)
                            # pipeline returns a list of dicts with 'generated_text' or 'generated_text'
                            if isinstance(out, list) and len(out) > 0:
                                first = out[0]
                                reply = first.get('generated_text') or first.get('generated_text', None) or str(first)
                            else:
                                reply = str(out)
                            st.markdown(f"**App:** {reply}")
                            st.session_state['messages'].append({"role": "app", "content": reply})
                        else:
                            # No model configured for this provider selection
                            msg = (
                                "No model is available to answer this question. "
                                "Select 'huggingface' in the sidebar to use a local HF model, "
                                "or configure 'groq' with GROQ_API_URL and GROQ_API_KEY in Streamlit secrets."
                            )
                            st.warning(msg)
                            st.session_state['messages'].append({"role": "app", "content": msg})
                    except Exception as e:
                        response_placeholder.markdown(f"**App:** Error: {str(e)}")
                        st.error(f"Error generating response: {str(e)}")

else:    
    # No data uploaded state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Please upload a CSV file to start")
        
        # Example questions
        st.markdown("### 💡 Example questions you can ask:")
        st.markdown("""
        - What are the main trends in my data?
        - Show me a correlation matrix
        - Create a bar chart of the top 10 categories
        - What's the average value by month?
        - Are there any outliers in the price column?
        """)
# footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
💡 Tip: Be specific with your questions for better results | 
🔒 Your data stays private and is not stored
</div>
""", unsafe_allow_html=True)