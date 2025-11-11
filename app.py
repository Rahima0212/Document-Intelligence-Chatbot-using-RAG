import os
import uuid
import json
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.ocr import extract_text_from_uploaded_file
from src.utils import chunk_text
from src.storage import ChromaStore
from src.gemini_client import GeminiClient
from src.db import LocalStorage

# Page config
st.set_page_config(
    page_title='RAG Chat',
    page_icon='💭',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize components
if 'storage' not in st.session_state:
    st.session_state.storage = LocalStorage()
if 'chroma_store' not in st.session_state:
    st.session_state.chroma_store = ChromaStore()
if 'gemini' not in st.session_state:
    st.session_state.gemini = GeminiClient(debug=True)
if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = None

# Sidebar
with st.sidebar:
    st.title("💭 RAG Chat")
    
    # New Chat button
    if st.button("🆕 New Chat", use_container_width=True):
        # Generate a new conversation ID
        new_conv_id = str(uuid.uuid4())
        # Create initial title
        initial_title = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        st.session_state.storage.create_conversation(new_conv_id, initial_title)
        st.session_state.current_conversation = new_conv_id
        st.rerun()

    # List of conversations
    st.subheader("Your Conversations")
    conversations = st.session_state.storage.get_conversations()
    
    for conv in conversations:
        # Create a unique key for each button
        btn_key = f"conv_btn_{conv['id']}"
        if st.button(
            f"📝 {conv['title']}", 
            key=btn_key,
            use_container_width=True,
            type="secondary" if conv['id'] != st.session_state.current_conversation else "primary"
        ):
            st.session_state.current_conversation = conv['id']
            st.rerun()
    
    # API Status section at the bottom
    st.markdown("---")
    with st.expander("🔧 API Status"):
        status = st.session_state.gemini.get_status()
        if status['available']:
            st.success("✅ Gemini API configured")
        else:
            st.error("❌ Gemini API not configured")
        for key, value in status.items():
            if key != 'api_key_masked':
                st.text(f"{key}: {value}")

# Main chat area
if st.session_state.current_conversation:
    # Get current conversation details
    conversations = {c['id']: c for c in st.session_state.storage.get_conversations()}
    current_conv = conversations[st.session_state.current_conversation]
    
    # Chat title
    st.title(current_conv['title'])
    
    # Document upload section
    with st.expander('📄 Upload Document'):
        uploaded = st.file_uploader(
            'Upload PDF, DOCX, TXT, image',
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key=f"uploader_{current_conv['id']}"
        )
        if uploaded is not None:
            with st.status("Processing document...", expanded=True) as status:
                # Extract text
                text = extract_text_from_uploaded_file(uploaded)
                if not text:
                    st.warning('No text extracted from file.')
                    status.update(label="Failed to extract text", state="error")
                else:
                    status.update(label="Chunking text...", state="running")
                    # Process chunks
                    chunks = chunk_text(text, chunk_size=1000, overlap=200)
                    doc_id = str(uuid.uuid4())
                    chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                    metadatas = [{
                        'source': uploaded.name,
                        'chunk_index': i,
                        'conversation_id': current_conv['id']
                    } for i in range(len(chunks))]
                    
                    status.update(label="Storing in database...", state="running")
                    # Store in ChromaDB
                    st.session_state.chroma_store.add_documents(chunks, metadatas, chunk_ids)
                    
                    # Store document reference
                    st.session_state.storage.add_document(
                        current_conv['id'],
                        doc_id,
                        uploaded.name,
                        {'num_chunks': len(chunks)}
                    )
                    
                    status.update(label=f"Ready to chat", state="complete")
    
    # Display conversation
    messages = st.session_state.storage.get_conversation_messages(current_conv['id'])
    for msg in messages:
        # Normalize role to 'user' or 'assistant' for Streamlit chat
        raw_role = msg.get('role', 'assistant')
        role = raw_role.lower() if isinstance(raw_role, str) else 'assistant'
        role = 'user' if role == 'user' else 'assistant'

        with st.chat_message(role):
            st.markdown(msg['content'])
    
    # Chat input with streaming assistant response
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to storage and display it immediately
        user_msg_id = str(uuid.uuid4())
        st.session_state.storage.add_message(
            current_conv['id'],
            user_msg_id,
            'user',
            prompt
        )

        with st.chat_message('user'):
            st.markdown(prompt)

        # Retrieve relevant chunks for context
        results = st.session_state.chroma_store.query(
            query_text=prompt,
            where={'conversation_id': current_conv['id']},
            top_k=4
        )

        context_chunks = []
        sources = set()
        for i, doc in enumerate(results.get('documents', [[]])[0]):
            context_chunks.append(doc)
            meta = results.get('metadatas', [[]])[0][i]
            if meta and 'source' in meta:
                sources.add(meta['source'])

        context = "\n---\n".join(context_chunks)

        # Build chat history for the model (last few messages)
        messages = st.session_state.storage.get_conversation_messages(current_conv['id'])
        chat_history = [{'role': m['role'], 'content': m['content']} for m in messages[-6:]]

        system_prompt = (
            "You are a helpful assistant. Use the provided context from documents to answer questions. "
            "If the answer cannot be found in the context, say so clearly. "
            "When you use information from the context, be specific about which document it came from."
        )

        # Create an assistant chat container and show a loading indicator
        with st.chat_message('assistant'):
            assistant_placeholder = st.empty()
            loading_placeholder = st.empty()
            loading_placeholder.text('⏳ Thinking...')

        # Try streaming from Gemini if available
        accumulated = ""
        try:
            if getattr(st.session_state.gemini, 'generate_chat_completion_stream', None) and st.session_state.gemini.available:
                stream = st.session_state.gemini.generate_chat_completion_stream(
                    messages=chat_history + [{'role': 'user', 'content': prompt}],
                    system_prompt=f"{system_prompt}\n\nContext:\n{context}"
                )

                for chunk in stream:
                    text = chunk.get('text') if isinstance(chunk, dict) else None
                    if not text:
                        # fallback to stringifying the chunk
                        try:
                            text = str(chunk)
                        except Exception:
                            text = ''
                    accumulated += text or ''
                    # Update assistant message in-place
                    assistant_placeholder.markdown(accumulated)

            else:
                # Fallback to non-streaming call
                response = st.session_state.gemini.generate_chat_completion(
                    messages=chat_history + [{'role': 'user', 'content': prompt}],
                    system_prompt=f"{system_prompt}\n\nContext:\n{context}"
                )
                accumulated = response["choices"][0]["message"]["content"]
                assistant_placeholder.markdown(accumulated)

        except Exception as e:
            assistant_placeholder.markdown(f"Error while generating response: {e}")
            accumulated = f"Error: {e}"

        finally:
            # Remove loading indicator
            try:
                loading_placeholder.empty()
            except Exception:
                pass

        # Store assistant message with metadata
        assistant_msg_id = str(uuid.uuid4())
        st.session_state.storage.add_message(
            current_conv['id'],
            assistant_msg_id,
            'assistant',
            accumulated,
            metadata={'sources': list(sources)} if sources else None
        )
        
        # Schedule title generation for the next Streamlit run after the response is complete
        if 'last_msg_id' not in st.session_state:
            st.session_state.last_msg_id = None
        
        # Only update title if this is a new message
        if st.session_state.last_msg_id != assistant_msg_id:
            st.session_state.last_msg_id = assistant_msg_id
            st.session_state.should_update_title = True
        
        # Ensure UI shows the stored message first
        st.rerun()

    # If we should update the title, do it in a separate run after the message is displayed
    elif getattr(st.session_state, 'should_update_title', False):
        try:
            # Clear the flag first to prevent loops
            st.session_state.should_update_title = False
            
            messages = st.session_state.storage.get_conversation_messages(current_conv['id'])
            if len(messages) >= 2:
                # Add a small delay before title generation
                time.sleep(1)
                
                recent_msgs = messages[-4:]
                conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
                title_response = st.session_state.gemini.generate_chat_completion(
                    messages=[{
                        "role": "user", 
                        "content": "Generate a very short, descriptive title (3-5 words) for this conversation. "
                                 "Title should be clear and informative, focusing on the main topic or question. "
                                 "Just return the title text, nothing else.\n\n" + conv_text
                    }]
                )
                new_title = title_response["choices"][0]["message"]["content"].strip()
                if 2 < len(new_title.split()) < 8:
                    st.session_state.storage.update_conversation_title(current_conv['id'], new_title)
                    st.rerun()  # One final rerun to show the new title
        except Exception as e:
            print(f"Title generation error (non-critical): {e}")
            st.session_state.should_update_title = False
else:
    # No conversation selected - show welcome message
    st.title("👋 Welcome to RAG Chat!")
    st.markdown("""
        Click the **🆕 New Chat** button to start a conversation.
        
        You can:
        - Upload documents (PDF, DOCX, TXT, images)
        - Chat with your documents using Gemini AI
        - See conversation history in the sidebar
        - Auto-generate chat titles
        - View source documents for answers
    """)

if __name__ == '__main__':
    # The main app logic is already handled in the initialization and UI sections above
    pass
