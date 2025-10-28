import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from typing import List
import re

import pypdf

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot with Groq API")

# Improved retriever with better matching
class ImprovedRetriever(BaseRetriever):
    documents: List[Document]
    k: int = 6
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Enhanced retrieval with better scoring"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scores = []
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            
            # Multiple scoring factors
            # 1. Exact phrase match (highest priority)
            exact_match_score = 100 if query_lower in content_lower else 0
            
            # 2. Word overlap score
            content_words = set(content_lower.split())
            word_overlap = len(query_words.intersection(content_words))
            
            # 3. Keyword density
            keyword_count = sum(content_lower.count(word) for word in query_words)
            
            # 4. Boost metadata chunks
            is_metadata = doc.metadata.get("type") == "metadata"
            metadata_boost = 50 if is_metadata else 0
            
            # Combined score
            total_score = exact_match_score + (word_overlap * 10) + keyword_count + metadata_boost
            
            scores.append((doc, total_score))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scores[:self.k]]

def load_pdf_fast(file_path: str, max_pages: int = 50) -> tuple:
    """Fast PDF loader that also extracts metadata"""
    documents = []
    pdf_title = "Unknown Document"
    
    try:
        with open(file_path, 'rb') as file:
            pdf = pypdf.PdfReader(file)
            total_pages = min(len(pdf.pages), max_pages)
            
            # Try to extract PDF title from metadata
            if pdf.metadata and pdf.metadata.title:
                pdf_title = pdf.metadata.title
            
            for i in range(total_pages):
                try:
                    text = pdf.pages[i].extract_text()
                    if text and len(text.strip()) > 50:
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "page": i + 1,
                                "total_pages": total_pages,
                                "source": pdf_title
                            }
                        ))
                except:
                    continue
                    
    except Exception as e:
        st.error(f"PDF reading error: {str(e)}")
    
    return documents, pdf_title

def simple_split(documents: List[Document], chunk_size: int = 800) -> List[Document]:
    """Smaller chunks for better retrieval"""
    chunks = []
    for doc in documents:
        text = doc.page_content
        # Keep sentences together when possible
        sentences = re.split(r'[.!?]\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=doc.metadata
                    ))
                current_chunk = sentence + ". "
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=doc.metadata
            ))
    
    return chunks

def extract_document_info(documents: List[Document]) -> dict:
    """Extract title and other metadata from first few pages"""
    info = {
        "title": "Unknown Document",
        "chapters": []
    }
    
    if not documents:
        return info
    
    # Check first page for title
    first_page_text = documents[0].page_content
    lines = first_page_text.strip().split('\n')
    
    # The first non-empty line is often the title
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if len(line) > 5 and len(line) < 200:  # Reasonable title length
            # Skip lines that look like metadata
            if not any(skip in line.lower() for skip in ['page', 'author:', 'by:', 'date:', 'copyright']):
                info["title"] = line
                break
    
    # Extract chapter information from all documents
    for doc in documents:
        text = doc.page_content
        # Look for chapter headings (case insensitive)
        chapter_matches = re.findall(r'CHAPTER\s+(\d+)[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
        for match in chapter_matches:
            chapter_num = match[0]
            chapter_name = match[1].strip()[:100]  # Limit length
            info["chapters"].append({
                "number": chapter_num,
                "name": chapter_name,
                "page": doc.metadata.get("page", "Unknown")
            })
    
    return info

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    st.divider()
    st.header("📄 Upload PDF")
    
    max_pages = st.number_input("Max pages to process", 1, 200, 50, 5)
    uploaded = st.file_uploader("Choose PDF", type=['pdf'])
    
    if uploaded:
        if st.button("🚀 Process Document"):
            st.session_state.process_trigger = True
            st.session_state.uploaded_file = uploaded
    
    st.divider()
    model = st.selectbox("Model", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"])
    
    if st.button("🗑️ Reset"):
        st.session_state.clear()
        st.rerun()

# Session state initialization
if 'msgs' not in st.session_state:
    st.session_state.msgs = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'process_trigger' not in st.session_state:
    st.session_state.process_trigger = False
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
if 'pdf_title' not in st.session_state:
    st.session_state.pdf_title = ""
if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0
if 'num_pages' not in st.session_state:
    st.session_state.num_pages = 0
if 'chapter_info' not in st.session_state:
    st.session_state.chapter_info = []

# Process PDF only when button clicked
if st.session_state.process_trigger and not st.session_state.processing_done:
    st.session_state.process_trigger = False
    
    placeholder = st.empty()
    
    with placeholder.container():
        st.info("📖 Processing PDF...")
        progress = st.progress(0)
        status = st.empty()
        
        try:
            uploaded_file = st.session_state.uploaded_file
            
            status.text("Step 1/5: Saving file...")
            progress.progress(20)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                f.write(uploaded_file.read())
                path = f.name
            
            status.text(f"Step 2/5: Reading PDF (max {max_pages} pages)...")
            progress.progress(35)
            
            docs, pdf_title = load_pdf_fast(path, max_pages=max_pages)
            
            if not docs:
                st.error("❌ No text found in PDF. Try another file.")
                os.unlink(path)
                st.stop()
            
            # Extract document information
            status.text("Step 3/5: Analyzing document structure...")
            progress.progress(50)
            
            doc_info = extract_document_info(docs)
            st.session_state.pdf_title = doc_info["title"]
            st.session_state.chapter_info = doc_info["chapters"]
            
            status.text(f"Step 4/5: Splitting {len(docs)} pages...")
            progress.progress(65)
            
            chunks = simple_split(docs, chunk_size=800)
            
            # Add document metadata to first chunk for better retrieval
            if chunks:
                metadata_text = f"Document Title: {doc_info['title']}\n\n"
                
                if doc_info['chapters']:
                    metadata_text += f"This document has {len(doc_info['chapters'])} chapters.\n\nChapter List:\n"
                    metadata_text += "\n".join([f"Chapter {ch['number']}: {ch['name']}" for ch in doc_info['chapters']])
                
                metadata_chunk = Document(
                    page_content=metadata_text,
                    metadata={"page": 0, "type": "metadata"}
                )
                chunks.insert(0, metadata_chunk)
            
            # Store all chunks
            st.session_state.all_chunks = chunks
            
            # Limit for speed if needed
            if len(chunks) > 150:
                retrieval_chunks = chunks[:150]
            else:
                retrieval_chunks = chunks
            
            status.text(f"Step 5/5: Creating retriever with {len(retrieval_chunks)} chunks...")
            progress.progress(85)
            
            # Create improved retriever
            st.session_state.retriever = ImprovedRetriever(documents=retrieval_chunks, k=6)
            st.session_state.num_chunks = len(chunks)
            st.session_state.num_pages = len(docs)
            
            progress.progress(100)
            os.unlink(path)
            
            st.session_state.processing_done = True
            
            placeholder.empty()
            st.sidebar.success(f"✅ Processed successfully!")
            
        except Exception as e:
            placeholder.empty()
            st.error(f"❌ Processing failed: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Show status
if st.session_state.processing_done:
    st.sidebar.success("✅ Document ready!")
    
    # Display extracted information
    if st.session_state.pdf_title != "Unknown Document":
        st.sidebar.info(f"📄 **Title:** {st.session_state.pdf_title}")
    
    chapter_count = len(st.session_state.get('chapter_info', []))
    if chapter_count > 0:
        st.sidebar.info(f"📚 **Chapters:** {chapter_count}")
        with st.sidebar.expander("View Chapters"):
            for ch in st.session_state.chapter_info[:10]:  # Show first 10
                st.text(f"Ch {ch['number']}: {ch['name']}")
    
    st.sidebar.info(f"📊 {st.session_state.num_chunks} chunks from {st.session_state.num_pages} pages")

# Display chat history
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat interface
if q := st.chat_input("Ask about your document..."):
    if not api_key:
        st.error("⚠️ Enter Groq API key")
        st.stop()
    
    if not st.session_state.retriever:
        st.error("⚠️ Upload and process PDF first (click 'Process Document' button)")
        st.stop()
    
    # Add user message
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model,
                    temperature=0,
                    max_tokens=1000
                )
                
                # Get document info
                doc_title = st.session_state.get('pdf_title', 'Unknown')
                total_pages = st.session_state.get('num_pages', 0)
                chapter_count = len(st.session_state.get('chapter_info', []))
                
                # Create custom prompt with metadata
                custom_prompt = f"""You are analyzing a document titled "{doc_title}" with {total_pages} pages and {chapter_count} chapters.

Use the context below to answer the question accurately. Pay special attention to:
- Counting items carefully (chapters, sections, stories, etc.)
- Extracting exact titles and names from headings
- Being precise with numbers and facts
- Looking for chapter information in the context

Context from the document:
{{context}}

Question: {{question}}

Instructions:
- Answer based ONLY on the context provided
- When counting chapters, look for "Chapter" headings or the metadata section
- When asked for title, look for "Document Title:" in the metadata or the first heading
- Be precise and quote exact text when relevant
- If the answer is not in the context, say "I cannot find that information in the provided context"

Answer:"""
                
                prompt = PromptTemplate(
                    template=custom_prompt,
                    input_variables=["context", "question"]
                )
                
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.retriever,
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                
                result = chain.invoke({"query": q})
                answer = result['result']
                
                st.write(answer)
                
                # Show sources
                if result.get('source_documents'):
                    with st.expander("📚 View Sources Used"):
                        for i, doc in enumerate(result['source_documents'], 1):
                            page_num = doc.metadata.get('page', 'N/A')
                            doc_type = doc.metadata.get('type', 'content')
                            st.markdown(f"**Source {i} ({doc_type.title()}, Page {page_num}):**")
                            st.text(doc.page_content[:300] + "...")
                            st.divider()
                
                st.session_state.msgs.append({"role": "assistant", "content": answer})
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Footer
if not st.session_state.retriever:
    st.info("👆 Upload a PDF and click 'Process Document' to start")
else:
    st.divider()
    st.caption("✅ Ready to answer questions about your document!")

st.divider()
st.caption("🚀 Groq API | 📄 Enhanced RAG | 💾 Smart Retrieval")
