# RAG Chatbot with Groq API & Streamlit

## Overview

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that enables users to upload PDF documents and ask natural language questions about the content. The system retrieves relevant information from documents and generates accurate answers using Groq's Llama API.

## How It Works

1. **PDF Processing**: Documents are parsed and text is extracted from all pages
2. **Text Chunking**: Content is split into 800-character chunks while preserving sentence structure
3. **Metadata Extraction**: System automatically identifies document titles and chapters
4. **Intelligent Retrieval**: An improved retriever uses multi-factor scoring (phrase matching, word overlap, keyword density) to find relevant chunks
5. **Answer Generation**: Retrieved chunks are sent to Groq Llama API to generate contextual answers
6. **Source Attribution**: Each answer includes citations showing which document sections were used

## Key Features

- Automatic PDF text extraction and processing
- Advanced multi-factor retrieval algorithm
- Fast response generation via Groq API
- Persistent chat history within session
- Source citations for transparency
- In-memory storage for privacy and speed
- Clean, intuitive Streamlit interface


## Technology Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyPDF
- **RAG Framework**: LangChain
- **LLM Provider**: Groq (Llama models)
- **Language**: Python 3.8+

## Core Components

**ImprovedRetriever**: Custom retrieval system that scores chunks based on:
- Exact phrase matches
- Word overlap with query
- Keyword density
- Document metadata boost

**Document Analyzer**: Extracts:
- Document title from first page
- Chapter structure and names
- Page references for citation

**RAG Chain**: Orchestrates the complete Q&A pipeline from user query to final response with sources.

## Storage Architecture

The application uses **in-memory storage** with no external database:
- All document chunks stored in Python RAM during session
- Data is temporary and cleared when application restarts
- Provides privacy, speed, and instant access
- Limited by available system memory

## Performance

- Small PDF (10 pages): 2-5 seconds
- Medium PDF (50 pages): 10-15 seconds
- Large PDF (100+ pages): 20-30 seconds
- Query response: 1-3 seconds average


## Sample Output

The `output/` directory contains:
- Interface screenshots showing the application layout
- Example Q&A interactions demonstrating chatbot capabilities
- Sample PDF document for testing

## Dependencies

See `requirements.txt` for all required packages:
- streamlit, langchain, langchain-groq, langchain-classic
- langchain-core, langchain-community
- pypdf, python-dotenv

## Technical Notes

- No external database required; all processing is local
- Groq API is used only for LLM inference
- Document content never leaves your machine
- Ideal for research, documentation analysis, and content understanding

## Future Enhancements

- Vector-based retrieval using FAISS or ChromaDB for semantic search capabilities
- Support for additional file formats (DOCX, TXT, images)
- Persistent database integration (PostgreSQL, MongoDB)
- Multi-document conversation support
- Chat history export functionality


---




