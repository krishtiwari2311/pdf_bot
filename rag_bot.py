import os
import base64
import uuid
import time
from typing import List, Dict, Any
from datetime import datetime
from google import genai

# PDF and image processing
import fitz  # PyMuPDF
from PIL import Image as PIL_Image

# Data handling
import pandas as pd
import numpy as np
import regex as re

# Vector store - using LanceDB instead of FAISS
import lancedb
import pyarrow as pa

# For the UI
import gradio as gr

# LangChain components

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Environment setup
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = ""

client = genai.Client(api_key=GEMINI_API_KEY)

# Models configuration
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0,
    "top_p": 0.95,
}

# Initialize LanceDB vectorstore
def initialize_vector_store():
    """Initialize LanceDB vectorstore"""
    def embedding_function(texts: List[str]) -> List[List[float]]:
        # Handle single text or list of texts
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings for each text
        embeddings = []
        for text in texts:
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-exp-03-07",
                    content=text
                )
                # Extract embedding values from response
                embedding = result.embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                # Return a zero vector as fallback
                embeddings.append([0] * 768)  # Default embedding dimension
        
        return embeddings

    # Initialize LanceDB
    db = lancedb.connect('vector_store.lance')
    
    try:
        # Try to get existing table
        table = db.open_table("documents")
        print("Loaded existing vector store")
    except ValueError:
        # Create new table if it doesn't exist
        # Define schema using PyArrow
        schema = pa.schema([
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 768)),  # 768-dimensional embedding
            pa.field("source", pa.string()),
            pa.field("chunk", pa.int32()),
            pa.field("image_path", pa.string())
        ])
        
        # Create table with schema and data
        table = db.create_table(
            "documents",
            schema=schema,
            mode="create"
        )
        
        # Add placeholder document
        vector = embedding_function(["placeholder"])[0]
        table.add([{
            "text": "placeholder",
            "vector": vector,
            "source": "placeholder",
            "chunk": 0,
            "image_path": ""
        }])
        print("Created new vector store")
    
    return table

def create_image_folder(image_path):
    """Create directory for extracted images if it doesn't exist"""
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    return image_path

def extract_pdf_content(pdf_path: str, image_folder: str):
    """Extract content from PDF including text, tables, and images"""
    doc = fitz.open(pdf_path)
    file_basename = os.path.basename(pdf_path)
    
    # Lists to store extracted data
    pages_data = []
    
    # Zoom factors for better resolution
    zoom_x = 2.0
    zoom_y = 2.0
    mat = fitz.Matrix(zoom_x, zoom_y)
    
    # Process each page
    for page_num, page in enumerate(doc):
        try:
            # Extract page as image
            pix = page.get_pixmap(matrix=mat)
            img_path = f"{image_folder}/{file_basename}_page_{page_num}.png"
            pix.save(img_path)
            
            # Get text directly from PDF
            pdf_text = page.get_text()
            
            # Get Gemini model for content extraction
            model = "gemini-2.0-flash"
            
            # Load the saved image for Gemini processing
            with open(img_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            # Extract text using Gemini
            try:
                response_text = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=["Extract all text content in the image, preserving paragraph structure, headings, and formatting",
                             {"inlineData": {"data": base64.b64encode(image_bytes).decode("utf-8"), "mimeType": "image/png"}}]
                )
                text_content = response_text.text
            except Exception as e:
                print(f"Error extracting text: {str(e)}")
                text_content = "Text extraction failed"
            
            # Extract tables using Gemini
            try:
                response_table = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=["Detect any tables in this image. Extract content maintaining the structure. If no tables are found, state 'No tables detected.'",
                             {"inlineData": {"data": base64.b64encode(image_bytes).decode("utf-8"), "mimeType": "image/png"}}]
                )
                table_content = response_table.text
            except Exception as e:
                print(f"Error extracting tables: {str(e)}")
                table_content = "Table extraction failed"
            
            # Extract image descriptions
            try:
                response_image = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=["Detect any charts, diagrams or informative images (excluding decorative elements). Provide a detailed description of each. If none are found, state 'No informative images detected.'",
                             {"inlineData": {"data": base64.b64encode(image_bytes).decode("utf-8"), "mimeType": "image/png"}}]
                )
                image_content = response_image.text
            except Exception as e:
                print(f"Error extracting image descriptions: {str(e)}")
                image_content = "Image extraction failed"
            
            # Combine all extracted content
            combined_content = f"""
            PAGE {page_num + 1} OF DOCUMENT {file_basename}
            
            PDF TEXT:
            {pdf_text}
            
            GEMINI EXTRACTED TEXT:
            {text_content}
            
            TABLE CONTENT:
            {table_content}
            
            IMAGE DESCRIPTIONS:
            {image_content}
            """
            
            # Store the data
            pages_data.append({
                "page_num": page_num,
                "image_path": img_path,
                "content": combined_content,
                "source": f"{file_basename} - Page {page_num + 1}"
            })
            
            print(f"Processed page {page_num + 1} of {file_basename}")
            
            # Avoid rate limits
            time.sleep(2)
            
        except ResourceExhausted:
            print(f"API quota exceeded, waiting longer for page {page_num + 1}")
            time.sleep(10)  # Wait longer if hitting quota limits
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {str(e)}")
    
    return pages_data

def process_and_store_pdf(pdf_path, image_folder, vector_store):
    """Process PDF and store chunks in vector store"""
    # Extract content
    print(f"Processing PDF: {pdf_path}")
    pages_data = extract_pdf_content(pdf_path, image_folder)
    
    # Create text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=800,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create documents and store in vector store
    all_docs = []
    
    for page_data in pages_data:
        content = page_data["content"]
        source = page_data["source"]
        image_path = page_data["image_path"]
        
        # Split text into chunks
        chunks = text_splitter.split_text(content)
        
        # Create documents
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk": i,
                    "image_path": image_path
                }
            )
            all_docs.append(doc)
    
    def embedding_function(texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-exp-03-07",
                    content=text
                )
                embedding = result.embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                embeddings.append([0.0] * 768)
        return embeddings
    
    # Convert documents to LanceDB format and add to vector store
    records = []
    for doc in all_docs:
        vector = embedding_function([doc.page_content])[0]
        records.append({
            "text": doc.page_content,
            "vector": [float(x) for x in vector],  # Ensure all values are float
            "source": doc.metadata["source"],
            "chunk": doc.metadata["chunk"],
            "image_path": doc.metadata["image_path"]
        })
    
    if records:
        vector_store.add(records)
    
    return len(records)

def format_sources(source_documents):
    """Format source documents for display"""
    return "\n\n".join([
        f"SOURCE {i+1}: {doc.metadata['source']}, Chunk {doc.metadata['chunk']}"
        for i, doc in enumerate(source_documents)
    ])

def get_reference_image(source_documents):
    """Get the reference image path from the most relevant document"""
    if source_documents and len(source_documents) > 0:
        return source_documents[0].metadata.get("image_path")
    return None

def generate_answer(query, vector_store):
    """Query the vector store and generate an answer using Gemini"""
    def embedding_function(texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-exp-03-07",
                    content=text
                )
                embedding = result.embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                embeddings.append([0.0] * 768)
        return embeddings
    
    # Generate query embedding
    query_vector = embedding_function([query])[0]
    
    # Search for similar documents
    results = vector_store.search(query_vector).limit(5).to_list()
    
    if not results:
        return {
            "answer": "I don't have any relevant information to answer this question.",
            "reference_image": None,
            "sources": "No sources found"
        }
    
    # Format context from results
    context = "\n\n".join([result["text"] for result in results])
    
    # Create a prompt for the LLM and generate response
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"""
            Answer the question based on the following context from document pages.
            If the context doesn't contain relevant information to fully answer the question, say "I don't have enough information to answer this question completely."
            Provide a comprehensive answer that addresses all aspects of the question.
            
            CONTEXT:
            {context}
            
            QUESTION: {query}
            
            ANSWER:
        """]
    )
    
    # Update source formatting
    sources = "\n\n".join([
        f"SOURCE {i+1}: {result['source']}, Chunk {result['chunk']}"
        for i, result in enumerate(results)
    ])
    
    # Get reference image from most relevant result
    reference_image = results[0]["image_path"] if results else None
    
    return {
        "answer": response.text,
        "reference_image": reference_image,
        "sources": sources
    }

def test_answer_quality(answer):
    """Test if the answer indicates insufficient information"""
    # Classification prompt
    classification_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"""
            Classify the text as one of the following categories:
            - Information Present
            - Information Not Present
            
            Examples:
            Text: The provided context does not contain information.
            Category: Information Not Present
            
            Text: I cannot answer this question from the provided context.
            Category: Information Not Present
            
            Text: {answer}
            Category:
        """]
    )
    
    # Return True if information is present, False otherwise
    return "Not Present" not in classification_response.text

# Gradio UI functions
def process_pdf(pdf_files):
    """Process uploaded PDFs and store in vector store"""
    if not pdf_files:
        return "No PDF files uploaded."
    
    # Initialize vector store
    vector_store = initialize_vector_store()
    
    # Create image folder
    image_folder = create_image_folder("extracted_images")
    
    # Process each PDF
    total_chunks = 0
    processed_files = []
    
    for pdf_file in pdf_files:
        pdf_path = pdf_file.name
        chunks = process_and_store_pdf(pdf_path, image_folder, vector_store)
        total_chunks += chunks
        processed_files.append(os.path.basename(pdf_path))
    
    return f"Processed {len(processed_files)} PDFs, extracted {total_chunks} chunks. Ready for querying!"

def query_docs(query):
    """Process a user query and return the answer"""
    if not query:
        return "Please enter a question.", None, ""
    
    # Initialize vector store
    vector_store = initialize_vector_store()
    
    # Generate answer
    result = generate_answer(query, vector_store)
    
    answer = result["answer"]
    image_path = result["reference_image"]
    sources = result["sources"]
    
    # Return the answer and image
    return answer, image_path if image_path and os.path.exists(image_path) else None, sources

# Gradio interface
def create_ui():
    with gr.Blocks(title="PDF RAG Assistant") as app:
        gr.Markdown("# PDF RAG Assistant with Gemini & LanceDB")
        gr.Markdown("This application uses Gemini's multimodal capabilities to extract content from PDFs and create a RAG system.")
        
        with gr.Tab("Upload PDFs"):
            pdf_input = gr.File(
                file_count="multiple",
                label="Upload PDF Files",
                file_types=[".pdf"]
            )
            process_button = gr.Button("Process PDFs")
            process_output = gr.Textbox(label="Processing Status")
            process_button.click(process_pdf, inputs=[pdf_input], outputs=process_output)
        
        with gr.Tab("Ask Questions"):
            query_input = gr.Textbox(label="Ask a question about your documents")
            query_button = gr.Button("Get Answer")
            
            with gr.Row():
                answer_output = gr.Textbox(label="Answer", lines=10)
                image_output = gr.Image(label="Reference Image")
            
            sources_output = gr.Textbox(label="Sources", lines=5)
            
            query_button.click(query_docs, inputs=[query_input], outputs=[answer_output, image_output, sources_output])
    
    return app

# Main function
def main():

    # Create the Gradio UI
    app = create_ui()
    
    # Launch the app
    app.launch()

if __name__ == "__main__":
    main()
