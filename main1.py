import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import requests
from neo4j import GraphDatabase
import streamlit as st
import arxiv
import ollama
# import faiss
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import fitz
import io

app = FastAPI()

class Query(BaseModel):
    topic: str
    year: int = datetime.datetime.now().year

class QARequest(BaseModel):
    topic: str
    year: int = datetime.datetime.now().year
    question: str

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define helper function to interact with Ollama
def run_ollama_command(prompt, model="llama2"):
    try:
        print("running model")
        result = subprocess.run(
            ["ollama", "generate", model, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

# Initialize Neo4j database agent
class DatabaseAgent:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_paper(self, title, year, abstract, url):
        with self.driver.session() as session:
            session.run(
                "MERGE (p:Paper {title: $title, year: $year, abstract: $abstract, url: $url})",
                title=title, year=year, abstract=abstract, url=url
            )

    def retrieve_papers_from_db(self, t, y):
        try:
            with self.driver.session() as session:
                query = "MATCH (p:Paper) WHERE p.year = $year AND toLower(p.title) CONTAINS toLower($topic) RETURN p"
                print(f"Running query: {query} with parameters: year={y}, topic={t}")
                result = session.run(query, year=y, topic=t)
                papers = [record["p"] for record in result]
                return papers
        except Exception as e:
            print(f"Error querying the database: {e}")
            return []

db_agent = DatabaseAgent("neo4j+s://8818c386.databases.neo4j.io", "neo4j", "xoBWbJZDELN7PLiSzNw7RDHP_mPRK3oYGuTmdAqeygo")

@app.post("/search_papers")
async def search_papers(query: Query):
    try:
        # Use the arxiv API to search for papers based on topic and year
        search = arxiv.Search(
            query=query.topic,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )

        # Filter papers by year and format them for output
        papers = []
        for result in search.results():
            publication_year = result.published.year
            if publication_year == query.year:
                paper_data = {
                    "title": result.title,
                    "year": publication_year,
                    "abstract": result.summary,
                    "url": result.entry_id
                }
                papers.append(paper_data)
                
                # Store paper in Neo4j database
                try:
                    db_agent.store_paper(
                        title=paper_data["title"],
                        year=paper_data["year"],
                        abstract=paper_data["abstract"],
                        url=paper_data["url"]
                    )
                    print(f"Stored paper: {paper_data['title']}")
                except Exception as db_error:
                    print(f"Database error when storing paper: {db_error}")
                    raise HTTPException(status_code=500, detail=f"Database error: {db_error}")
            
        
        if not papers:
            raise HTTPException(status_code=404, detail="No papers found for the given topic and year")

        return {"papers": papers}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_papers")
async def summarize_papers(query: Query):
    papers = db_agent.retrieve_papers_from_db(query.topic, query.year)
    summaries = [run_ollama_command(paper["abstract"], model="llama2") for paper in papers]
    return {"summaries": summaries}

@app.post("/generate_future_works")
async def generate_future_works(query: Query):
    papers = db_agent.retrieve_papers_from_db(query.topic, query.year)
    future_works = [
        run_ollama_command(f"Suggest future research directions based on {paper['abstract']}", model="llama2")
        for paper in papers
    ]
    return {"future_works": future_works}

def extract_pdf_text_from_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch the PDF.")
        
        if "application/pdf" not in response.headers.get('Content-Type', ''):
            raise HTTPException(status_code=400, detail="The URL does not point to a valid PDF.")

        # Load the PDF from the in-memory byte stream
        pdf_document = fitz.open(io.BytesIO(response.content))

        # Extract text from each page
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")  # Extract text in plain format

        if text:
            return text
        else:
            raise HTTPException(status_code=500, detail="No text found in the PDF.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def qa(query: QARequest):
    try:
        # Retrieve relevant papers from the database
        papers = db_agent.retrieve_papers_from_db(query.topic, query.year)
        
        if not papers:
            raise HTTPException(status_code=404, detail="No relevant papers found for the topic and year")
        
        # Assuming we only take the first paper here for simplicity
        selected_paper = papers[0]
        
        # Retrieve full content of the paper from the URL
        paper_url = selected_paper["url"]
        print(f"Paper URL: {paper_url}")

        # Extract text from the PDF URL
        full_text = extract_pdf_text_from_url(paper_url)

        # Split the full text into smaller chunks for similarity matching
        chunks = [full_text[i:i + 512] for i in range(0, len(full_text), 512)]
        
        # Embed each chunk and the question
        chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
        question_embedding = embedding_model.encode(query.question, convert_to_tensor=True)
        
        # Use cosine similarity to find the most relevant chunk
        similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
        best_chunk_idx = similarities.argmax().item()
        best_chunk = chunks[best_chunk_idx]
        
        # Create a prompt using the best-matching chunk as context
        prompt = f"Context:\n{best_chunk}\n\nQuestion: {query.question}\nAnswer based on the context above."

        # Generate answer using Ollama
        answer = run_ollama_command(prompt)

        # Return answer and the source of the paper
        return {
            "answer": answer,  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))