import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import streamlit as st

# Initialize SentenceTransformer model (you can replace this with any compatible model)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use any sentence transformer model here
dimension = embedding_model.get_sentence_embedding_dimension()  # Get the dimension dynamically from the model
index = faiss.IndexFlatL2(dimension)  # FAISS index for L2 distance
stored_data = []  # To store content (title, subtitle, and paragraphs)

# Function to fetch and parse content from a URL
def fetch_url_content(url):
    """
    Fetches content from the provided URL (title, subtitle, and paragraphs).
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.find('h1').get_text() if soup.find('h1') else 'No title found'
    subtitles = [h2.get_text() for h2 in soup.find_all(['h2', 'h3', 'h4'])]
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    
    # Combine title, subtitles, and paragraphs into a list of texts
    content = [{'text': title}] + [{'text': subtitle} for subtitle in subtitles] + [{'text': paragraph} for paragraph in paragraphs]
    
    return content

# Function to add the fetched content to the FAISS index
def add_to_faiss(content):
    """
    Adds the provided content (titles, subtitles, paragraphs) to the FAISS index.
    """
    global index, stored_data
    
    # Prepare the data (texts) to be embedded
    texts = [item['text'] for item in content]
    
    # Generate embeddings for the content using SentenceTransformer
    embeddings = embedding_model.encode(texts)
    
    # Convert embeddings to np.float32 and ensure correct shape
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Ensure the embeddings shape is correct (n_samples, n_features)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)  # For a single query, expand dimensions
    
    # Add to FAISS index
    index.add(embeddings)
    
    # Store the content for future querying
    stored_data.extend(content)

# Function to query FAISS index based on a user query
def query_faiss(user_query):
    """
    Queries the FAISS index for the most relevant content based on the user's input.
    """
    # Generate embedding for the user query
    query_embedding = embedding_model.encode([user_query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # Search FAISS index for the nearest vectors
    distances, indices = index.search(query_embedding, k=3)  # Retrieve the top 3 closest results
    
    # Prepare the results to display
    results = [{"text": stored_data[i]["text"], "distance": d} for i, d in zip(indices[0], distances[0]) if i < len(stored_data)]
    
    return results

# Streamlit interface for user interaction
def main():
    st.title('Web-based Chatbot with FAISS')
    
    # Input URL and fetch data
    url = st.text_input('Enter a URL to fetch content (e.g., https://en.wikipedia.org/wiki/Machine_learning):')
    if url:
        content = fetch_url_content(url)
        add_to_faiss(content)
        st.success('Content fetched and stored in FAISS successfully.')
    
    # Input query to chat and retrieve information
    user_query = st.text_input('Ask a question:')
    if user_query:
        results = query_faiss(user_query)
        if results:
            st.write(f"Results for your query '{user_query}':")
            for result in results:
                st.write(f"- {result['text']}\n")
        else:
            st.write("Sorry, no relevant information found.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
