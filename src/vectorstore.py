import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "housing_clean.csv")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")

def create_vectorstore():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Process one-hot encoded neighborhoods into a single column
    neighborhood_cols = [c for c in df.columns if c.startswith('Neighborhood_')]
    if neighborhood_cols:
        df['Neighborhood'] = df[neighborhood_cols].idxmax(axis=1).str.replace('Neighborhood_', '')
    else:
        df['Neighborhood'] = "Unknown"
        
    bldg_type_cols = [c for c in df.columns if c.startswith('Bldg Type_')]
    if bldg_type_cols:
        df['BldgType'] = df[bldg_type_cols].idxmax(axis=1).str.replace('Bldg Type_', '')
    else:
        df['BldgType'] = "House"

    # Create a rich text description for semantic search
    def create_description(row):
        return (
            f"A {row.get('BldgType', 'House')} in {row.get('Neighborhood', 'Unknown')} neighborhood. "
            f"Price is ${row.get('SalePrice', 'Unknown')}. "
            f"It has {row.get('Bedroom AbvGr', 'Unknown')} bedrooms, "
            f"{row.get('TotalBath', 'Unknown')} bathrooms, "
            f"and a total area of {row.get('TotalSF', 'Unknown')} sq ft. "
            f"House age is {row.get('HouseAge', 'Unknown')} years."
        )
        
    df['text_description'] = df.apply(create_description, axis=1)
    
    # We only need a subset of the dataset and some metadata
    # Limiting to 500 rows for faster processing during prototyping
    df_subset = df.head(500).copy()
    
    # Use DataFrameLoader to load the text_description column as the page_content
    # and keep the other columns as metadata
    print("Preparing documents...")
    loader = DataFrameLoader(df_subset, page_content_column="text_description")
    documents = loader.load()
    
    print("Generating embeddings and building FAISS index...")
    # Using a fast, free local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save locally
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vectorstore saved successfully at {VECTORSTORE_DIR}")

if __name__ == "__main__":
    create_vectorstore()
