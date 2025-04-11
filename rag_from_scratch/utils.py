import re
import torch
from typing import Any, List
from collections import deque
from PyPDF2 import PdfReader

# Set up Chromadb
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models import Collection

# Insert chunks into vector store
import os
import uuid
from typing import Tuple




def text_extract(pdf_path: str) -> str:
    """
    Extracts text from all pages of a given PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF, concatenated with newline separators.
    """

    # An empty list to store extracted text from PDF pages
    pdf_pages = []

    # Open the PDF file in binary read mode
    with open(pdf_path, 'rb') as file:

        # Create a PdfReader object to read the PDF
        pdf_reader = PdfReader(file)

        # Iterate through all pages in the PDF
        for page in pdf_reader.pages:

            # Extract text from the current page
            text = page.extract_text()

            # Append the extracted text to the list
            pdf_pages.append(text)

    # Join all extracted text using newline separator
    pdf_text = "\n".join(pdf_pages)

    # Return the extracted text as a single string
    return pdf_text


def text_chunk(text: str, max_length: int = 1000) -> List[str]:
    """
    Splits a given text into chunks while ensuring that sentences remain intact.

    The function maintains sentence boundaries by splitting based on punctuation
    (. ! ?) and attempts to fit as many sentences as possible within `max_length`
    per chunk.

    Args:
        text (str): The input text to be chunked.
        max_length (int, optional): Maximum length of each chunk. Default is 1000.

    Returns:
        List[str]: A list of text chunks, each containing full sentences.
    """

    # Split text into sentences while ensuring punctuation (. ! ?) stays at the end
    sentences = deque(re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')))

    # An empty list to store the final chunks
    chunks = []

    # Temporary string to hold the current chunk
    chunk_text = ""

    while sentences:
        # Access sentence from the deque and strip any extra spaces
        sentence = sentences.popleft().strip()

        # Check if the sentence is non-empty before processing
        if sentence:
            # If adding this sentence exceeds max_length and chunk_text is not empty, store the current chunk
            if len(chunk_text) + len(sentence) > max_length and chunk_text:

                # Save the current chunk
                chunks.append(chunk_text)

                # Start a new chunk with the current sentence
                chunk_text = sentence
            else:
                # Append the sentence to the current chunk with a space
                chunk_text += " " + sentence

    # Add the last chunk if there's any remaining text
    if chunk_text:
        chunks.append(chunk_text)

    return chunks


def create_vector_store(db_path: str, model_name: str) -> Collection:
    """
    Creates a persistent ChromaDB vector store with OpenAI embeddings.

    Args:
        db_path (str): Path where the ChromaDB database will be stored.

    Returns:
        Collection: A ChromaDB collection object for storing and retrieving embedded vectors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a ChromaDB PersistentClient with the specified database path
    client = chromadb.PersistentClient(path=db_path)
    
    # Create an embedding function using OpenAI's text embedding model
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
        trust_remote_code=True
    )

    # Create a new collection in the ChromaDB database with the embedding function
    try:
        db = client.create_collection(
            name="pdf_chunks",  # Name of the collection where embeddings will be stored
            embedding_function=embeddings
        )
    except Exception as err:
        db = client.get_collection(
            name="pdf_chunks",
            embedding_function=embeddings
        )

    # Return the created ChromaDB collection
    return db


def insert_chunks_vectordb(chunks: List[str], db: Collection, file_path: str) -> None:
    """
    Inserts text chunks into a ChromaDB vector store with metadata.

    Args:
        chunks (List[str]): List of text chunks to be stored.
        db (Collection): The ChromaDB collection where the chunks will be inserted.
        file_path (str): Path of the source file for metadata.

    Returns:
        None
    """

    # Extract the file name from the given file path
    file_name = os.path.basename(file_path)

    # Generate unique IDs for each chunk
    id_list = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Create metadata for each chunk, storing the chunk index and source file name
    metadata_list = [{"chunk": i, "source": file_name} for i in range(len(chunks))]

    # Define batch size for inserting chunks to optimize performance
    batch_size = 40

    # Insert chunks into the database in batches
    for i in range(0, len(chunks), batch_size):
        end_id = min(i + batch_size, len(chunks))  # Ensure we don't exceed list length

        # Add the batch of chunks to the vector store
        db.add(
            documents=chunks[i:end_id],
            metadatas=metadata_list[i:end_id],
            ids=id_list[i:end_id]
        )

    print(f"{len(chunks)} chunks added to the vector store")


def retrieve_chunks(db: Collection, query: str, n_results: int = 2) -> List[Any]:
    """
    Retrieves relevant chunks from the  vector store for the given query.

    Args:
        db (Collection): The vector store object
        query (str): The search query text.
        n_results (int, optional): The number of relevant chunks to retrieve. Defaults to 2.

    Returns:
        List[Any]: A list of relevant chunks retrieved from the vector store.
    """

    # Perform a query on the database to get the most relevant chunks
    relevant_chunks = db.query(query_texts=[query], n_results=n_results)

    # Return the retrieved relevant chunks
    return relevant_chunks


def build_context(relevant_chunks) -> str:
    """
    Builds a single context string by combining texts from relevant chunks.

    Args:
        relevant_chunks: relevant chunks retrieved from the vector store.

    Returns:
        str: A single string containing all document chunks combined with newline separators.
    """

    # combine the text from relevant chunks with newline separator
    context = "\n".join(relevant_chunks['documents'][0])

    # Return the combined context string
    return context



def get_context(pdf_path: str, query: str, db_path: str, model_name: str = "Alibaba-NLP/gte-multilingual-base") -> Tuple[str, str]:
    """
    Retrieves the relevant chunks from the vector store and then builds context from them.

    Args:
        pdf_path (str): The file path to the PDF document.
        query (str): The query string to search within the vector store.
        db_path (str): The file path to the persistent vector store database.

    Returns:
        Tuple[str, str]: A tuple containing the context related to the query and the original query string.
    """

    # Check if the vector store already exists
    if os.path.exists(db_path):
        print("Loading existing vector store...")

        # Initialize the persistent client for the existing database
        client = chromadb.PersistentClient(path=db_path)

        # Get the collection of PDF chunks from the existing vector store
        db = client.get_collection(name="pdf_chunks")
    else:
        print("Creating new vector store...")

        # Extract text from the provided PDF
        pdf_text = text_extract(pdf_path)

        # Chunk the extracted text
        chunks = text_chunk(pdf_text)

        # Create a new vector store
        db = create_vector_store(db_path=db_path, model_name=model_name)

        # Insert the text chunks into the vector store
        insert_chunks_vectordb(chunks, db, pdf_path)

    # Retrieve the relevant chunks based on the query
    relevant_chunks = retrieve_chunks(db, query)

    # Build the context from the relevant chunks
    context = build_context(relevant_chunks)

    # Return the context and the original query
    return context, query


def create_augmented_prompt(context: str, query: str) -> str:
    """
    Generates a rag prompt based on the given context and query.

    Args:
        context (str): The context the LLM should use to answer the question.
        query (str): The user query that needs to be answered based on the context.

    Returns:
        str: The generated rag prompt.
    """

    # Format the prompt with the provided context and query
    rag_prompt = f""" You are an AI model trained for question answering. You should answer the
    given question based on the given context only.
    Question : {query}
    \n
    Context : {context}
    \n
    If the answer is not present in the given context, respond as: The answer to this question is not available
    in the provided content.
    """

    # Return the formatted prompt
    return rag_prompt


from transformers import AutoModelForCausalLM, AutoTokenizer
def ask_llm(prompt):
    """
    Sends a prompt to the Alibaba's Qwen LLM and returns the answer.

    Args:
        prompt (str): The augmented prompt.

    Returns:
        str: The LLM generated answer.
    """
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def rag_pipeline(pdf_path: str, query: str, db_path: str) -> str:
    """
    Runs a Retrieval-Augmented Generation (RAG) pipeline to retrieve context from a vector store,
    generate the rag prompt, and then get the answer from the model.

    Args:
        pdf_path (str): The file path to the PDF document from which context is extracted.
        query (str): The query for which a response is needed, based on the context.
        db_path (str): The file path to the persistent vector store database used for context retrieval.

    Returns:
        str: The model's response based on the context and the provided query.
    """

    # get the context
    context, query = get_context(pdf_path, query, db_path)

    # Generate the rag prompt based on the context and query
    augment_prompt = create_augmented_prompt(context, query)

    # Get the response from the model using the rag prompt
    response = ask_llm(augment_prompt)

    # Return the model's response
    return response
