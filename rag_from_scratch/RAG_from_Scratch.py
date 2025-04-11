from utils import rag_pipeline

if __name__ == "__main__":
    # Set the chroma DB path
    db_path="./chroma_alibaba_gte.db"

    # PDF path
    pdf_path = "attention_is_all_you_need.pdf"

    # RAG query
    query = "What is self-attention?"

    # Run the RAG pipeline
    answer = rag_pipeline(pdf_path, query, db_path)

    print(f"Query:\n'{query}'\n")
    print(f"Generated answer:\n'{answer}'")
    

''' Output should be something like below.
Query:
'What is self-attention?'

Generated answer:
'Self-attention is an attention mechanism that relates different positions within a single sequence to compute a representation of the sequence. It allows every position in the decoder to attend over all positions in the input sequence, mimicking the typical encoder-decoder attention mechanisms in sequence-to-sequence models like those found in [31, 2, 8]. Self-attention is particularly useful in various tasks such as reading comprehension, abstractive summarization, textual entailment, and learning task-independent sentence representations. It reduces the complexity of calculating relationships between distant positions compared to other methods but sacrifices some level of effective resolution due to averaging attention-weighted positions.'
'''