import boto3
import json
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
        Embeds a list of texts using the Amazon Bedrock API.
        Loops through each text and calls the API individually.
    """
    all_embeddings = []
    for text in texts:
        # The body must be a JSON-formatted string
        body = json.dumps({
            "inputText": text
        })

        # Call the Bedrock API
        response = bedrock_client.invoke_model(
            body=body,
            modelId=EMBED_MODEL,
            accept="application/json",
            contentType="application/json"
        )

        # Parse the response
        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")
        all_embeddings.append(embedding)

    return all_embeddings


