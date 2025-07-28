from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

#client = OpenAI(
 # api_key=os.getenv("NVIDIA_API_KEY"),
  #base_url="https://integrate.api.nvidia.com/v1"
#)
client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)





def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def embed_text(text: str, input_type: str = "passage") -> list[float]:
    """Embed text with proper input type handling"""
    # Truncate text if too long (keep first 8000 chars to be safe)
    if len(text) > 8000:
        text = text[:8000]
        logger.warning(f"Truncated text to 8000 characters for embedding")
    
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return response.data[0].embedding


def embed_file(file_path: str) -> list[float]:
    """Embed file content using passage input type"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
        
        # Add file path context to the content for better matching
        content_with_context = f"File: {os.path.basename(file_path)}\n\n{content}"
        
        return embed_text(content_with_context, input_type="passage")
    except Exception as e:
        logger.error(f"Failed to embed file {file_path}: {e}")
        raise


def chunk_large_file(content: str, max_chunk_size: int = 4000) -> list[str]:
    """Split large files into smaller chunks for better embedding"""
    if len(content) <= max_chunk_size:
        return [content]
    
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def embed_file_chunked(file_path: str) -> list[list[float]]:
    """Embed file in chunks and return list of embeddings"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
        
        chunks = chunk_large_file(content)
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            # Add context about which chunk this is
            chunk_with_context = f"File: {os.path.basename(file_path)} (part {i+1}/{len(chunks)})\n\n{chunk}"
            embedding = embed_text(chunk_with_context, input_type="passage")
            embeddings.append(embedding)
        
        return embeddings
    except Exception as e:
        logger.error(f"Failed to embed file chunks {file_path}: {e}")
        raise


def retrieve_embeddings(query: str, file_path: str) -> tuple[list[float], list[float]]:
    query_embedding = embed_query(query)
    file_embedding = embed_file(file_path)
    return query_embedding, file_embedding

