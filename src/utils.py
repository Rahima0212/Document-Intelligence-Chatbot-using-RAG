import re
from typing import List

def clean_text(text: str) -> str:
    # Basic cleaning
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk a long text into overlapping chunks.

    Args:
        text: input string
        chunk_size: max chars per chunk
        overlap: overlap between chunks

    Returns:
        list of chunks
    """
    text = text.replace('\n', ' \n ')
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = i
        curr = []
        curr_len = 0
        while j < len(words) and curr_len + len(words[j]) + 1 <= chunk_size:
            curr.append(words[j])
            curr_len += len(words[j]) + 1
            j += 1
        chunks.append(' '.join(curr))
        if j >= len(words):
            break
        # step forward with overlap
        step_back = max(1, (overlap // 5))
        i = j - step_back
    return [c.strip() for c in chunks if c.strip()]
