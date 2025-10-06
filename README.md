Automatic Transcript Fetching – Retrieves the YouTube video transcript directly using the video ID.

Smart Text Chunking – Splits the transcript into small overlapping chunks for better context understanding.

Vector Search with Pinecone – Stores and retrieves the most relevant transcript parts using semantic similarity.

Intelligent Q&A – Uses Llama 3 (via Ollama) to answer user queries based only on the video transcript.

Fallback Handling – Gracefully handles cases where transcripts are disabled.
