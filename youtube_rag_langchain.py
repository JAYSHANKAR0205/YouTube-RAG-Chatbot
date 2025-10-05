from langchain_ollama import OllamaLLM, OllamaEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import PromptTemplate

video_id="mce7qxMpnOY" # Only video Id, not url
try:
  ytt_api=YouTubeTranscriptApi()
  transcript_list=ytt_api.fetch(video_id,languages=["en"])
  raw_transcript=transcript_list.to_raw_data()
  transcript=" ".join([sentence['text'] for sentence in raw_transcript])
except TranscriptsDisabled:
  print("Transcript not found")

spilitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100)
chunks=spilitter.create_documents([transcript])
print(chunks[0].page_content)

embeddings=OllamaEmbeddings(model="nomic-embed-text")
vector_store=Pinecone.from_documents(chunks,embeddings)

retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

model=OllamaLLM(model="llama3:8b",temperature=0.7)
prompt=PromptTemplate(template="You are useful assistant. Answer the question only from provided Transcript. If the context is insufficient then say I don't know.       {context} and {question}",
                      input_variables=["context","question"])


question=input("enter your query about this video : ") # query given by the user it have to be changed. 
retrieve_doc=retriever.invoke(question)

def format_context(retrieve_doc):
  context=" /n/n ".join([doc.page_content for doc in retrieve_doc])
  return context


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

parallel_chain=RunnableParallel({"context":retriever|RunnableLambda(format_context),"question":RunnablePassthrough()})
parallel_chain.invoke(question)

parser=StrOutputParser()
prompt_chain=prompt|model|parser
final_chain=parallel_chain|prompt_chain

answer=final_chain.invoke(question)
print(answer)