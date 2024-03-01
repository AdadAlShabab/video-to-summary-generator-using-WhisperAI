import whisper

model = whisper.load_model("base")
result = model.transcribe("files.mp4")

transcribed_text = result['text']

# Save the transcribed text to a file
with open("text.txt", "w") as file:
    file.write(transcribed_text)

print("Transcribed text saved to text.txt")

from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.summarize import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap

llm = OpenAI(model_name="text-davinci-003", temperature=0)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

with open('text.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)