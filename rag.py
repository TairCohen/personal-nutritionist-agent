
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import pandas as pd

try:
  from google.colab import drive
  IN_COLAB = True
except:
  IN_COLAB = False


def get_retriever():
    if IN_COLAB:
        drive.mount('/content/drive')
        filtered_file_path = "/content/drive/MyDrive/סיכומים מתואר שני/NLP/calories_dataset_consistent_rephrasing.csv"
    else:
       filtered_file_path = "calories_dataset_consistent_rephrasing.csv"
    # data = pd.read_csv(filtered_file_path)
    docs = CSVLoader(file_path=filtered_file_path).load_and_split()
    # embeddings = OpenAIEmbeddings()
    index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
    vector_store = FAISS(
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=docs)
    retriever = vector_store.as_retriever()
    return retriever


def get_rag(llm):
  retriever = get_retriever()
  # Set up system prompt
  system_prompt = (
      "You are an AI nutrition assistant that estimates the total calories in a dish based on a text description or an image.\n\n"
      "### *Estimation Methodology:*\n"
      "1. *Check for an exact match in the database.*\n"
      "   - If an exact match exists, return its calorie count per 100g.\n"
      "2. *If no exact match exists, break the dish into ingredients and estimate calories.*\n"
      "   - Identify the *most relevant base food* (e.g., a plain omelet for 'cheese omelet').\n"
      "   - Check for *similar variations* (e.g., 'Egg or omelet, fried without oil' as the base).\n"
      "   - *Only include ingredients explicitly mentioned in the description.*\n"
      "   - Add ingredients like cheese based on the closest match in the database. *Do not assume any extra ingredients (e.g., mushrooms) unless explicitly mentioned.*\n"
      "   - Adjust calorie estimates proportionally to the expected ingredient ratio.\n"
      "3. *Do NOT assume extra ingredients unless explicitly mentioned.*\n"
      "4. *Do NOT use the calorie value of a mixed dish (e.g., 'omelet with mushrooms and cheese') as a direct replacement for a different variant (e.g., 'cheese omelet').*\n"
      "5. *Clearly explain the steps taken, including any assumptions about portions.*\n"
      "6. *For each ingredient:*\n"
      "   - Provide the closest match from the database (e.g., 'Egg or omelet, fried without oil') and its calorie count per 100g.\n"
      "   - If the exact calorie count for an ingredient is missing, explain that and provide an estimated serving size (e.g., 150g for eggs, 30g for cheese).\n"
      "   - Use the standard serving size to calculate the calories from each ingredient based on the proportion of the total dish.\n"
      "7. *Provide the final total calories for the dish.*\n\n"
      "Use the retrieved database context below to find accurate calorie values:\n"
      "{context}\n\n"
      "If the exact ingredient is not found, use the closest alternative and explain why.\n"
      "If specific calorie counts are missing, make assumptions based on standard serving sizes and ingredient ratios. Always provide the final total calorie estimate.")
  
  prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{input}"),

  ])
  # Create the question-answer chain
  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)
  return rag_chain

