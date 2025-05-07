from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from PIL import Image as PIL_Image
from pprint import pprint
import base64
from io import BytesIO
from rag import get_rag
from langchain_openai import ChatOpenAI
from IPython.display import display
from image_upload import ImageUploader
try:
  from google.colab import files
  IN_COLAB = True
except:
  IN_COLAB = False


llm = ChatOpenAI(model="gpt-4-turbo")
rag_chain = get_rag(llm)

class State(MessagesState):
    summary: str
    img_name: str
    img_base64: str
    img_interpertation: str
    # documents: list

# System message
sys_msg = SystemMessage(content="You are an AI nutrition assistant that estimates the total calories in a dish based on a text description or an image")


def assistant(state: State):
    ai_message = [AIMessage("Got a dish in mind? Upload a photo, and I’ll estimate how many calories it contains!", name="Bot")]
    return {"messages": [llm.invoke([sys_msg] + ai_message)]}


def upload_image(state: State):
    if IN_COLAB: # Upload image using Google Colab
        uploaded = files.upload()
        for name, file in uploaded.items():
            print(f"Uploaded file: {name}")
    else: # Upload image using Jupyter Notebook
        uploader = ImageUploader()
        name = uploader.get_file_name()
    return {"img_name": name}


def display_image(state: State):
    for m in state["messages"]:
        m.pretty_print()

    name = state["img_name"]
    print(f"Displaying image: {name}")
    img = PIL_Image.open(name)
    display(img)
    buffered = BytesIO() # Convert image to bytes buffer
    img.save(buffered, format="JPEG")  # or "PNG", depending on input
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8') # Encode to Base64
    return {"img_base64": img_base64}


# Function to identify food from an image
def identify_food(state: State):
    img_base64 = state["img_base64"]
    sys_task = ("Give only a list of the ingredients that make up the dish in the picture."
        " and for each ingredient, give its weight in grams for the dish in the picture.")
    response = llm.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": sys_task},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        )
    ])
    print(response.pretty_print())
    return {"messages": response, "img_interpertation": response.content}


def get_calories(state: State):
    food_items = state["img_interpertation"]
    response = rag_chain.invoke({"input": f"How much food energy is in {food_items}?"})
    print(response['answer'])
    return {"messages": response['answer']}