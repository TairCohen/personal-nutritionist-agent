from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from PIL import Image as PIL_Image
from pprint import pprint
import base64
from io import BytesIO
from rag import get_rag
from langchain_openai import ChatOpenAI
from IPython.display import display
from langchain.tools import tool
from image_upload import ImageUploader
import time
try:
  from google.colab import files
  IN_COLAB = True
except:
  IN_COLAB = False


class State(MessagesState):
    summary: str
    img_path: str
    img_base64: str
    img_interpertation: str
    # documents: list

# System message
sys_msg = SystemMessage(content="You are an AI nutrition assistant that estimates the total calories in a dish based on a text description or an image")


def assistant(state: State) -> State:
    # ai_message = [AIMessage("Got a dish in mind? Upload a photo, and I’ll estimate how many calories it contains!", name="Bot")]
    ai_message = state["messages"]
    return {"messages": [llm.invoke([sys_msg] + ai_message)]}


def upload_image(state: State) -> State:
    """since we dont have now the option of upload image
    the function will return an image name (path).

    Args:
        state: State
    """
    print("upload imagegg")
    if IN_COLAB: # Upload image using Google Colab
        uploaded = files.upload()
        for name, file in uploaded.items():
            print(f"Uploaded file: {name}")
    else: # Upload image using Jupyter Notebook
        print("upload image")
        name = "download.jpg"
        # uploader = ImageUploader()
        # while not uploader.get_dd():
        #     # print("⏳ Waiting for image upload...")
        #     time.sleep(0.5)
        # print("Image uploaded successfully.")
        # name = uploader.get_file_name()
    return {"img_path": name}


@tool
def get_image_path() -> str:
    """The function will return an image path.
    """
    print("ddd")
    path = "download.jpg"
    # state["img_path"] = path
    # return state
    return path


def get_last_tool_output(state):
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            return message.content  # This is "download.jpg"
    return None  # fallback if no tool message found
   
    
def display_image(state: State):
    # print("ff", state)
    name = get_last_tool_output(state)
    print(name)
    # name = state["img_path"]
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

# def tools_condition(state: State) -> str:
#     if state["messages"][-1].tool_calls:
#         tool_name = state["messages"][-1].tool_calls[0].name
#         return tool_name  # e.g., "upload_image"
#     return "__else__"

def tools_condition(state: State) -> str:
    # Assuming you are using OpenAI tool calls (e.g., function calling)
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        print(f"tool details {last_message.tool_calls}")
        tool_name = last_message.tool_calls[0]['name'] 
        # Must match a key in path_map!
        if tool_name == "upload_image":
            return "upload_image"
        elif tool_name == "get_image_path":
            return "get_image_path"
    return "__else__"

llm = ChatOpenAI(model="gpt-4-turbo")
rag_chain = get_rag(llm)
# tools = [upload_image]
tools = [get_image_path] 
llm = llm.bind_tools(tools, parallel_tool_calls=False)
