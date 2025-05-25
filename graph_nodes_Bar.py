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

from langchain.schema import AIMessage

def assistant(state: State) -> State:
    ai_message = state["messages"]
    # Get the LLM response (which might include tool calls)
    response = llm.invoke([sys_msg] + ai_message)
    
    # Check if the LLM tool call includes 'get_image_path'
    tool_calls = []
    if response and hasattr(response, "tool_calls"):
        tool_calls = response.tool_calls
    
    # If there is a tool_call to get_image_path, add user prompt text
    if any(tc["name"] == "get_image_path" for tc in tool_calls):
        # Return an AIMessage with prompt + tool_calls
        return {
            "messages": [
                AIMessage(
                    content="Please upload your image so I can analyze it.",
                    tool_calls=tool_calls,
                )
            ]
        }
    else:
        # Normal flow, just return the LLM response as is
        return {"messages": [response]}


# def extract_tool_call_id(messages, default="unknown_tool_call_id"):
#     for message in reversed(messages):
#         if hasattr(message, "tool_calls") and message.tool_calls:
#             return message.tool_calls[0]["id"]
#     print("❌ No tool_call found in state['messages']")
#     return default


import base64

def upload_image(state: State) -> State:
    """
    Upload an image and return both the file path and base64-encoded image content.
    """
    print("upload imagegg")
    if IN_COLAB:
        uploaded = files.upload()
        for name, file in uploaded.items():
            print(f"✅ Uploaded file: {name}")
            # Convert to base64 string
            img_base64 = base64.b64encode(file).decode("utf-8")
            return {"img_path": name, "img_base64": img_base64}
    else:
        # Handle notebook scenario — load image from file
        name = "your_default_image.jpg"  # You can change this or prompt the user
        with open(name, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        return {"img_path": name, "img_base64": img_base64}


@tool
def get_image_path() -> dict:
    """Prompt user to upload an image and return the image file path."""
    print("✅ Tool ran — uploading file")
    # This function is invoked by LangGraph's tool handler
    # You can't access tool_call_id here unless you track it manually
    if IN_COLAB:
        uploaded = files.upload()
        for name in uploaded.keys():
            print(f"✅ Uploaded file: {name}")
            return {"img_path": name}  # tool_call_id won't be here!
    else:
        from tkinter import filedialog
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        return {"img_path": path}


from langchain_core.messages import ToolMessage

def get_image_path_node(state):
    print("🚨 FULL STATE", state)
    # Extract the tool_call_id to attach to ToolMessage
    tool_call_id = extract_tool_call_id(state["messages"])
    
    # Get image path from tool output
    img_path = state.get("img_path")
    
    print(f"🔁 tool_call_id = {tool_call_id}")
    print(f"📸 img_path = {img_path}")
    
    if not img_path:
        return {
            "messages": [ToolMessage(
                content="No image was uploaded.",
                tool_call_id=tool_call_id
            )]
        }
    
    return {
        "messages": [ToolMessage(
            content=f"Image received: {img_path}",
            tool_call_id=tool_call_id
        )],
        "img_path": img_path
        # "tool_call_id": tool_call_id
    }


def get_last_tool_output(State):
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            print("DEBUG - last tool output content:", message.content)
            return message.content
    return None
 

from langchain_core.messages import AIMessage
import json
from PIL import Image as PIL_Image
from IPython.display import display
from io import BytesIO
import base64

def display_image(state:State):
    messages = state["messages"]
    last_tool = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)

    if not last_tool:
        return {
            "messages": [AIMessage(content="No tool output found. Please upload an image first.")]
        }

    try:
        tool_data = json.loads(last_tool.content)
        img_path = tool_data.get("img_path")
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Failed to parse tool message: {e}")]
        }

    if not img_path:
        return {
            "messages": [AIMessage(content="Tool did not return an image path.")]
        }

    try:
        img = PIL_Image.open(img_path)
        display(img)

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print(img_base64)
        return {
            "messages": [AIMessage(content=f"Image loaded: {img_path}")],
            "img_path": img_path,            # ✅ Add this to state
            "img_base64": img_base64         # ✅ Optional, for image-to-LLM
        }

    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Failed to open image: {e}")]
        }


# # Function to identify food from an image
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
    # return {"messages": response, "img_interpertation": response.content}
    # ✅ FIX: Ensure img_interpertation is a simple string
    return {
        "messages": [response],
        "img_interpertation": str(response.content)
    }


from langchain_core.messages import AIMessage


def get_calories(state:State):
    food_items = state.get("img_interpertation") or state.get("text_description")
    if not food_items:
        return {
            "messages": [AIMessage(content="I couldn't find a dish description to analyze. Please describe or upload an image.")]
        }

    response = rag_chain.invoke({"input": f"How much food energy is in {food_items}?"})
    print(response['answer'])

    return {
        "messages": [AIMessage(content=response['answer'])]
    }

def tools_condition(state) -> str:
    # Assuming you are using OpenAI tool calls (e.g., function calling)
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"tool details: {last_message.tool_calls}")
        tool_name = last_message.tool_calls[0]['name'] 

        # Match this to your graph routing
        if tool_name == "upload_image":
            return "upload_image"
        elif tool_name == "get_image_path":
            return "get_image_path"

    return "__else__"
    
    
llm = ChatOpenAI(model="gpt-4-turbo")
rag_chain = get_rag(llm)
tools = [get_image_path] 
llm = llm.bind_tools(tools, parallel_tool_calls=False)
