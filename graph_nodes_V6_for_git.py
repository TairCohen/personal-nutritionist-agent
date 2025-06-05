from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Optional
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

class State(MessagesState): # Keep inheriting from MessagesState
    summary: str
    # Make image fields optional and provide a default value (e.g., None)
    img_path: Optional[str] = None
    img_base64: Optional[str] = None
    img_interpertation: Optional[str] = None
    text_description:  Optional[str] = None
    # documents: list # If documents might be missing sometimes, make it Optional too
    # documents: Optional[list] = Field(default_factory=list) # Better to use Field with default_factory for lists

    # If you add other fields that are not always present, make them Optional as well.


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
    print("display_image is being called:")
    print(f"display_image state {state}")
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




# def identify_food(llm: ChatOpenAI):
#     def inner(state: State):
#         print("identify_food is being called:")
#         print(f"identify_food state {state}")
#         img_base64 = state["img_base64"]
#         sys_task = ("Give only a list of the ingredients that make up the dish in the picture."
#                     " and for each ingredient, give its weight in grams for the dish in the picture.")
#         response = llm.invoke([
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text": sys_task},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#                 ]
#             )
#         ])
#         print(response.pretty_print())
#         # return {
#         #     "messages": [response],
#         #     "img_interpertation": str(response.content)
#         # }
#         result = {
#             "messages": [response],
#             "img_interpertation": str(response.content)
#         }
#         print(f"identify_food returning: {result}")
#         return result
#     # return inner
#     return inner

# def identify_food(llm: ChatOpenAI):
#     def inner(state: State):
#         print("identify_food is being called:")
#         print(f"identify_food state {state}")
#         img_base64 = state["img_base64"]
#         sys_task = ("Give only a list of the ingredients that make up the dish in the picture."
#                     " and for each ingredient, give its weight in grams for the dish in the picture.")

#         print("➡️ About to call llm.invoke in identify_food")
#         try:
#             response = llm.invoke([
#                 HumanMessage(
#                     content=[
#                         {"type": "text", "text": sys_task},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#                     ]
#                 )
#             ])
#             print("⬅️ llm.invoke call successful in identify_food")
#             print("LLM Response (pretty_print):")
#             print(response.pretty_print()) # Keep this print
#             print("LLM Response content:", response.content) # Also print content directly

#             result = {
#                 "messages": [response],
#                 "img_interpertation": str(response.content)
#             }
#             print(f"identify_food returning: {result}")
#             return result
#         except Exception as e:
#             print(f"❌ An error occurred during identify_food: {e}")
#             import traceback
#             traceback.print_exc() # Print the full traceback
#             # Decide how to handle the error - perhaps return a state indicating error or re-raise
#             raise e # Re-raise the exception to stop execution and see the traceback clearly

#     return inner

# def identify_food(llm: ChatOpenAI):
#     def inner(state: State):
#         print("identify_food is being called:")
#         print(f"identify_food state {state.keys()}") # Print keys to avoid massive output
#         # print(f"identify_food state {state}") # Keep this if needed for detailed inspection, but can be large

#         img_base64 = state.get("img_base64") # Use .get for safety
#         if not img_base64:
#             print("❌ Error: img_base64 not found in state.")
#             return {"messages": [AIMessage(content="Error processing image: Image data not found.")]}

#         sys_task = ("Give only a list of the ingredients that make up the dish in the picture."
#                     " and for each ingredient, give its weight in grams for the dish in the picture.")

#         print("➡️ About to prepare and call llm.invoke in identify_food")
#         try:
#             # Construct the content list for the multi-modal input
#             content_list = [
#                 {"type": "text", "text": sys_task},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#             ]
#             print(f"Prepared LLM content list (first item content): {str(content_list[0]['text'])[:50]}...")
#             print(f"Prepared LLM content list (image type): {content_list[1]['type']}")
#             print(f"Base64 string length: {len(img_base64) if img_base64 else 0}")


#             response = llm.invoke([HumanMessage(content=content_list)])

#             print("⬅️ llm.invoke call successful in identify_food")
#             print("LLM Response (pretty_print):")
#             print(response.pretty_print()) # Keep this print
#             print("LLM Response content:", response.content) # Also print content directly

#             result = {
#                 "messages": [response],
#                 "img_interpertation": str(response.content)
#             }
#             print(f"identify_food returning: {result}")
#             return result

#         except Exception as e:
#             print(f"❌ An error occurred during identify_food llm.invoke or processing: {e}")
#             import traceback
#             traceback.print_exc() # Print the full traceback
#             # Return an error message in the state so the graph can potentially handle it gracefully
#             return {
#                 "messages": [AIMessage(content=f"Error processing image with LLM: {e}")]
#                 # Optionally add an error flag to the state
#                 # "error_state": True
#             }
#         # Note: With the try/except returning, the code after the try/except in 'inner' won't be reached on error.
#     return inner
# def identify_food(llm: ChatOpenAI):
#     def inner(state: State):
#         print("identify_food is being called:")
#         print(f"identify_food state {state.keys()}")

#         img_base64 = state.get("img_base64")
#         if not img_base64:
#             print("❌ Error: img_base64 not found in state.")
#             return {"messages": [AIMessage(content="Error processing image: Image data not found.")]}

#         sys_task = ("Give only a list of the ingredients that make up the dish in the picture."
#                     " and for each ingredient, give its weight in grams for the dish in the picture.")

#         response = None # Initialize response to None

#         print("➡️ About to prepare and call llm.invoke in identify_food")
#         try:
#             content_list = [
#                 {"type": "text", "text": sys_task},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#             ]
#             print(f"Prepared LLM content list (first item content): {str(content_list[0]['text'])[:50]}...")
#             print(f"Prepared LLM content list (image type): {content_list[1]['type']}")
#             print(f"Base64 string length: {len(img_base64) if img_base64 else 0}")

#             response = llm.invoke([HumanMessage(content=content_list)]) # Assign response here

#             print("⬅️ llm.invoke call successful in identify_food")
#             print("LLM Response (pretty_print):")
#             print(response.pretty_print())
#             print("LLM Response content:", response.content)

#         except Exception as e:
#             print(f"❌ An error occurred during identify_food llm.invoke or processing: {e}")
#             import traceback
#             traceback.print_exc()
#             # Decide how to handle the error - perhaps return a state indicating error or re-raise
#             # If an error occurs, we'll return an error message here
#             return {
#                 "messages": [AIMessage(content=f"Error processing image with LLM: {e}")]
#             }

#         # THIS PART NEEDS TO BE OUTSIDE THE TRY/EXCEPT TO RUN ON SUCCESS
#         if response is not None: # Check if response was successfully obtained
#             result = {
#                 "messages": [response], # Add the successful LLM response to messages
#                 "img_interpertation": str(response.content) # Add the content to img_interpertation
#             }
#             print(f"identify_food returning: {result}")
#             return result
#         else:
#             # This case should ideally not be reached if the try/except handles errors,
#             # but it's good practice to have a final return.
#              print("⚠️ identify_food finished without successful response or error.")
#              return {"messages": [AIMessage(content="Unexpected issue during image processing.")]}


#     return inner

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
    # ✅ FIX: Ensure img_interpertation is a simple string
    return {
        "messages": [response],
        "img_interpertation": str(response.content)
    }


from langchain_core.messages import AIMessage


def get_calories(state:State): # ,rag_chain=None
    """Estimates the calorie count for food items described in the state.""" # <-- Add this docstring
    print(f"🔥 get_calories node called. Current state keys: {state.keys()}")
    print(f"🔥 State text_description: {state.get('text_description')}")
    print(f"🔥 State img_interpertation: {state.get('img_interpertation')}")

    food_items = state.get("img_interpertation")
    if not food_items:
        food_items = state.get("text_description")
    print("🧪 food_items:", food_items)

    if not food_items:
        print("⚠️ get_calories: No food description found.")
        # Returning a simple string or specific value to indicate failure
        # The ToolNode will wrap this in a ToolMessage.
        # return "Error: Could not find a dish description to analyze for calorie estimation."
        return {
    "messages": [AIMessage(content="Error: Could not find a dish description to analyze for calorie estimation.")]
}


    print(f"🔍 get_calories: Estimating calories for: {food_items}")


    try:
        # Call the RAG chain
        response = rag_chain.invoke({"input": f"How much food energy is in {food_items}?"})
        print("🧠 Full RAG response:", response)

        calorie_answer = response.get('answer', 'Could not estimate calories.') # Use .get() in case 'answer' is missing
        print(f"✅ RAG Chain response: {calorie_answer}")
        # Return the result string directly
        # return calorie_answer
        return {
            "messages": [AIMessage(content=f"Estimated calories: {calorie_answer}")],
            "calorie_estimate": calorie_answer
        }

    except Exception as e:
        print(f"❌ Error during RAG chain invocation: {e}")
        # Return an error message string
        return {
                "messages": [AIMessage(content=f"Error getting calorie estimation: {str(e)}")]
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
