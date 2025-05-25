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


# def assistant(state: State) -> State:
#     # ai_message = [AIMessage("Got a dish in mind? Upload a photo, and I’ll estimate how many calories it contains!", name="Bot")]
#     ai_message = state["messages"]
#     return {"messages": [llm.invoke([sys_msg] + ai_message)]}


# def assistant(state: State) -> State:
#     ai_message = state["messages"]
    
#     # Bind tools to enable tool calling
#     bound_llm = llm.bind_tools([get_image_path, identify_food, get_calories])
    
#     # Invoke with tools and system prompt
#     output = bound_llm.invoke([sys_msg] + ai_message)

#     return {"messages": [output]}

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
#     tool_calls = [m for m in messages if getattr(m, "type", getattr(m, "get", lambda x: {})("type")) == "tool_call"]
#     if not tool_calls:
#         print("❌ No tool_call found in state['messages']")
#         return default
#     last_tool_call = tool_calls[-1]
#     return getattr(last_tool_call, "id", getattr(last_tool_call, "get", lambda x: default)("id"))

def extract_tool_call_id(messages, default="unknown_tool_call_id"):
    for message in reversed(messages):
        if hasattr(message, "tool_calls") and message.tool_calls:
            return message.tool_calls[0]["id"]
    print("❌ No tool_call found in state['messages']")
    return default


# def upload_image(state: State) -> State:
#     """since we dont have now the option of upload image
#     the function will return an image name (path).

#     Args:
#         state: State
#     """
#     print("upload imagegg")
#     if IN_COLAB: # Upload image using Google Colab
#         uploaded = files.upload()
#         for name, file in uploaded.items():
#             print(f"Uploaded file: {name}")
#     else: # Upload image using Jupyter Notebook
#         print("upload image")
#         name = "hamburger.jpg"
#         # uploader = ImageUploader()
#         # while not uploader.get_dd():
#         #     # print("⏳ Waiting for image upload...")
#         #     time.sleep(0.5)
#         # print("Image uploaded successfully.")
#         # name = uploader.get_file_name()
#     return {"img_path": name}
# import tkinter as tk
# from tkinter import filedialog

# def upload_image(state: State) -> State:
#     """Prompt user to upload image (Colab or Jupyter) and return its path."""
#     print("📤 upload_image triggered")
#     name = ""

#     if IN_COLAB:
#         uploaded = files.upload()
#         for filename in uploaded.keys():
#             print(f"✅ Uploaded in Colab: {filename}")
#             name = filename  # Path relative to notebook
#     else:
#         print("🖼️ Please select an image...")
#         root = tk.Tk()
#         root.withdraw()
#         name = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
#         if not name:
#             print("❌ No image selected.")
#             name = "NO_IMAGE"

#     return {"img_path": name}


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




# @tool
# def get_image_path() -> str:
#     """The function will return an image path.
#     """
#     print("ddd")
#     path = "hamburger.jpg"
#     # state["img_path"] = path
#     # return state
#     return path

# @tool
# def get_image_path() -> str:
#     """Ask the user to upload an image and return the path to it."""
#     print("🖼️ get_image_path tool called")
    
#     if IN_COLAB:
#         uploaded = files.upload()
#         for name in uploaded.keys():
#             print(f"✅ Uploaded file: {name}")
#             return name
#     else:
#         # Jupyter: simulate or ask for actual file
#         from tkinter import filedialog
#         import tkinter as tk

#         print("📁 Please choose an image file to upload...")
#         root = tk.Tk()
#         root.withdraw()  # Hide the root window
#         file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        
#         if file_path:
#             print(f"✅ Uploaded file: {file_path}")
#             return file_path
#         else:
#             print("❌ No file selected.")
#             return ""


# @tool
# def get_image_path() -> dict:
#     """Prompt user to upload an image and return the image file path."""
#     if IN_COLAB:
#         uploaded = files.upload()
#         for name in uploaded.keys():
#             print(f"✅ Uploaded file: {name}")
#             # DEBUG
#             print(f"DEBUG get_image_path returning: {{'img_path': {name}}}")
#             return {"img_path": name}  # Just filename, relative path in Colab
#     else:
#         from tkinter import filedialog
#         import tkinter as tk

#         print("🖼️ Please choose an image file to upload...")
#         root = tk.Tk()
#         root.withdraw()
#         path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
#         if path:
#             print(f"✅ Selected file: {path}")
#             print(f"DEBUG get_image_path returning: {path}")
#             return {"img_path": path}  # Full path as dict, same key
#         else:
#             print("❌ No file selected.")
#             return {"img_path": ""}  # Or handle error better
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


# from langchain_core.messages import ToolMessage, ToolCall


# def get_image_path_node(State):
#     tool_call_id = extract_tool_call_id(state["messages"])
#     # Simulate image path logic
#     image_path = state.get("image_path", "/path/to/image.jpg")  # Replace with real logic
#     return {
#         "messages": ToolMessage(
#             content=f"Image received: {image_path}",
#             tool_call_id=tool_call_id
#         ),
#         "image_path": image_path
#     }

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


# def get_last_tool_output(state):
#     for message in reversed(state["messages"]):
#         if isinstance(message, ToolMessage):
#             return message.content  # This is "download.jpg"
#     return None  # fallback if no tool message found

def get_last_tool_output(State):
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            print("DEBUG - last tool output content:", message.content)
            return message.content
    return None
 
    
# def display_image(state: State):
#     print("DEBUG display_image received state keys:", list(state.keys()))
#     img_path = state.get("img_path")
#     if not img_path:
#         return {"messages": [AIMessage(content="No image was uploaded. Please try again.")]}
#     print(f"Displaying image: {img_path}")
#     name = get_last_tool_output(state)
#     if not name or not name.strip():
#         return {"messages": [AIMessage(content="No image was uploaded. Please try again.")]}
#     # print(name)
#     # name = state["img_path"]
#     print(f"Displaying image: {name}")
#     img = PIL_Image.open(name)
#     display(img)
#     buffered = BytesIO() # Convert image to bytes buffer
#     img.save(buffered, format="JPEG")  # or "PNG", depending on input
#     img_bytes = buffered.getvalue()
#     img_base64 = base64.b64encode(img_bytes).decode('utf-8') # Encode to Base64
#     return {"img_base64": img_base64}


# def display_image(state: State):
#     print("DEBUG display_image received state keys:", list(state.keys()))
#     img_path = state.get("img_path")
#     if not img_path or not img_path.strip():
#         return {"messages": [AIMessage(content="No image was uploaded. Please try again.")]}
    
#     print(f"Displaying image: {img_path}")
#     try:
#         img = PIL_Image.open(img_path)
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         return {"messages": [AIMessage(content="Failed to open the uploaded image.")]}
    
#     display(img)
    
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")  # or PNG depending on input type
#     img_bytes = buffered.getvalue()
#     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
#     return {"img_base64": img_base64}

# def display_image(state: dict):
#     print("DEBUG display_image received state keys:", list(state.keys()))
#     img_path = state.get("img_path")
#     if not img_path or not img_path.strip():
#         state["messages"] = [AIMessage(content="No image was uploaded. Please try again.")]
#         return state
    
#     print(f"Displaying image: {img_path}")
#     try:
#         img = PIL_Image.open(img_path)
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         state["messages"] = [AIMessage(content="Failed to open the uploaded image.")]
#         return state
    
#     display(img)
    
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_bytes = buffered.getvalue()
#     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
#     # Add the base64 to state but keep existing keys intact
#     state["img_base64"] = img_base64
#     return state

# def display_image(state: State):
#     print("DEBUG display_image received state keys:", list(state.keys()))
#     img_path = state.get("img_path")
#     tool_call_id = state.get("tool_call_id")

#     if not img_path or not img_path.strip():
#         return {
#             "messages": [
#                 ToolMessage(content="No image was uploaded. Please try again.", tool_call_id=tool_call_id)
#             ]
#         }

#     try:
#         img = PIL_Image.open(img_path)
#         display(img)

#         buffered = BytesIO()
#         img.save(buffered, format="JPEG")
#         img_bytes = buffered.getvalue()
#         img_base64 = base64.b64encode(img_bytes).decode('utf-8')

#         return {
#             "messages": [
#                 ToolMessage(
#                     content=f"Image displayed successfully.",
#                     tool_call_id=tool_call_id
#                 )
#             ],
#             "img_base64": img_base64,
#             "img_path": img_path,
#             "tool_call_id": tool_call_id
#         }

#     except Exception as e:
#         return {
#             "messages": [
#                 ToolMessage(content=f"Failed to open image: {e}", tool_call_id=tool_call_id)
#             ]
#         }
# def display_image(state):
#     tool_call_id = state.get("tool_call_id") or extract_tool_call_id(state["messages"])

#     # tool_call_id = extract_tool_call_id(state["messages"])
#     image_path = state.get("image_path")
#     if not image_path:
#         return {
#             "messages": ToolMessage(
#                 content="No image was uploaded. Please try again.",
#                 tool_call_id=tool_call_id
#             )
#         }
#     return {
#         "messages": ToolMessage(
#             content=f"Displaying image from path: {image_path}",
#             tool_call_id=tool_call_id
#         )
#     }

# def display_image(state):
#     img_path = state.get("img_path")
#     if not img_path:
#         return {
#             "messages": [AIMessage(content="No image was uploaded. Please try again.")]
#         }

#     try:
#         img = PIL_Image.open(img_path)
#         display(img)

#         buffered = BytesIO()
#         img.save(buffered, format="JPEG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#         return {
#             "messages": [AIMessage(content="Image displayed successfully.")],
#             "img_base64": img_base64  # optional: might be used downstream
#         }

#     except Exception as e:
#         return {
#             "messages": [AIMessage(content=f"Failed to open image: {e}")]
#         }
import json

# def display_image(state):
#     messages = state["messages"]
#     last_tool_message = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)

#     if not last_tool_message:
#         return {
#             "messages": [AIMessage(content="No image was uploaded. Please try again.")]
#         }

#     try:
#         tool_data = json.loads(last_tool_message.content)
#         img_path = tool_data.get("img_path")
#     except Exception as e:
#         return {
#             "messages": [AIMessage(content=f"Failed to parse tool output: {e}")]
#         }

#     if not img_path:
#         return {
#             "messages": [AIMessage(content="Tool did not return an image path.")]
#         }

#     # Optional: display or process the image
#     return {
#         "messages": [AIMessage(content=f"Image loaded: {img_path}")],
#         "img_path": img_path  # Inject into state for later use
#     }

# import json
# from langchain_core.messages import ToolMessage, AIMessage

# def display_image(state):
#     # ✅ Find last ToolMessage
#     messages = state["messages"]
#     last_tool = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)

#     if not last_tool:
#         return {
#             "messages": [AIMessage(content="No tool output found. Please upload an image first.")]
#         }

#     try:
#         # ✅ Extract img_path from the ToolMessage content
#         tool_data = json.loads(last_tool.content)
#         img_path = tool_data.get("img_path")
#     except Exception as e:
#         return {
#             "messages": [AIMessage(content=f"Failed to parse tool message: {e}")]
#         }

#     if not img_path:
#         return {
#             "messages": [AIMessage(content="Tool did not return an image path.")]
#         }

#     try:
#         from PIL import Image as PIL_Image
#         from IPython.display import display
#         from io import BytesIO
#         import base64

#         img = PIL_Image.open(img_path)
#         display(img)

#         buffered = BytesIO()
#         img.save(buffered, format="JPEG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#         return {
#             "messages": [AIMessage(content=f"Image loaded: {img_path}")],
#             "img_path": img_path,
#             "img_base64": img_base64
#         }

#     except Exception as e:
#         return {
#             "messages": [AIMessage(content=f"Failed to open image: {e}")]
#         }
from langchain_core.messages import AIMessage
import json
from PIL import Image as PIL_Image
from IPython.display import display
from io import BytesIO
import base64

def display_image(state):
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

        return {
            "messages": [AIMessage(content=f"Image loaded: {img_path}")],
            "img_path": img_path,            # ✅ Add this to state
            "img_base64": img_base64         # ✅ Optional, for image-to-LLM
        }

    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Failed to open image: {e}")]
        }


# # # Function to identify food from an image
# def identify_food(state: State):
#     img_base64 = state["img_base64"]
#     sys_task = ("Give only a list of the ingredients that make up the dish in the picture."
#         " and for each ingredient, give its weight in grams for the dish in the picture.")
#     response = llm.invoke([
#         HumanMessage(
#             content=[
#                 {"type": "text", "text": sys_task},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#             ]
#         )
#     ])
#     print(response.pretty_print())
#     # return {"messages": response, "img_interpertation": response.content}
#     # ✅ FIX: Ensure img_interpertation is a simple string
#     return {
#         "messages": [response],
#         "img_interpertation": str(response.content)
#     }
# from langchain.schema import ToolMessage

# def identify_food(state):
#     # tool_call_id = extract_tool_call_id(state["messages"])
#     tool_call_id = state.get("tool_call_id") or extract_tool_call_id(state["messages"])

#     print("DEBUG identify_food received state keys:", list(state.keys()))
#     print("DEBUG tool_call_id:", state.get("tool_call_id"))
#     print("DEBUG img_path:", state.get("img_path"))
#     # tool_call_id = state.get("tool_call_id")
#     img_path = state.get("img_path")

#     # No image? Ask user to upload
#     if not img_path:
#         if not tool_call_id:
#             # No tool call context, just ask to upload normally
#             return {"messages": [AIMessage(content="Please upload your image so I can analyze it.")]}
#         else:
#             # In tool call context but no image? Tool message prompt
#             return {"messages": [ToolMessage(content="No image found. Please upload your meal image.", tool_call_id=tool_call_id)]}

#     # If no tool_call_id, just confirm image received and wait for tool call context
#     if not tool_call_id:
#         return {"messages": [AIMessage(content="Image received. Processing will start shortly...")]}

#     # Try to read image bytes for recognition
#     try:
#         with open(img_path, "rb") as f:
#             image_bytes = f.read()
#     except Exception as e:
#         return {"messages": [ToolMessage(content=f"Failed to load image: {e}", tool_call_id=tool_call_id)]}

#     # Call your existing food recognition function
#     food_info = your_existing_food_recognition_function(image_bytes)

#     # Format response
#     response_text = f"Dish recognized: {food_info['dish_name']}\nIngredients and weights:\n"
#     for ingredient in food_info["ingredients"]:
#         response_text += f"- {ingredient['name']}: {ingredient['grams']} g\n"
#     response_text += f"Estimated total calories: {food_info.get('total_calories', 'unknown')} kcal"

#     # return {"messages": [ToolMessage(content=response_text, tool_call_id=tool_call_id)]}
#     return {
#     "messages": [AIMessage(content="Image analysis complete...")]
# }

# def identify_food(state):
#     tool_call_id = state.get("tool_call_id") or extract_tool_call_id(state["messages"])
#     img_path = state.get("img_path")

#     if not img_path:
#         return {
#             "messages": [AIMessage(content="No image found. Please upload your meal image.")]
#         }

#     try:
#         with open(img_path, "rb") as f:
#             image_bytes = f.read()
#     except Exception as e:
#         return {
#             "messages": [AIMessage(content=f"Failed to load image: {e}")]
#         }

#     # Simulate your food recognition logic
#     food_info = your_existing_food_recognition_function(image_bytes)

#     response_text = f"Dish recognized: {food_info['dish_name']}\nIngredients and weights:\n"
#     for ingredient in food_info["ingredients"]:
#         response_text += f"- {ingredient['name']}: {ingredient['grams']} g\n"
#     response_text += f"Estimated total calories: {food_info.get('total_calories', 'unknown')} kcal"

#     return {
#         "messages": [AIMessage(content=response_text)],
#         "img_interpertation": response_text  # for use in get_calories
#     }


from langchain_core.messages import AIMessage

def identify_food(state):
    img_path = state.get("img_path")

    if not img_path:
        return {
            "messages": [AIMessage(content="No image found. Please upload your meal image.")]
        }

    try:
        with open(img_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Failed to read the image: {e}")]
        }

    # Replace this with your actual image analysis logic
    food_info = your_existing_food_recognition_function(image_bytes)

    response = f"Dish: {food_info['dish_name']}\n"
    for i in food_info["ingredients"]:
        response += f"- {i['name']}: {i['grams']}g\n"
    response += f"Estimated total calories: {food_info.get('total_calories', '?')} kcal"

    return {
        "messages": [AIMessage(content=response)],
        "img_interpertation": response  # feed into get_calories
    }


# from langchain.schema import ToolMessage

# def identify_food(state: MessagesState):
#     img_base64 = state["img_base64"]
#     sys_task = ("Give only a list of the ingredients that make up the dish in the picture."
#         " and for each ingredient, give its weight in grams for the dish in the picture.")
#     response = llm.invoke([
#         HumanMessage(
#             content=[
#                 {"type": "text", "text": sys_task},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#             ]
#         )
#     ])
#     ingredients_list = response.content  # Text listing ingredients + weights

#     print(response.pretty_print())

#     # Return the output as a ToolMessage with tool_call_id if applicable
#     # or simply as AIMessage if no tool_call context.
#     return {
#         "messages": [
#             AIMessage(content=ingredients_list)
#         ],
#         # optionally, pass ingredients explicitly if your framework supports it
#         "ingredients": ingredients_list
#     }



# def get_calories(state: State):
#     food_items = state["img_interpertation"]
#     response = rag_chain.invoke({"input": f"How much food energy is in {food_items}?"})
#     print(response['answer'])
#     return {"messages": response['answer']}
    
    
# --- Calorie Estimation from Image or Text ---
# def get_calories(state: State):
#     tool_call_id = extract_tool_call_id(state["messages"])
#     print("img_interpertation:", state.get("img_interpertation"))
#     food_items = state.get("img_interpertation") or state.get("text_description")
#     if not food_items:
#         return {"messages": [AIMessage(content="I couldn't find a dish description to analyze. Please describe or upload an image.")]}

#     response = rag_chain.invoke({"input": f"How much food energy is in {food_items}?"})
#     print(response['answer'])
#     # return {"messages": response['answer']}
#     return {"messages": [AIMessage(content=response['answer'])]}

# def get_calories(state: MessagesState):
#     food_items = state.get("img_interpertation") or state.get("text_description")
#     if not food_items:
#         return {"messages": [AIMessage(content="I couldn't find a dish description to analyze. Please describe or upload an image.")]}

#     response = rag_chain.invoke({"input": f"How much food energy is in {food_items}?"})
#     print(response['answer'])
    
#     # Wrap the answer string in AIMessage and return as a list
#     return {"messages": [AIMessage(content=response['answer'])]}

# def get_calories(state: MessagesState):
#     # Read last message content for ingredients text
#     messages = state.get("messages", [])
#     if not messages:
#         return {"messages": [AIMessage(content="No input messages found.")]}
    
#     last_msg_content = messages[-1].content if hasattr(messages[-1], "content") else None
#     if not last_msg_content:
#         return {"messages": [AIMessage(content="I couldn't find a dish description to analyze. Please describe or upload an image.")]}

#     response = rag_chain.invoke({"input": f"How much food energy is in {last_msg_content}?"})
#     print(response['answer'])
    
#     return {"messages": [AIMessage(content=response['answer'])]}

from langchain_core.messages import AIMessage

# def get_calories(state):
#     tool_call_id = state.get("tool_call_id") or extract_tool_call_id(state["messages"])
    
#     food_items = state.get("img_interpertation") or state.get("text_description")
    
#     if not food_items:
#         return {
#             "messages": [AIMessage(content="I couldn't find a dish description to analyze. Please describe or upload an image.")]
#         }

#     response = rag_chain.invoke({"input": f"How much food energy is in {food_items}?"})
#     print(response['answer'])

#     return {
#         "messages": [AIMessage(content=response['answer'])],
#         "tool_call_id": tool_call_id
#     }
def get_calories(state):
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


# def tools_condition(state: State) -> str:
#     if state["messages"][-1].tool_calls:
#         tool_name = state["messages"][-1].tool_calls[0].name
#         return tool_name  # e.g., "upload_image"
#     return "__else__"

# 

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
# tools = [upload_image]
tools = [get_image_path] 
llm = llm.bind_tools(tools, parallel_tool_calls=False)
