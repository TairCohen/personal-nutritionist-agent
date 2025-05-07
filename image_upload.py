import io
from ipywidgets import FileUpload, Button, VBox
from IPython.display import display
from PIL import Image

class ImageUploader:
    def __init__(self, accept='image/*'):
        self.uploader = FileUpload(accept=accept, multiple=False)
        self.button = Button(description="Process Image")
        self.button.on_click(self._on_click)
        self.file_name = None
        self.image = None
        display(VBox([self.uploader, self.button]))

    def _on_click(self, b):
        if self.uploader.value:
            uploaded_file = self.uploader.value[0]  # tuple format
            self.file_name = uploaded_file['name']
            image_bytes = uploaded_file['content']
            self.image = Image.open(io.BytesIO(image_bytes))
            self.image.save(self.file_name)
            print(f"Image uploaded and saved as: {self.file_name}")
        else:
            print("No file uploaded yet.")

    def get_file_name(self):
        return self.file_name

    def get_image(self):
        return self.image