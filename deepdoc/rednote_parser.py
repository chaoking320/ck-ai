import sys
import threading
# import pdfplumber
from openai import OpenAI


LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()

dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",

    # prompt_layout_only_en: layout detection
    "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_layout_only_en: parse ocr text except the Page-header and Page-footer
    "prompt_ocr": """Extract the text content from this image.""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "prompt_grounding_ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",

    # "prompt_table_html": """Convert the table in this image to HTML.""",
    # "prompt_table_latex": """Convert the table in this image to LaTeX.""",
    # "prompt_formula_latex": """Convert the formula in this image to LaTeX.""",
}


class rednote_vision_model():

    def __init__(self, model_name="DotsOCR", base_url="http://172.18.4.101:8080/v1", lang="Chinese", key="0", **kwargs):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name
        self.lang = lang
        # Configure retry parameters
        self.max_retries = kwargs.get("max_retries", 10)
        self.base_delay = kwargs.get("retry_interval", 2)
        self.max_rounds = kwargs.get("max_rounds", 5)
        self.is_tools = False
        self.tools = []
        self.toolcall_sessions = {}

    def image2base64(self, image):
        """Convert PIL Image to base64 string"""
        import base64
        from io import BytesIO
        
        # If image is a file path string
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image)
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

    def prompt(self, base64_image):
        """Generate basic prompt for image description"""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe this image in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

    def vision_llm_prompt(self, base64_image, prompt_text):
        """Generate prompt with custom text for vision LLM"""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

    def extract_text(self, image):
        """Extract text from image using OCR"""
        prompt = dict_promptmode_to_prompt["prompt_ocr"]
        return self.describe_with_prompt(image, prompt)

    def describe(self, image):
        b64 = self.image2base64(image)
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.prompt(b64),
        )
        return res.choices[0].message.content.strip(), res.usage.total_tokens

    def describe_with_prompt(self, image, prompt=None):
        b64 = self.image2base64(image)
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.vision_llm_prompt(b64, prompt),
        )
        return res.choices[0].message.content.strip(), res.usage.total_tokens


if __name__ == "__main__":
    # cv_model = rednote_vision_model()
    # prompt = dict_promptmode_to_prompt["prompt_layout_all_en"]
    # image = Image.open("/root/dev/jk-kms-ragflow/demo_image1.jpg")
    # docs = cv_model.describe_with_prompt(image, prompt)
    # print(docs)
    parser = rednote_vision_model()
    image_path = "hello_ai.png"
    try:
        from PIL import Image
        image = Image.open(image_path)
        result, tokens = parser.extract_text(image)
        print(f"Result: {result}")
        print(f"Tokens used: {tokens}")
    except Exception as e:
        print(f"Error processing image: {e}")
