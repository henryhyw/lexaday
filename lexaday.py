import os
import json
import random
import base64
from PIL import Image
from io import BytesIO
import requests
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.utils import formataddr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
# Comma-separated list of receiver emails in format: userID:email
raw_receivers = os.getenv("RECEIVER_EMAILS", "").split(",")
RECEIVER_MAPPING = {}
for r in raw_receivers:
    if ":" in r:
        uid, email = r.split(":", 1)
        RECEIVER_MAPPING[uid.strip()] = email.strip()

# (Optional) A link to online records or similar
RECORDS_LINK = os.getenv("RECORDS_LINK", "N/A")

# Files for shared data
RSDATA_FILENAME = "rsdata.json"  # Each item must have keys: "id", "term", "meaning", "examples", "motivational_quotes", etc.
IMAGES_FOLDER = "images"

# --- USER DATA FILE FUNCTIONS ---
def sanitize_filename(name: str) -> str:
    # Allow only alphanumeric characters for file names.
    return "".join(c if c.isalnum() else "_" for c in name)

def get_user_data_filename(user_id: str) -> str:
    return f"user_{sanitize_filename(user_id)}.json"

def load_user_data(user_id: str):
    filename = get_user_data_filename(user_id)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    # If not present, initialize with run_count 0 and empty records list.
    data = {"run_count": 0, "records": []}
    save_user_data(user_id, data)
    return data

def save_user_data(user_id: str, data):
    filename = get_user_data_filename(user_id)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_rsdata(rsdata):
    with open(RSDATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(rsdata, f, indent=4, ensure_ascii=False)

# --- STRUCTURED OUTPUT SCHEMAS ---
class ExampleSentenceResponse(BaseModel):
    sample_sentence: str

class MotivationalQuoteResponse(BaseModel):
    motivational_quote: str

# --- HELPER FUNCTION FOR HF TEXT GENERATION ---
def hf_generate_text(prompt: str, max_tokens: int, model: str, temperature: float) -> str:
    """
    Tries to generate text using the HF InferenceClient with the Qwen model,
    cycling through up to 10 tokens if needed.
    """
    token_keys = [
        "HF_API_TOKEN", "HF_API_TOKEN2", "HF_API_TOKEN3", "HF_API_TOKEN4", "HF_API_TOKEN5",
        "HF_API_TOKEN6", "HF_API_TOKEN7", "HF_API_TOKEN8", "HF_API_TOKEN9", "HF_API_TOKEN10"
    ]
    tokens = [os.getenv(key) for key in token_keys if os.getenv(key)]
    if not tokens:
        raise Exception("❌ No Hugging Face API tokens found in .env file.")

    for idx, token in enumerate(tokens):
        try:
            client = InferenceClient(provider="nebius", api_key=token)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "exceeded" in str(e) or "quota" in str(e):
                print(f"⚠️ Token {idx + 1} quota exceeded. Trying next token...")
            else:
                print(f"❌ Qwen API error with token {idx + 1}: {e}")
    return ""

# --- API FUNCTIONS ---
def query_chatgpt4_structured(prompt: str, model="Qwen/Qwen2.5-72B-Instruct") -> str:
    """
    Query Qwen to generate a sample sentence (structured as JSON).
    """
    try:
        content = hf_generate_text(prompt, max_tokens=60, model=model, temperature=0.7)
        try:
            parsed_json = json.loads(content)
            return parsed_json.get("sample_sentence", "").strip()
        except json.JSONDecodeError:
            return content.strip()
    except Exception as e:
        print("❌ Qwen API error:", e)
        return ""

def query_motivational_quote(word: str, model="Qwen/Qwen2.5-72B-Instruct") -> str:
    """
    Generate a moderate long motivational quote of the day that includes the given word.
    """
    prompt = (
        f"Generate a moderate long motivational quote of the day that includes the word '{word}'. "
        "Respond with a JSON object exactly matching this schema: {\"motivational_quote\": string}."
    )
    try:
        content = hf_generate_text(prompt, max_tokens=100, model=model, temperature=0.7)
        try:
            parsed_json = json.loads(content)
            return parsed_json.get("motivational_quote", "").strip()
        except json.JSONDecodeError:
            return content.strip()
    except Exception as e:
        print("❌ Qwen API error (motivational quote):", e)
        return ""

def query_stable_diffusion_prompt(term, model="Qwen/Qwen2.5-72B-Instruct") -> str:
    """
    Generate a Stable Diffusion prompt that visually represents the root word.
    """
    prompt = (
        f"Generate a prompt for Stable Diffusion that visually represents the concept: '{term['meaning']}'. "
        "The prompt should be concise, clear, and include only the necessary visual details. "
        "Respond with a JSON object exactly matching this schema: {\"sd_prompt\": string}."
    )
    try:
        content = hf_generate_text(prompt, max_tokens=100, model=model, temperature=0.7)
        try:
            parsed_json = json.loads(content)
            return parsed_json.get("sd_prompt", "").strip()
        except json.JSONDecodeError:
            return content.strip()
    except Exception as e:
        print("❌ Qwen API error (stable diffusion prompt):", e)
        return ""

def generate_image_dalle(prompt: str, filename: str):
    """
    Generate an image using the Hugging Face Inference API with Stable Diffusion.
    Automatically switch between HF API tokens if quota is exceeded.
    """
    print(f"\n🎨 Generating image for prompt: {prompt[:60]}...")

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    token_keys = [
        "HF_API_TOKEN", "HF_API_TOKEN2", "HF_API_TOKEN3", "HF_API_TOKEN4", "HF_API_TOKEN5",
        "HF_API_TOKEN6", "HF_API_TOKEN7", "HF_API_TOKEN8", "HF_API_TOKEN9", "HF_API_TOKEN10"
    ]
    tokens = [os.getenv(key) for key in token_keys if os.getenv(key)]

    if not tokens:
        raise Exception("❌ No Hugging Face API tokens found in .env file.")

    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 30,
            "guidance_scale": 7,
            "width": 1024,
            "height": 1024
        },
        "options": {"wait_for_model": True}
    }

    for idx, token in enumerate(tokens):
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "image/png"
        }
        print(f"🪪 Using HF token {idx + 1}...")
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"✅ Image saved as {filename}")
                return
            elif "exceeded your monthly included credits" in response.text:
                print(f"⚠️ Token {idx + 1} quota exceeded. Trying next token...")
            else:
                print(f"❌ Error with token {idx + 1}: {response.text[:300]}")
        except Exception as e:
            print(f"❌ Request failed with token {idx + 1}: {e}")

    print("🚫 All Hugging Face tokens exhausted or failed. Image not generated.")

# --- DATA MANAGEMENT FUNCTIONS (for shared rsdata and per-user records) ---
def load_rsdata():
    if os.path.exists(RSDATA_FILENAME):
        with open(RSDATA_FILENAME, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️ {RSDATA_FILENAME} not found. Please ensure it exists.")
        return []

def introduce_new_term(records, rsdata, current_run):
    introduced_ids = {record["id"] for record in records}
    candidates = [term for term in rsdata if term["id"] not in introduced_ids]
    if not candidates:
        print("No new terms available.")
        return None
    new_term = random.choice(candidates)
    record = {
        "id": new_term["id"],
        "first_introduced": current_run,
        "last_reviewed": current_run,
        "review_interval": 1
    }
    records.append(record)
    print(f"🆕 New term introduced: {new_term['term']} (meaning: {new_term['meaning']})")
    return new_term

def get_due_review_records(records, current_run):
    due = []
    for record in records:
        if (current_run - record["last_reviewed"]) >= record["review_interval"]:
            due.append(record)
    return due

def get_term_by_id(rsdata, term_id):
    for term in rsdata:
        if term["id"] == term_id:
            return term
    return None

# --- NEW HELPER FUNCTION: Update imagesIndex.json ---
def update_images_index(image_filename: str):
    """Append the image filename to imagesIndex.json if it's not already present."""
    index_filename = "imagesIndex.json"
    images_list = []
    if os.path.exists(index_filename):
        with open(index_filename, "r", encoding="utf-8") as f:
            try:
                images_list = json.load(f)
            except json.JSONDecodeError:
                images_list = []
    if image_filename not in images_list:
        images_list.append(image_filename)
        with open(index_filename, "w", encoding="utf-8") as f:
            json.dump(images_list, f, indent=4, ensure_ascii=False)

# --- UPDATED generate_term_image FUNCTION ---
def generate_term_image(term, user_id):
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    filename = os.path.join(IMAGES_FOLDER, f"{term['id']}_{user_id}.png")
    # If the image exists, update the index and return it
    if os.path.exists(filename):
        print(f"Image for term '{term['term']}' already exists for user {user_id}: {filename}")
        update_images_index(os.path.basename(filename))
        return filename

    sd_prompt = query_stable_diffusion_prompt(term)
    if not sd_prompt:
        sd_prompt = f"Create an image that visually represents the root word '{term['term']}' which means '{term['meaning']}'."
    generate_image_dalle(sd_prompt, filename)
    update_images_index(os.path.basename(filename))
    return filename

def generate_example_sentence(example_word, example_meaning, term):
    prompt = (
        f"Using the word '{example_word}', which means '{example_meaning}' as used in the context of the term '{term['term']}', "
        "provide one simple, clear, and concise sentence. "
        "Respond with a JSON object exactly matching this schema: {\"sample_sentence\": string}."
    )
    return query_chatgpt4_structured(prompt)

# --- EMBED IMAGES INLINE AND SEND HTML EMAIL ---
def send_email(subject: str, html_body: str, to_email: str, inline_image_paths: list):
    msg_root = MIMEMultipart('related')
    msg_root['Subject'] = subject
    msg_root['From'] = formataddr(("Lexaday Bot", SENDER_EMAIL))
    msg_root['To'] = to_email

    msg_alternative = MIMEMultipart('alternative')
    msg_root.attach(msg_alternative)

    html_part = MIMEText(html_body, 'html', _charset="utf-8")
    msg_alternative.attach(html_part)

    for path in inline_image_paths:
        if not os.path.exists(path):
            continue
        filename = os.path.basename(path)
        try:
            with open(path, 'rb') as f:
                img_data = f.read()
            mime_img = MIMEImage(img_data)
            mime_img.add_header('Content-ID', f'<{filename}>')
            mime_img.add_header('Content-Disposition', 'inline', filename=filename)
            msg_root.attach(mime_img)
        except Exception as e:
            print(f"❌ Failed to embed image {path}: {e}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg_root)
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")

# --- MAIN WORKFLOW FOR EACH USER ---
def process_for_user(user_id: str, user_email: str):
    print(f"\n========== Processing for {user_email} (User ID: {user_id}) ==========")
    user_data = load_user_data(user_id)
    user_data["run_count"] += 1
    current_run = user_data["run_count"]
    print(f"🔢 Current run count for {user_email}: {current_run}\n")
    
    rsdata = load_rsdata()
    records = user_data.get("records", [])
    
    new_term = introduce_new_term(records, rsdata, current_run)
    new_term_image_path = None
    if new_term:
        new_term_image_path = generate_term_image(new_term, user_id)
    
    # Generate a motivational quote (replacing the earlier "encouraging sentence")
    if new_term and new_term.get("examples"):
        example_word = random.choice(new_term["examples"])[0]
    else:
        example_word = new_term["term"] if new_term else "keep going"
    motivational_quote = query_motivational_quote(example_word)
    motivational_quote = motivational_quote.replace(example_word, f"<strong>{example_word}</strong>")
    
    # Store the motivational quote in rsdata for the term
    if new_term:
        if "motivational_quotes" not in new_term or not isinstance(new_term["motivational_quotes"], list):
            new_term["motivational_quotes"] = []
        new_term["motivational_quotes"].append(motivational_quote)
    
    # Update sample sentences for new term examples (stored within each example list starting from the third entry)
    if new_term and new_term.get("examples"):
        examples = new_term["examples"]
        for i, ex in enumerate(examples):
            ex_word = ex[0]
            ex_meaning = ex[1]
            sentence = generate_example_sentence(ex_word, ex_meaning, new_term)
            if isinstance(new_term["examples"][i], list):
                new_term["examples"][i].append(sentence)
            else:
                new_term["examples"][i] = [ex_word, ex_meaning, sentence]
    
    due_records = get_due_review_records(records, current_run)
    review_images = []
    if due_records:
        for rec in due_records:
            term_obj = get_term_by_id(rsdata, rec["id"])
            if term_obj:
                path = generate_term_image(term_obj, user_id)
                if path and os.path.exists(path):
                    review_images.append(path)
                # Update sample sentences for review term examples
                if term_obj.get("examples"):
                    for i, ex in enumerate(term_obj["examples"]):
                        ex_word = ex[0]
                        ex_meaning = ex[1]
                        sentence = generate_example_sentence(ex_word, ex_meaning, term_obj)
                        if isinstance(term_obj["examples"][i], list):
                            term_obj["examples"][i].append(sentence)
                        else:
                            term_obj["examples"][i] = [ex_word, ex_meaning, sentence]
                rec["last_reviewed"] = current_run
                rec["review_interval"] *= 2
    else:
        print("✅ No terms are due for review this run.")

    save_user_data(user_id, user_data)
    save_rsdata(rsdata)

    html_body = f"""
<html>
<head>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: auto;
            width: 90%;
        }}
        .card {{
            margin-top: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
            width: 100%;
        }}
        .image img {{
            display: block;
            width: 80%;
            margin: 10px 10%;
            height: auto;
        }}
        .examples {{
            padding: 10px 15px;
        }}
        .example {{
            margin-bottom: 10px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 10px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 0.9em;
            color: #777777;
        }}
        a {{
            color: #1a73e8;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center;">Daily Lexaday Update</h1>
        <p style="text-align: center; font-style: italic; margin-top: 10px;">{motivational_quote}</p>
"""
    if new_term:
        html_body += f"""
            <div class="card">
                <table style="width: 100%; background-color: #fafafa; padding: 10px; border-collapse: collapse;">
                    <tr>
                        <td style="width: 50%; text-align: center; font-size: 1.8em; font-weight: bold;">
                            {new_term['term']}
                        </td>
                        <td style="width: auto; text-align: center;">|</td>
                        <td style="width: 50%; text-align: center; font-size: 1.4em; color: #555;">
                            {new_term['meaning']}
                        </td>
                    </tr>
                </table>
                <div class="image">
                    <img src="cid:{new_term['id']}_{user_id}.png" alt="{new_term['term']} image"/>
                </div>
                <hr/>
                <div class="examples">
                    <h3>Examples</h3>
        """
        examples = new_term.get("examples", [])
        for i, ex in enumerate(examples):
            ex_word = ex[0]
            ex_meaning = ex[1]
            sample_sentence = ex[-1] if len(ex) > 2 else ""
            html_body += f"""
                    <div class="example">
                        <p><strong>{ex_word}:</strong> {ex_meaning}</p>
                        <p>{sample_sentence}</p>
                    </div>
            """
            if i != len(examples) - 1:
                html_body += "<hr/>"
        html_body += """
                </div>
            </div>
        """
    if due_records:
        html_body += """
            <div class="review-section">
                <h2 style="text-align: center;">Let's Revisit These</h2>
        """
        for rec in due_records:
            term_obj = get_term_by_id(rsdata, rec["id"])
            if term_obj:
                html_body += f"""
                <div class="card">
                    <table style="width: 100%; background-color: #fafafa; padding: 10px; border-collapse: collapse;">
                        <tr>
                            <td style="width: 50%; text-align: center; font-size: 1.8em; font-weight: bold;">
                                {term_obj['term']}
                            </td>
                            <td style="width: auto; text-align: center;">|</td>
                            <td style="width: 50%; text-align: center; font-size: 1.4em; color: #555;">
                                {term_obj['meaning']}
                            </td>
                        </tr>
                    </table>
                    <div class="image">
                        <img src="cid:{term_obj['id']}_{user_id}.png" alt="{term_obj['term']} image"/>
                    </div>
                    <hr/>
                    <div class="examples">
                        <h3>Examples</h3>
                """
                examples = term_obj.get("examples", [])
                for i, ex in enumerate(examples):
                    ex_word = ex[0]
                    ex_meaning = ex[1]
                    sample_sentence = ex[-1] if len(ex) > 2 else ""
                    html_body += f"""
                        <div class="example">
                            <p><strong>{ex_word}:</strong> {ex_meaning}</p>
                            <p>{sample_sentence}</p>
                        </div>
                    """
                    if i != len(examples) - 1:
                        html_body += "<hr/>"
                html_body += """
                    </div>
                </div>
                """
        html_body += "</div>"
        
    html_body += """
            </div>
        </body>
        </html>
        """

    inline_images = []
    if new_term and new_term_image_path and os.path.exists(new_term_image_path):
        inline_images.append(new_term_image_path)
    review_images_set = set(review_images)
    for path in review_images_set:
        if path not in inline_images:
            inline_images.append(path)

    subject = "Daily Lexaday Update"
    send_email(subject, html_body, user_email, inline_images)

# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    if not RECEIVER_MAPPING:
        print("No receiver emails specified in environment variables.")
    else:
        for user_id, email in RECEIVER_MAPPING.items():
            if email:
                process_for_user(user_id, email)