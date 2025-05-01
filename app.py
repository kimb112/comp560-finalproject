import os
import gradio as gr
import openai
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from googleapiclient.discovery import build
import numpy as np

# Load models and API key
embedder = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = os.getenv("openai_api_key")
YOUTUBE_API_KEY = os.getenv("youtube_api_key")

# OpenAI refinement function
def refine_query_with_openai(user_input):
    client = openai.OpenAI(api_key=openai.api_key)
    prompt = (
        "You are a helpful tutor and expert search assistant.\n"
        "Step 1: Briefly answer the user's question (1–2 sentences). Begin your answer with 'Answer:'.\n"
        "Step 2: Convert the question into a concise, keyword-rich YouTube search query based on your answer. Begin with 'Refined Search Query:'.\n"
        "Avoid vague language. Use precise scientific or technical terms if applicable.\n\n"
        "Example:\n"
        "User Question: how does the body make its own sugar?\n"
        "Answer: The body produces its own sugar through gluconeogenesis, which converts non-carbohydrate sources into glucose.\n"
        "Refined Search Query: gluconeogenesis explained\n\n"
        f"User Question: {user_input}\n"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=200
    )
    output = response.choices[0].message.content.strip()
    match = re.search(r"(?:Answer:\s*)?(.*?)\s*Refined Search Query:\s*(.*)", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return output, user_input

# YouTube search function
def fetch_youtube_data(query, max_results=20):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        part="snippet", maxResults=max_results, q=query, type="video"
    )
    response = request.execute()
    videos = [{
        "title": item["snippet"]["title"],
        "description": item["snippet"]["description"],
        "video_id": item["id"]["videoId"]
    } for item in response["items"]]
    return videos

# Full pipeline function
def semantic_search(user_query, explain_first):
    if explain_first:
        answer, refined_query = refine_query_with_openai(user_query)
    else:
        answer = None
        refined_query = user_query

    videos = fetch_youtube_data(refined_query)
    video_texts = [f"{v['title']} {v['description']}" for v in videos]
    embeddings = embedder.encode(video_texts)
    query_embedding = embedder.encode([refined_query])

    model = NearestNeighbors(n_neighbors=5, metric="cosine")
    model.fit(embeddings)
    distances, indices = model.kneighbors(query_embedding)

    video_blocks = ""
    for idx in indices[0]:
        v = videos[idx]
        video_id = v["video_id"]
        video_title = v["title"]
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        video_link = f"https://www.youtube.com/watch?v={video_id}"

        video_blocks += f"""
            <div style="margin: 0 10px; text-align: center; width: 280px;">
                <h4 style="font-size: 14px;">{video_title}</h4>
                <iframe width="280" height="157.5" src="{embed_url}" frameborder="0" allowfullscreen></iframe><br>
            </div>
        """

    html_output = f"""
    <div style="display: flex; justify-content: center; flex-wrap: wrap;">
        {video_blocks}
    </div>
    """

    explanation_text = f"<p><b>AI Answer:</b> {answer}</p><p>Here are some videos that explain this further:</p>" if answer else ""
    return explanation_text + html_output

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Semantic YouTube Search with AI Help")
    query_input = gr.Textbox(label="Ask a question")
    explain_box = gr.Checkbox(label="I don't know the answer — explain it to me first")
    search_btn = gr.Button("Search")
    results_output = gr.HTML()

    search_btn.click(fn=semantic_search, inputs=[query_input, explain_box], outputs=results_output)

demo.launch()
