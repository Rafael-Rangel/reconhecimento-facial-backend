from fastapi import FastAPI, UploadFile, File
import os
import json
import torch
import psycopg2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configuração do PostgreSQL (Aiven)
import os
SERVICE_URI = os.getenv("DATABASE_URL")

def connect_db():
    """Tenta conectar ao banco de dados. Se falhar, retorna None."""
    try:
        conn = psycopg2.connect(SERVICE_URI)
        return conn
    except Exception as e:
        print(f"❌ Erro ao conectar ao banco de dados: {e}")
        return None  # Evita que a API feche caso o banco esteja offline

# Carregar modelo CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

app = FastAPI()

def generate_embedding(image):
    """Gera embedding da imagem"""
    inputs = processor(images=image, return_tensors="pt").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    return embedding.squeeze().numpy()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Recebe uma imagem, gera embedding e salva no banco"""
    print(f"📤 Recebendo imagem para upload: {file.filename}")
    
    image = Image.open(file.file).convert("RGB")
    embedding = generate_embedding(image)

    conn = connect_db()
    if not conn:
        print("❌ Banco de dados offline!")
        return {"error": "Banco de dados offline"}

    cur = conn.cursor()
    cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", (file.filename, json.dumps(embedding.tolist())))
    conn.commit()

    cur.close()
    conn.close()
    print(f"✅ Imagem {file.filename} cadastrada com sucesso!")
    return {"message": "Imagem cadastrada com sucesso!"}

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    """Recebe uma imagem e busca correspondências no banco"""
    print(f"🔎 Buscando correspondências para a imagem enviada: {file.filename}")

    image = Image.open(file.file).convert("RGB")
    profile_embedding = generate_embedding(image)

    conn = connect_db()
    if not conn:
        print("❌ Banco de dados offline!")
        return {"error": "Banco de dados offline"}

    cur = conn.cursor()
    cur.execute("SELECT picture, embedding FROM pictures;")
    matches = cur.fetchall()
    cur.close()
    conn.close()

    threshold = 0.4
    matched_images = []

    for match in matches:
        img_filename, img_embedding_str = match
        img_embedding = np.array(json.loads(img_embedding_str), dtype=np.float32)
        img_embedding = img_embedding / np.linalg.norm(img_embedding)

        similarity = np.dot(img_embedding, profile_embedding)

        if similarity >= threshold:
            matched_images.append({"image": img_filename, "similarity": similarity})
            print(f"✅ Correspondência encontrada: {img_filename} (Similaridade: {similarity:.2f})")

    if not matched_images:
        print("❌ Nenhuma correspondência encontrada!")

    return {"matches": matched_images}

@app.get("/healthz")
def health_check():
    """Verifica se o banco de dados está online"""
    conn = connect_db()
    if conn:
        return {"status": "OK", "db_connection": "Success"}
    return {"status": "FAIL", "db_connection": "Error"}