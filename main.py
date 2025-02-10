from fastapi import FastAPI, UploadFile, File
import os
import json
import torch
import psycopg2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time

# Configuração do PostgreSQL (Aiven)
SERVICE_URI = os.getenv("DATABASE_URL")  # Pegando a variável de ambiente do Render

def connect_db():
    """Tenta conectar ao banco de dados. Se falhar, retorna None."""
    try:
        conn = psycopg2.connect(SERVICE_URI, connect_timeout=10)  # Timeout para evitar travamento
        return conn
    except Exception as e:
        print(f"❌ Erro ao conectar ao banco de dados: {e}")
        return None

# Carregar modelo CLIP com tempo de espera para garantir inicialização
print("⏳ Carregando modelo CLIP...")
start_time = time.time()
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print(f"✅ Modelo CLIP carregado em {time.time() - start_time:.2f} segundos!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo CLIP: {e}")

app = FastAPI()

@app.get("/")
def root():
    """Endpoint raiz para evitar erro 404"""
    return {"message": "API de Reconhecimento Facial está funcionando! 🚀"}

@app.get("/healthz")
def health_check():
    """Verifica se o banco de dados está online"""
    conn = connect_db()
    if conn:
        conn.close()
        return {"status": "OK", "db_connection": "Success"}
    return {"status": "FAIL", "db_connection": "Error"}

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
        return {"error": "Banco de dados offline"}

    cur = conn.cursor()
    cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", (file.filename, json.dumps(embedding.tolist())))
    conn.commit()

    cur.close()
    conn.close()
    print(f"✅ Imagem {file.filename} cadastrada com sucesso!")
    return {"message": "Imagem cadastrada com sucesso!"}

@app.get("/get_all_images")
def get_all_images():
    """Retorna todas as imagens armazenadas no banco"""
    print("📸 Buscando todas as imagens no banco...")
    
    conn = connect_db()
    if not conn:
        return {"error": "Banco de dados offline"}

    cur = conn.cursor()
    cur.execute("SELECT picture FROM pictures;")
    images = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    print(f"✅ {len(images)} imagens encontradas.")
    return {"images": images}

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    """Recebe uma imagem e busca correspondências no banco"""
    print(f"🔎 Buscando correspondências para a imagem enviada: {file.filename}")

    image = Image.open(file.file).convert("RGB")
    profile_embedding = generate_embedding(image)

    conn = connect_db()
    if not conn:
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

    return {"matches": matched_images"}