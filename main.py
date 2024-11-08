from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import random
from typing import List
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import albumentations as A
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import io
import base64
from scipy.io import wavfile
from scipy.signal import convolve
import os
import trimesh
from pathlib import Path
import tempfile
import json
from scipy.spatial.transform import Rotation
from pydantic import BaseModel

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store text data in memory (in production, use proper database)
text_storage = {
    "original_text": "",
    "processed_text": "",
    "augmented_text": "",
    "preprocessing_steps": [],
    "preprocessing_results": {},
    "augmentation_results": {}
}

# Add these to your existing text_storage or create new image_storage
image_storage = {
    "original_image": None,
    "preprocessing_results": {},
    "merged_preprocessing": None,
    "augmentation_results": {},
    "preprocessing_steps": []
}

# Add to your existing storage
audio_storage = {
    "original_audio": None,
    "original_sr": None,
    "preprocessing_results": {},
    "merged_preprocessing": None,
    "augmentation_results": {}
}

# Add a storage dictionary for 3D models
model_storage = {
    "original_model": None,
    "preprocessed_model": None,
    "preprocessing_results": {},
    "augmentation_results": {}
}

# Add OFF file handler
def load_off_file(file_path):
    """
    Load an OFF file and convert it to trimesh format
    """
    try:
        print(f"Opening OFF file: {file_path}")  # Debug log
        vertices = []
        faces = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Skip empty lines and comments
            lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            
            # Read header
            header = lines[0].strip()
            if header != 'OFF':
                raise ValueError(f'Not a valid OFF file. Header was: {header}')
            
            # Read number of vertices, faces, edges
            counts = lines[1].strip().split()
            n_vertices = int(counts[0])
            n_faces = int(counts[1])
            
            print(f"Expected vertices: {n_vertices}, faces: {n_faces}")  # Debug log
            
            # Read vertices
            current_line = 2
            for i in range(n_vertices):
                vertex = lines[current_line + i].strip().split()
                vertices.append([float(x) for x in vertex[:3]])
            
            current_line += n_vertices
            
            # Read faces
            for i in range(n_faces):
                face = lines[current_line + i].strip().split()
                # OFF format specifies number of vertices first
                n_face_vertices = int(face[0])
                if n_face_vertices == 3:
                    faces.append([int(x) for x in face[1:4]])
                else:
                    print(f"Warning: Found non-triangular face with {n_face_vertices} vertices")
                    # Convert to triangles if possible
                    for j in range(1, n_face_vertices - 1):
                        faces.append([int(face[1]), int(face[j+1]), int(face[j+2])])
        
        print(f"Loaded {len(vertices)} vertices and {len(faces)} faces")  # Debug log
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)
        
        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print("Mesh created successfully")  # Debug log
        return mesh
        
    except Exception as e:
        print(f"Error in load_off_file: {str(e)}")  # Debug log
        raise Exception(f"Failed to load OFF file: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "data_types": [
            {
                "name": "Text Data",
                "description": "Upload and augment text documents",
                "route": "/text_upload"
            },
            {
                "name": "Image Data",
                "description": "Upload and augment image files",
                "route": "/image_upload"
            },
            {
                "name": "Audio Data",
                "description": "Upload and augment audio files",
                "route": "/audio_upload"
            },
            {
                "name": "3D Data",
                "description": "Upload and augment 3D models and point clouds",
                "route": "/3d_upload"
            }
        ]
    })

@app.get("/text_upload", response_class=HTMLResponse)
async def text_upload_page(request: Request):
    return templates.TemplateResponse("text_upload.html", {
        "request": request,
        "text_content": text_storage["original_text"]
    })

@app.post("/upload_text")
async def upload_text(file: UploadFile = File(...)):
    content = await file.read()
    text_storage["original_text"] = content.decode()
    return {"text": text_storage["original_text"]}

@app.post("/update_text")
async def update_text(text: str = Form(...)):
    text_storage["original_text"] = text
    return {"text": text_storage["original_text"]}

@app.get("/preprocess", response_class=HTMLResponse)
async def preprocess_page(request: Request):
    # Only show results if there's original text
    show_results = bool(text_storage.get("original_text", "").strip())
    
    return templates.TemplateResponse("preprocess.html", {
        "request": request,
        "original_text": text_storage.get("original_text", ""),
        "preprocessing_results": text_storage.get("preprocessing_results", {}) if show_results else {},
        "merged_result": text_storage.get("merged_preprocessing", "") if show_results else "",
        "selected_steps": text_storage.get("preprocessing_steps", [])
    })

@app.post("/apply_preprocessing")
async def apply_preprocessing(steps: List[str] = Form(...)):
    text = text_storage["original_text"]
    results = {}
    merged_text = text
    
    for step in steps:
        if step == "tokenize":
            results["tokenize"] = word_tokenize(text)
            merged_text = word_tokenize(merged_text)
            
        elif step == "lowercase":
            results["lowercase"] = text.lower()
            merged_text = merged_text.lower() if isinstance(merged_text, str) else " ".join(merged_text).lower()
            
        elif step == "remove_punctuation":
            results["remove_punctuation"] = text.translate(
                str.maketrans("", "", string.punctuation)
            )
            merged_text = merged_text.translate(
                str.maketrans("", "", string.punctuation)
            ) if isinstance(merged_text, str) else " ".join(merged_text).translate(
                str.maketrans("", "", string.punctuation)
            )
            
        elif step == "remove_stopwords":
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text)
            results["remove_stopwords"] = " ".join(
                [word for word in words if word.lower() not in stop_words]
            )
            merged_words = merged_text if isinstance(merged_text, list) else word_tokenize(merged_text)
            merged_text = " ".join([word for word in merged_words if word.lower() not in stop_words])
            
        elif step == "remove_numbers":
            results["remove_numbers"] = re.sub(r'\d+', '', text)
            merged_text = re.sub(r'\d+', '', merged_text) if isinstance(merged_text, str) else " ".join(
                re.sub(r'\d+', '', word) for word in merged_text
            )
            
        elif step == "lemmatize":
            lemmatizer = WordNetLemmatizer()
            words = word_tokenize(text)
            results["lemmatize"] = " ".join(
                [lemmatizer.lemmatize(word) for word in words]
            )
            merged_words = merged_text if isinstance(merged_text, list) else word_tokenize(merged_text)
            merged_text = " ".join([lemmatizer.lemmatize(word) for word in merged_words])

    text_storage["preprocessing_steps"] = steps
    text_storage["preprocessing_results"] = results
    text_storage["merged_preprocessing"] = merged_text
    
    if "augmentation_results" in text_storage and text_storage["augmentation_results"]:
        previous_techniques = text_storage.get("last_used_techniques", [])
        if previous_techniques:
            original_results = {}
            preprocessed_results = {}
            
            for technique in previous_techniques:
                if technique == "synonym_replacement":
                    original_results["synonym_replacement"] = synonym_replacement(text)
                    preprocessed_results["synonym_replacement"] = synonym_replacement(merged_text)
                elif technique == "random_insertion":
                    original_results["random_insertion"] = random_insertion(text)
                    preprocessed_results["random_insertion"] = random_insertion(merged_text)
                elif technique == "random_swap":
                    original_results["random_swap"] = random_swap(text)
                    preprocessed_results["random_swap"] = random_swap(merged_text)
                elif technique == "random_deletion":
                    original_results["random_deletion"] = random_deletion(text)
                    preprocessed_results["random_deletion"] = random_deletion(merged_text)
            
            text_storage["augmentation_results"] = {
                "original": original_results,
                "preprocessed": preprocessed_results
            }
    
    return {"step_results": results, "merged_result": merged_text}

@app.get("/augment", response_class=HTMLResponse)
async def augment_page(request: Request):
    # Only show results if there's original text
    show_results = bool(text_storage.get("original_text", "").strip())
    
    return templates.TemplateResponse("augment.html", {
        "request": request,
        "original_text": text_storage.get("original_text", ""),
        "processed_text": text_storage.get("preprocessing_results", {}) if show_results else {},
        "augmented_text": text_storage.get("augmentation_results", {}) if show_results else {}
    })

@app.post("/apply_augmentation")
async def apply_augmentation(techniques: List[str] = Form(...)):
    original_results = {}
    preprocessed_results = {}
    
    original_text = text_storage["original_text"]
    preprocessed_text = text_storage.get("merged_preprocessing", original_text)
    
    text_storage["last_used_techniques"] = techniques
    
    for technique in techniques:
        if technique == "synonym_replacement":
            original_results["synonym_replacement"] = synonym_replacement(original_text)
            preprocessed_results["synonym_replacement"] = synonym_replacement(preprocessed_text)
            
        elif technique == "random_insertion":
            original_results["random_insertion"] = random_insertion(original_text)
            preprocessed_results["random_insertion"] = random_insertion(preprocessed_text)
            
        elif technique == "random_swap":
            original_results["random_swap"] = random_swap(original_text)
            preprocessed_results["random_swap"] = random_swap(preprocessed_text)
            
        elif technique == "random_deletion":
            original_results["random_deletion"] = random_deletion(original_text)
            preprocessed_results["random_deletion"] = random_deletion(preprocessed_text)

    text_storage["augmentation_results"] = {
        "original": original_results,
        "preprocessed": preprocessed_results
    }
    
    return {
        "original": original_results,
        "preprocessed": preprocessed_results
    }

# Augmentation helper functions
def synonym_replacement(text, n=1):
    words = word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    
    for _ in range(n):
        if not random_word_list:
            break
        random_word = random.choice(random_word_list)
        synonyms = []
        
        for syn in wordnet.synsets(random_word):
            for lemma in syn.lemmas():
                if lemma.name() != random_word:
                    synonyms.append(lemma.name())
                    
        if synonyms:
            synonym = random.choice(list(set(synonyms)))
            random_idx = random.randint(0, len(new_words)-1)
            new_words[random_idx] = synonym
            
    return " ".join(new_words)

def random_insertion(text, n=1):
    words = word_tokenize(text)
    new_words = words.copy()
    
    for _ in range(n):
        add_word = random.choice(words)
        random_idx = random.randint(0, len(new_words))
        new_words.insert(random_idx, add_word)
        
    return " ".join(new_words)

def random_swap(text, n=1):
    words = word_tokenize(text)
    new_words = words.copy()
    
    for _ in range(n):
        if len(new_words) >= 2:
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
    return " ".join(new_words)

def random_deletion(text, p=0.1):
    words = word_tokenize(text)
    new_words = []
    
    for word in words:
        if random.random() > p:
            new_words.append(word)
            
    return " ".join(new_words)

# Add new endpoints for image handling
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Store the original image
        image_storage["original_image"] = image
        
        # Convert to base64 for display
        encoded_image = encode_image(image)
        
        return {"success": True, "image": encoded_image}
    except Exception as e:
        print(f"Error in upload_image: {str(e)}")  # For debugging
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/apply_image_preprocessing")
async def apply_image_preprocessing(steps: List[str] = Form(...)):
    try:
        if image_storage["original_image"] is None:
            raise HTTPException(status_code=400, detail="No image uploaded")
        
        image = image_storage["original_image"].copy()
        results = {}
        sizes = {}
        merged_image = image.copy()
        
        for step in steps:
            try:
                if step == "grayscale":
                    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                elif step == "blur":
                    processed = cv2.GaussianBlur(merged_image, (5, 5), 0)
                elif step == "sharpen":
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    processed = cv2.filter2D(merged_image, -1, kernel)
                elif step == "contrast":
                    processed = cv2.convertScaleAbs(merged_image, alpha=1.5, beta=0)
                elif step == "brightness":
                    processed = cv2.convertScaleAbs(merged_image, alpha=1.0, beta=50)
                elif step == "equalize":
                    if len(merged_image.shape) == 3:
                        yuv = cv2.cvtColor(merged_image, cv2.COLOR_BGR2YUV)
                        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                        processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    else:
                        processed = cv2.equalizeHist(merged_image)
                
                # Store results and sizes
                results[step] = encode_image(processed)
                h, w = processed.shape[:2]
                sizes[step] = f"{w}x{h} px"
                merged_image = processed.copy()
                
            except Exception as step_error:
                print(f"Error processing step {step}: {str(step_error)}")
                continue
        
        # Store results
        image_storage["preprocessing_results"] = results
        image_storage["merged_preprocessing"] = encode_image(merged_image)
        image_storage["preprocessed_image"] = merged_image
        
        # Get merged size
        merged_height, merged_width = merged_image.shape[:2]
        merged_size = f"{merged_width}x{merged_height} px"
        
        return {
            "success": True,
            "step_results": results,
            "sizes": sizes,
            "merged_result": encode_image(merged_image),
            "merged_size": merged_size
        }
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/apply_image_augmentation")
async def apply_image_augmentation(techniques: List[str] = Form(...)):
    try:
        if image_storage["original_image"] is None:
            raise HTTPException(status_code=400, detail="No image uploaded")
        
        original_image = image_storage["original_image"]
        preprocessed_image = image_storage.get("preprocessed_image", original_image)
        
        original_results = {}
        preprocessed_results = {}
        
        # Get image dimensions
        original_height, original_width = original_image.shape[:2]
        preprocessed_height, preprocessed_width = preprocessed_image.shape[:2]
        
        for technique in techniques:
            if technique == "horizontal_flip":
                transform = A.HorizontalFlip(p=1.0)
                
            elif technique == "vertical_flip":
                transform = A.VerticalFlip(p=1.0)
                
            elif technique == "rotate":
                transform = A.Rotate(limit=45, p=1.0)
                
            elif technique == "random_brightness_contrast":
                transform = A.RandomBrightnessContrast(p=1.0)
                
            elif technique == "random_gamma":
                transform = A.RandomGamma(p=1.0)
                
            elif technique == "blur":
                transform = A.Blur(blur_limit=7, p=1.0)
                
            elif technique == "elastic":
                transform = A.ElasticTransform(p=1.0)
                
            elif technique == "grid_distortion":
                transform = A.GridDistortion(p=1.0)
            
            # Apply transformations
            augmented_original = transform(image=original_image)["image"]
            augmented_preprocessed = transform(image=preprocessed_image)["image"]
            
            original_results[technique] = {
                "image": encode_image(augmented_original),
                "size": f"{original_width}x{original_height} px"
            }
            preprocessed_results[technique] = {
                "image": encode_image(augmented_preprocessed),
                "size": f"{preprocessed_width}x{preprocessed_height} px"
            }

        return {
            "success": True,
            "original": original_results,
            "preprocessed": preprocessed_results,
            "original_size": f"{original_width}x{original_height} px",
            "preprocessed_size": f"{preprocessed_width}x{preprocessed_height} px"
        }
        
    except Exception as e:
        print(f"Error in augmentation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def encode_image(image):
    try:
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error in encode_image: {str(e)}")  # For debugging
        raise e

@app.get("/image_upload", response_class=HTMLResponse)
async def image_upload_page(request: Request):
    return templates.TemplateResponse("image_upload.html", {"request": request})

@app.get("/image_preprocess", response_class=HTMLResponse)
async def image_preprocess_page(request: Request):
    if image_storage["original_image"] is None:
        return RedirectResponse(url="/image_upload")
    
    original_image = image_storage["original_image"]
    height, width = original_image.shape[:2]
    original_size = f"{width}x{height} px"
    
    # Get sizes for preprocessing results
    result_sizes = {}
    preprocessing_results = image_storage.get("preprocessing_results", {})
    for step, img_data in preprocessing_results.items():
        # Decode base64 to get image dimensions
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        result_sizes[step] = f"{w}x{h} px"
    
    # Get merged result size
    merged_size = original_size
    if image_storage.get("merged_preprocessing"):
        merged_img = image_storage["preprocessed_image"]
        h, w = merged_img.shape[:2]
        merged_size = f"{w}x{h} px"
    
    return templates.TemplateResponse("image_preprocess.html", {
        "request": request,
        "original_image": encode_image(original_image),
        "original_size": original_size,
        "preprocessing_results": preprocessing_results,
        "result_sizes": result_sizes,
        "merged_result": image_storage.get("merged_preprocessing", ""),
        "merged_size": merged_size
    })

# Add the augmentation page route
@app.get("/image_augment", response_class=HTMLResponse)
async def image_augment_page(request: Request):
    if image_storage["original_image"] is None:
        return RedirectResponse(url="/image_upload")
    
    return templates.TemplateResponse("image_augment.html", {
        "request": request,
        "original_image": encode_image(image_storage["original_image"]),
        "preprocessed_image": image_storage.get("merged_preprocessing", "")
    })

@app.get("/audio_upload", response_class=HTMLResponse)
async def audio_upload_page(request: Request):
    return templates.TemplateResponse("audio_upload.html", {
        "request": request
    })

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Save temporarily to read with librosa
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(contents)
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(temp_path, sr=None)
            
            # Store original audio
            audio_storage["original_audio"] = audio
            audio_storage["original_sr"] = sr
            
            # Convert to base64 for client
            audio_base64 = encode_audio(audio, sr)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                "success": True,
                "message": "Audio uploaded successfully",
                "audio": audio_base64,
                "duration": len(audio) / sr,
                "sample_rate": sr
            }
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        print(f"Error in upload_audio: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/apply_audio_preprocessing")
async def apply_audio_preprocessing(steps: List[str] = Form(...)):
    print("Starting preprocessing with steps:", steps)  # Debug log
    try:
        if audio_storage["original_audio"] is None:
            raise HTTPException(status_code=400, detail="No audio uploaded")
        
        audio = audio_storage["original_audio"].copy()
        sr = audio_storage["original_sr"]
        results = {}
        
        for step in steps:
            try:
                print(f"Processing step: {step}")  # Debug log
                if step == "noise_reduction":
                    # Spectral noise reduction
                    S = librosa.stft(audio)
                    S_db = librosa.amplitude_to_db(np.abs(S))
                    processed = librosa.db_to_amplitude(
                        librosa.decompose.nn_filter(S_db)
                    )
                    processed = librosa.istft(processed * np.exp(1.j * np.angle(S)))
                
                elif step == "normalize":
                    processed = librosa.util.normalize(audio)
                
                elif step == "lowpass_filter":
                    nyquist = sr // 2
                    cutoff = nyquist // 4
                    b, a = signal.butter(4, cutoff / nyquist, btype='low')
                    processed = signal.filtfilt(b, a, audio)
                
                elif step == "highpass_filter":
                    nyquist = sr // 2
                    cutoff = nyquist // 8
                    b, a = signal.butter(4, cutoff / nyquist, btype='high')
                    processed = signal.filtfilt(b, a, audio)
                
                elif step == "trim_silence":
                    processed, _ = librosa.effects.trim(audio, top_db=20)
                
                elif step == "stereo_to_mono":
                    if len(audio.shape) > 1:
                        processed = librosa.to_mono(audio)
                    else:
                        processed = audio
                
                # Store results
                results[step] = {
                    "audio": encode_audio(processed, sr),
                    "duration": f"{len(processed) / sr:.2f}s",
                    "sample_rate": sr
                }
                
                # Update audio for next step
                audio = processed

            except Exception as step_error:
                print(f"Error in {step}: {str(step_error)}")
                continue
        
        # Store final processed audio
        audio_storage["preprocessing_results"] = results
        audio_storage["merged_preprocessing"] = {
            "audio": encode_audio(audio, sr),
            "duration": f"{len(audio) / sr:.2f}s",
            "sample_rate": sr
        }
        audio_storage["preprocessed_audio"] = audio
        
        print("Preprocessing completed successfully")  # Debug log
        return {
            "success": True,
            "step_results": results,
            "merged_result": audio_storage["merged_preprocessing"]
        }
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def encode_audio(audio, sr):
    """Convert audio array to base64 string"""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='wav')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return audio_base64

@app.get("/audio_preprocess", response_class=HTMLResponse)
async def audio_preprocess_page(request: Request):
    if audio_storage["original_audio"] is None:
        return RedirectResponse(url="/audio_upload")
    
    # Get audio duration and sample rate
    duration = len(audio_storage["original_audio"]) / audio_storage["original_sr"]
    
    return templates.TemplateResponse("audio_preprocess.html", {
        "request": request,
        "original_audio": encode_audio(audio_storage["original_audio"], 
                                     audio_storage["original_sr"]),
        "original_duration": f"{duration:.2f}s",
        "sample_rate": audio_storage["original_sr"],
        "preprocessing_results": audio_storage.get("preprocessing_results", {}),
        "merged_result": audio_storage.get("merged_preprocessing", None)
    })

@app.post("/apply_audio_augmentation")
async def apply_audio_augmentation(techniques: List[str] = Form(...)):
    try:
        if audio_storage["original_audio"] is None:
            raise HTTPException(status_code=400, detail="No audio uploaded")
        
        original_audio = audio_storage["original_audio"]
        preprocessed_audio = audio_storage.get("preprocessed_audio", original_audio)
        sr = audio_storage["original_sr"]
        
        original_results = {}
        preprocessed_results = {}
        
        for technique in techniques:
            try:
                # Apply augmentation to both original and preprocessed audio
                if technique == "time_stretch":
                    rate = random.uniform(0.8, 1.2)
                    aug_original = librosa.effects.time_stretch(original_audio, rate=rate)
                    aug_preprocessed = librosa.effects.time_stretch(preprocessed_audio, rate=rate)
                    params = f"Rate: {rate:.2f}x"
                    
                elif technique == "pitch_shift":
                    steps = random.randint(-4, 4)
                    aug_original = librosa.effects.pitch_shift(original_audio, sr=sr, n_steps=steps)
                    aug_preprocessed = librosa.effects.pitch_shift(preprocessed_audio, sr=sr, n_steps=steps)
                    params = f"Steps: {steps}"
                    
                elif technique == "add_noise":
                    noise_factor = 0.005
                    noise = np.random.randn(len(original_audio))
                    aug_original = original_audio + noise_factor * noise
                    aug_preprocessed = preprocessed_audio + noise_factor * noise[:len(preprocessed_audio)]
                    params = f"Noise factor: {noise_factor}"
                    
                elif technique == "speed_up":
                    speed_factor = random.uniform(0.9, 1.1)
                    aug_original = librosa.effects.time_stretch(original_audio, rate=speed_factor)
                    aug_preprocessed = librosa.effects.time_stretch(preprocessed_audio, rate=speed_factor)
                    params = f"Speed: {speed_factor:.2f}x"
                    
                elif technique == "volume_adjust":
                    volume_factor = random.uniform(0.5, 1.5)
                    aug_original = original_audio * volume_factor
                    aug_preprocessed = preprocessed_audio * volume_factor
                    params = f"Volume: {volume_factor:.2f}x"
                    
                elif technique == "reverb":
                    reverb_length = int(sr * 0.1)  # 100ms reverb
                    reverb = np.exp(-np.linspace(0, 4, reverb_length))
                    aug_original = signal.convolve(original_audio, reverb, mode='same')
                    aug_preprocessed = signal.convolve(preprocessed_audio, reverb, mode='same')
                    params = "Reverb: 100ms"

                # Store results with metadata
                original_results[technique] = {
                    "audio": encode_audio(aug_original, sr),
                    "duration": f"{len(aug_original) / sr:.2f}s",
                    "sample_rate": sr,
                    "parameters": params
                }
                
                preprocessed_results[technique] = {
                    "audio": encode_audio(aug_preprocessed, sr),
                    "duration": f"{len(aug_preprocessed) / sr:.2f}s",
                    "sample_rate": sr,
                    "parameters": params
                }
                
            except Exception as technique_error:
                print(f"Error in {technique}: {str(technique_error)}")
                continue
        
        return {
            "success": True,
            "original_results": original_results,
            "preprocessed_results": preprocessed_results
        }
        
    except Exception as e:
        print(f"Error in augmentation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def get_augmentation_params(technique):
    """Return human-readable parameters for each augmentation technique"""
    params = {
        "time_stretch": "Rate: 0.8-1.2x",
        "pitch_shift": "Steps: ±4 semitones",
        "add_noise": "Noise factor: 0.005",
        "speed_up": "Speed: 0.9-1.1x",
        "volume_adjust": "Volume: 0.5-1.5x",
        "reverb": "Reverb time: 100ms"
    }
    return params.get(technique, "")

# Add these new audio-specific functions to main.py

@app.get("/audio_analysis", response_class=JSONResponse)
async def get_audio_analysis():
    """Get detailed analysis of the audio file"""
    try:
        if audio_storage["original_audio"] is None:
            raise HTTPException(status_code=400, detail="No audio uploaded")
        
        audio = audio_storage["original_audio"]
        sr = audio_storage["original_sr"]
        
        # Basic audio statistics
        duration = len(audio) / sr
        max_amplitude = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Tempo and beats
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        return {
            "success": True,
            "analysis": {
                "duration": f"{duration:.2f}s",
                "sample_rate": f"{sr} Hz",
                "max_amplitude": f"{max_amplitude:.3f}",
                "rms_energy": f"{rms:.3f}",
                "estimated_tempo": f"{tempo:.1f} BPM",
                "avg_spectral_centroid": f"{np.mean(spectral_centroids):.1f} Hz",
                "avg_spectral_rolloff": f"{np.mean(spectral_rolloff):.1f} Hz"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/export_audio")
async def export_audio(request: Request):
    """Export processed audio in different formats"""
    try:
        if "preprocessed_audio" not in audio_storage:
            raise HTTPException(status_code=400, detail="No processed audio available")
        
        audio = audio_storage["preprocessed_audio"]
        sr = audio_storage["original_sr"]
        
        # Create exports in different formats
        exports = {}
        
        # WAV export
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sr, format='wav')
        wav_buffer.seek(0)
        exports["wav"] = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        # MP3 export (requires additional processing)
        mp3_buffer = io.BytesIO()
        sf.write(mp3_buffer, audio, sr, format='mp3')
        mp3_buffer.seek(0)
        exports["mp3"] = base64.b64encode(mp3_buffer.read()).decode('utf-8')
        
        return {
            "success": True,
            "exports": exports
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/audio_augment", response_class=HTMLResponse)
async def audio_augment_page(request: Request):
    if audio_storage["original_audio"] is None:
        return RedirectResponse(url="/audio_upload")
    
    return templates.TemplateResponse("audio_augment.html", {
        "request": request,
        "original_audio": encode_audio(audio_storage["original_audio"], 
                                     audio_storage["original_sr"]),
        "preprocessed_audio": encode_audio(audio_storage.get("preprocessed_audio", 
                                                           audio_storage["original_audio"]), 
                                         audio_storage["original_sr"])
    })

# 3D Routes
@app.get("/3d_upload", response_class=HTMLResponse)
async def model_upload_page(request: Request):
    return templates.TemplateResponse("3d_upload.html", {
        "request": request
    })

@app.post("/upload_3d")
async def upload_3d_model(file: UploadFile = File(...)):
    input_file = None
    output_file = None
    
    try:
        print(f"Receiving file: {file.filename}")
        suffix = Path(file.filename).suffix.lower()
        print(f"File extension: {suffix}")
        
        # Read content first
        content = await file.read()
        
        # Create input temporary file
        input_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        input_file.write(content)
        input_file.close()  # Close the file explicitly
        print(f"Temporary input file created at: {input_file.name}")

        # Load the model
        if suffix == '.off':
            try:
                print("Loading OFF file...")
                mesh = load_off_file(input_file.name)
                print("OFF file loaded successfully")
            except Exception as off_error:
                print(f"Error loading OFF file: {str(off_error)}")
                raise HTTPException(status_code=400, detail=str(off_error))
        else:
            mesh = trimesh.load(input_file.name)

        print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Store original model
        model_storage["original_model"] = mesh
        
        # Get model statistics
        stats = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "format": suffix[1:].upper()
        }

        # Export model for visualization
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
        output_file.close()  # Close immediately to avoid access issues
        
        # Export using the closed file's name
        mesh.export(output_file.name, file_type='obj')
        
        # Read the exported file
        with open(output_file.name, 'rb') as f:
            obj_data = f.read()

        return {
            "success": True,
            "statistics": stats,
            "message": f"Successfully loaded {suffix[1:].upper()} file",
            "model_data": base64.b64encode(obj_data).decode()
        }
        
    except Exception as e:
        print(f"Error in upload_3d_model: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    finally:
        # Clean up temporary files
        try:
            if input_file and Path(input_file.name).exists():
                Path(input_file.name).unlink()
                print(f"Cleaned up input file: {input_file.name}")
        except Exception as e:
            print(f"Warning: Could not delete input file: {e}")
            
        try:
            if output_file and Path(output_file.name).exists():
                Path(output_file.name).unlink()
                print(f"Cleaned up output file: {output_file.name}")
        except Exception as e:
            print(f"Warning: Could not delete output file: {e}")

@app.get("/3d_preprocess")
async def preprocess_page(request: Request):
    # Load the stored model data
    if "original_model" not in model_storage:
        raise HTTPException(status_code=400, detail="No model uploaded")
    
    mesh = model_storage["original_model"]
    
    # Convert to OBJ format for viewer
    obj_data = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as tmp:
            mesh.export(tmp.name, file_type='obj')
            tmp.close()  # Close the file before reading
            with open(tmp.name, 'r') as f:
                obj_data = f.read()
            try:
                os.unlink(tmp.name)  # Delete the temporary file
            except:
                pass  # Ignore deletion errors
    except Exception as e:
        print(f"Error preparing model data: {e}")
        raise HTTPException(status_code=500, detail="Error preparing model data")

    # Get model statistics
    stats = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bounds": mesh.bounds.tolist()
    }

    return templates.TemplateResponse(
        "3d_preprocess.html", 
        {
            "request": request,
            "model_data": base64.b64encode(obj_data.encode()).decode(),
            "stats": stats
        }
    )

@app.post("/apply_3d_preprocessing")
async def apply_preprocessing(steps: List[str] = Form(...)):
    if "original_model" not in model_storage:
        raise HTTPException(status_code=400, detail="No model uploaded")
    
    try:
        mesh = model_storage["original_model"].copy()
        results = {}
        
        for step in steps:
            step_mesh = mesh.copy()
            step_result = {"original_stats": {"vertices": len(step_mesh.vertices), "faces": len(step_mesh.faces)}}
            
            try:
                if step == "normalize_scale":
                    # Scale to unit cube
                    extents = step_mesh.extents
                    scale = 1.0 / max(extents)
                    step_mesh.apply_scale(scale)
                    step_result["changes"] = f"Scaled by factor {scale:.3f}"

                elif step == "center_model":
                    # Center at origin
                    center = step_mesh.centroid
                    step_mesh.apply_translation(-center)
                    step_result["changes"] = f"Centered at origin"

                elif step == "remove_duplicates":
                    # Remove duplicate vertices
                    original_vertices = len(step_mesh.vertices)
                    step_mesh.merge_vertices(merge_tex=True, merge_norm=True)
                    removed = original_vertices - len(step_mesh.vertices)
                    step_result["changes"] = f"Removed {removed} duplicate vertices"

                elif step == "fix_normals":
                    # Fix face winding and normals
                    step_mesh.fix_normals()
                    step_result["changes"] = "Fixed normals and face winding"

                elif step == "simplify_mesh":
                    # Use quadric decimation instead of vertex clustering
                    try:
                        original_faces = len(step_mesh.faces)
                        # Reduce to 50% of original faces
                        target_faces = int(original_faces * 0.5)
                        step_mesh = step_mesh.simplify_quadric_decimation(target_faces)
                        reduced = original_faces - len(step_mesh.faces)
                        step_result["changes"] = f"Reduced {reduced} faces"
                    except Exception as e:
                        # Fallback to basic decimation if quadric not available
                        original_faces = len(step_mesh.faces)
                        vertices = step_mesh.vertices
                        faces = step_mesh.faces
                        # Simple decimation: keep every other face
                        step_mesh.faces = faces[::2]
                        reduced = original_faces - len(step_mesh.faces)
                        step_result["changes"] = f"Reduced {reduced} faces (basic decimation)"

                elif step == "smooth_surface":
                    # Laplacian smoothing
                    vertices = step_mesh.vertices.copy()
                    for _ in range(3):
                        adjacency = step_mesh.vertex_adjacency_graph
                        for vertex_idx in range(len(vertices)):
                            neighbors = list(adjacency[vertex_idx])
                            if neighbors:
                                vertices[vertex_idx] = np.mean(step_mesh.vertices[neighbors], axis=0)
                    step_mesh.vertices = vertices
                    step_result["changes"] = "Applied surface smoothing"

                # Export step result
                with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as tmp:
                    step_mesh.export(tmp.name, file_type='obj')
                    tmp.close()
                    with open(tmp.name, 'r') as f:
                        obj_data = f.read()
                    os.unlink(tmp.name)

                step_result["model_data"] = base64.b64encode(obj_data.encode()).decode()
                step_result["processed_stats"] = {
                    "vertices": len(step_mesh.vertices),
                    "faces": len(step_mesh.faces)
                }
                
                # Apply the change to the main mesh
                mesh = step_mesh

            except Exception as step_error:
                print(f"Error in {step}: {str(step_error)}")
                step_result["error"] = str(step_error)

            results[step] = step_result

        # Export final result
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as tmp:
            mesh.export(tmp.name, file_type='obj')
            tmp.close()
            with open(tmp.name, 'r') as f:
                final_obj_data = f.read()
            os.unlink(tmp.name)

        # Store the preprocessed model
        model_storage["preprocessed_model"] = mesh

        return {
            "success": True,
            "stepwise_results": results,
            "final_model_data": base64.b64encode(final_obj_data.encode()).decode(),
            "final_statistics": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces)
            }
        }

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/3d_augment", response_class=HTMLResponse)
async def model_augment_page(request: Request):
    if model_storage["original_model"] is None:
        return RedirectResponse(url="/3d_upload")
    
    return templates.TemplateResponse("3d_augment.html", {
        "request": request
    })

@app.post("/apply_3d_augmentation")
async def apply_3d_augmentation(
    augmentations: List[str] = Form(..., alias="augmentations[]")
):
    if "original_model" not in model_storage or "preprocessed_model" not in model_storage:
        raise HTTPException(status_code=400, detail="Models not available")
    
    original_results = {}
    preprocessed_results = {}

    # Fixed parameters
    ROTATION_RANGE = 45.0
    SCALE_RANGE = 0.2
    NOISE_INTENSITY = 0.01
    TRANSLATION_RANGE = 0.3

    # Process original model
    original_mesh = model_storage["original_model"]
    for aug in augmentations:
        try:
            mesh_copy = original_mesh.copy()
            result = apply_augmentation(mesh_copy, aug, ROTATION_RANGE, SCALE_RANGE, 
                                     NOISE_INTENSITY, TRANSLATION_RANGE)
            original_results[aug] = result
        except Exception as e:
            print(f"Error in original model {aug}: {str(e)}")
            original_results[aug] = {"error": str(e)}

    # Process preprocessed model
    preprocessed_mesh = model_storage["preprocessed_model"]
    for aug in augmentations:
        try:
            mesh_copy = preprocessed_mesh.copy()
            result = apply_augmentation(mesh_copy, aug, ROTATION_RANGE, SCALE_RANGE, 
                                     NOISE_INTENSITY, TRANSLATION_RANGE)
            preprocessed_results[aug] = result
        except Exception as e:
            print(f"Error in preprocessed model {aug}: {str(e)}")
            preprocessed_results[aug] = {"error": str(e)}

    return {
        "success": True,
        "original_results": original_results,
        "preprocessed_results": preprocessed_results
    }

def apply_augmentation(mesh, aug_type, rotation_range, scale_range, noise_intensity, translation_range):
    result = {
        "original_stats": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces)
        }
    }

    if aug_type == "rotate":
        angle = np.random.uniform(-rotation_range, rotation_range)
        matrix = trimesh.transformations.rotation_matrix(
            angle=np.radians(angle),
            direction=[0, 1, 0],
            point=mesh.centroid
        )
        mesh.apply_transform(matrix)
        result["description"] = f"Rotated by {angle:.1f}°"

    elif aug_type == "scale":
        scale = 1.0 + np.random.uniform(-scale_range, scale_range)
        matrix = np.eye(4)
        matrix[:3, :3] *= scale
        mesh.apply_transform(matrix)
        result["description"] = f"Scaled by {scale:.2f}x"

    elif aug_type == "translate":
        translation = np.random.uniform(-translation_range, translation_range, 3)
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        mesh.apply_transform(matrix)
        result["description"] = f"Translated by {np.linalg.norm(translation):.2f} units"

    elif aug_type == "noise":
        noise = np.random.normal(0, noise_intensity, mesh.vertices.shape)
        mesh.vertices += noise
        result["description"] = f"Added noise (intensity: {noise_intensity:.3f})"

    elif aug_type == "smooth":
        vertices = mesh.vertices.copy()
        for _ in range(2):
            adjacency = mesh.vertex_adjacency_graph
            for vertex_idx in range(len(vertices)):
                neighbors = list(adjacency[vertex_idx])
                if neighbors:
                    vertices[vertex_idx] = np.mean(mesh.vertices[neighbors], axis=0)
        mesh.vertices = vertices
        result["description"] = "Applied smoothing"

    # Export the augmented model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.obj') as tmp:
        mesh.export(tmp.name, file_type='obj')
        tmp.close()  # Ensure the file is closed before reading
        with open(tmp.name, 'r', encoding='utf-8') as f:
            obj_data = f.read()
        os.unlink(tmp.name)

    result["model_data"] = base64.b64encode(obj_data.encode()).decode()
    result["processed_stats"] = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces)
    }

    return result

@app.get("/get_model_data")
async def get_model_data():
    if "original_model" not in model_storage:
        raise HTTPException(status_code=400, detail="No model uploaded")
    
    mesh = model_storage["original_model"]
    
    # Convert to OBJ format for viewer
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as tmp:
            mesh.export(tmp.name, file_type='obj')
            tmp.close()
            with open(tmp.name, 'r') as f:
                obj_data = f.read()
            os.unlink(tmp.name)
    except Exception as e:
        print(f"Error preparing model data: {e}")
        raise HTTPException(status_code=500, detail="Error preparing model data")

    return {
        "model_data": base64.b64encode(obj_data.encode()).decode(),
        "stats": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces)
        }
    }