from flask import Flask, render_template, request, send_file, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
from pathlib import Path
import sys

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 16MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import importlib.util
v1_models_path = Path(__file__).parent / 'v1' / 'models.py'
spec1 = importlib.util.spec_from_file_location("v1_models", v1_models_path)
v1_models = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(v1_models)
V1Model = v1_models.ConvAutoencoder

v2_models_path = Path(__file__).parent / 'v2' / 'models.py'
spec2 = importlib.util.spec_from_file_location("v2_models", v2_models_path)
v2_models = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(v2_models)
V2Model = v2_models.UNetDenoiser

# v3_models_path = Path(__file__).parent / 'v3' / 'models.py'
# spec3 = importlib.util.spec_from_file_location("v3_models", v3_models_path)
# v3_models = importlib.util.module_from_spec(spec3)
# spec3.loader.exec_module(v3_models)
# V3Model = v3_models.ImprovedAutoencoder

MODEL_CONFIGS = {
    'v1': {
        'model_class': V1Model,
        'model_path': 'v1/denoising_model.pth',
        'dropout_rate': 0.1,
        'name': 'ConvAutoencoder'
    },
    'v2': {
        'model_class': V2Model,
        'model_path': 'v2/denoising_model.pth',
        'dropout_rate': 0.05,
        'name': 'UNetDenoiser'
    },
    # 'v3': {
    #     'model_class': V3Model,
    #     'model_path': 'v3/denoising_model.pth',
    #     'init_args': {
    #         'latent_dim': 256,
    #         'dropout_rate': 0.05
    #     },
    #     'name': 'ImprovedAutoencoder'
    # },
}

def load_model(version):
    """Load model for given version"""
    if version not in MODEL_CONFIGS:
        raise ValueError(f"Unknown version: {version}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[version]
    model_path = Path(config['model_path'])
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    init_args = config.get('init_args')
    if init_args is not None:
        model = config['model_class'](**init_args).to(device)
    else:
        model = config['model_class'](dropout_rate=config.get('dropout_rate', 0.0)).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

models = {}
for version in MODEL_CONFIGS.keys():
    try:
        models[version] = load_model(version)
        print(f"✓ Loaded {MODEL_CONFIGS[version]['name']} ({version})")
    except FileNotFoundError as e:
        print(f"⚠️  Model file not found for {version}: {e}")
        print(f"   Make sure to train the model first or check the path in MODEL_CONFIGS")
    except Exception as e:
        print(f"⚠️  Could not load {version}: {e}")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def denoise_image(image_path, version='v2'):
    """Process image through denoising model"""
    if version not in models:
        raise ValueError(f"Model {version} not loaded")
    model = models[version]
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_tensor = model(img_tensor)
    denoised_img = transforms.ToPILImage()(denoised_tensor.squeeze(0).cpu())
    denoised_img = denoised_img.resize(original_size, Image.LANCZOS)
    return img, denoised_img

@app.route('/')
def index():
    """Main page with upload form"""
    available_versions = list(models.keys())
    return render_template('index.html', versions=available_versions)

@app.route('/<version>/denoise', methods=['GET', 'POST'])
def denoise_page(version):
    """Web page endpoint for denoising (returns HTML)"""
    if version not in models:
        return f"Version {version} not available. Available versions: {list(models.keys())}", 404
    
    if request.method == 'GET':
        return render_template('denoise.html', version=version, versions=list(models.keys()))
    
    # POST - process image
    if 'file' not in request.files:
        return render_template('denoise.html', version=version, versions=list(models.keys()), 
                             error='Brak pliku')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('denoise.html', version=version, versions=list(models.keys()), 
                             error='Nie wybrano pliku')
    
    try:
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        original_img, denoised_img = denoise_image(filepath, version)
        
        # Save results
        original_path = os.path.join('static/results', f'original_{version}_{filename}')
        denoised_path = os.path.join('static/results', f'denoised_{version}_{filename}')
        
        original_img.save(original_path)
        denoised_img.save(denoised_path)
        
        return render_template('result.html',
                             version=version,
                             versions=list(models.keys()),
                             original_image=f'/static/results/original_{version}_{filename}',
                             denoised_image=f'/static/results/denoised_{version}_{filename}')
    except Exception as e:
        return render_template('denoise.html', version=version, versions=list(models.keys()), 
                             error=f'Błąd przetwarzania: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
