import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Configura il dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carica il modello salvato
@st.cache_resource  # Cache per evitare di ricaricare ogni volta
def load_model():
    model = models.resnet18(weights=None)  # Carichiamo il modello senza pesi pre-addestrati
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Modifica l'output a 2 classi
    model.load_state_dict(torch.load('pneumonia_classifier.pth', map_location=device))
    model = model.to(device)
    model.eval()  # Imposta il modello in modalità di valutazione
    return model

model = load_model()

# Trasformazioni per preprocessare l'immagine
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambia la dimensione
    transforms.ToTensor(),  # Converte in tensore
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza
])

# Funzione per fare una predizione
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Carica e converte in RGB
    image_tensor = transform(image).unsqueeze(0).to(device)  # Preprocessa e aggiungi batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Ottieni l'indice della classe con la probabilità più alta
    classes = ['NORMAL', 'PNEUMONIA']  # Etichette delle classi
    return classes[predicted.item()]

# Interfaccia Streamlit
st.title("Classificatore di Pneumonia")
st.write("Carica un'immagine del torace per vedere se è normale o affetta da polmonite.")

# Carica immagine
uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostra l'immagine caricata
    image = Image.open(uploaded_file)
    st.image(image, caption="Immagine caricata", use_column_width=True)
    
    # Predici la classe
    with st.spinner("Sto classificando l'immagine..."):
        prediction = predict_image(uploaded_file)
    
    # Mostra il risultato
    st.success(f"L'immagine è classificata come: **{prediction}**")
