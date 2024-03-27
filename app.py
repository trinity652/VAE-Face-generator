import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from vae_model import VAE  # Import your VAE class

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_images(model, num_images=64):
    with torch.no_grad():
        z = torch.randn(num_images, 64)  # Adjust 64 to your latent size
        samples = model.decoder(z)
    return samples

def display_images(images):
    grid = make_grid(images, nrow=8)  # Adjust the nrow for how you want to display
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    st.pyplot(plt)

def main():
    st.title('VAE Generated Images')

    model_path = st.text_input('Enter model path:', 'runs/checkpoints/epoch_10.pth')
    if model_path:
        model = load_model(model_path)
        st.success('Model loaded successfully!')

        if st.button('Generate Images'):
            images = generate_images(model)
            display_images(images)

if __name__ == '__main__':
    main()
