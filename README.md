#### Face Generation with Variational Autoencoder (VAE)

#### Overview
This project uses a Variational Autoencoder (VAE) to generate human faces based on the CelebA dataset. A VAE is a generative model that learns to represent high-dimensional data (like images) in a lower-dimensional latent space, and then generates new data from this space.

#### Installation
To run this project, you need Python 3.8 and predominantly the following packages installed:
- torch
- torchvision
- matplotlib
- streamlit

Install the necessary packages using the following command:

```
pip install -r requirements.txt
```

Or the following:

```
pip install torch torchvision matplotlib streamlit
```

#### Dataset
The CelebA dataset is used, which contains over 200,000 celebrity images with 40 attribute annotations. Ensure the dataset is downloaded and placed in the `/data/img_align_celeba/` directory for the script to access.
Location of the dataset: https://drive.google.com/file/d/1rxGTimhTS34eiamhldhkoJrDoVmYps5P/view?usp=sharing

#### Files in the Project
- `vae_model.py`: Contains the VAE implementation.
- `train_vae.py`: Script to train the VAE on the CelebA dataset.
- `app.py`: Streamlit application to display generated images from the trained VAE model.

#### How to Run
1. Train the VAE model by running `train_vae.py`. This will save the model checkpoints and generate images in the `runs/` directory.
2. Start the Streamlit application with `streamlit run app.py`. Load the trained model and generate new images through the web interface.

#### Usage
- Modify `train_vae.py` to change training parameters, model architecture, or dataset processing.
- Use `app.py` to visualize the generated images and interact with the trained model.
