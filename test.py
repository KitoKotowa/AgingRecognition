import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import time

from gan_module import Generator

@torch.no_grad()
def main():
    st.title('Aging Face Generator 👴 👵')
    st.sidebar.title("Face Aging Modeling using GAN")
    st.sidebar.markdown("### Member\n1) Trần Thanh Ngân - 21127115\n2) Châu Tấn Kiệt - 21127329\n3) Hồ Bạch Như Quỳnh - 21127412\n4) Ngô Thị Thanh Thảo - 21127433\n5)  Lê Nguyễn Phương Uyên - 21127476 ")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.subheader("Original Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption='Original Image', use_column_width=True)

        # Button to generate aged face
        if st.button("Generate"):
            # Display spinner while generating aged face
            with st.spinner('Generating aged face...'):
                time.sleep(1)  # Simulate a 2-second delay (replace this with actual model generation code)
                
                # Load the generator model
                model = Generator(ngf=32, n_residual_blocks=9)
                ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
                model.load_state_dict(ckpt)
                model.eval()
                
                # Preprocess the uploaded image
                image = Image.open(uploaded_file).convert('RGB')
                trans = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
                img_tensor = trans(image).unsqueeze(0)
                
                # Generate the aged face
                aged_face = model(img_tensor)
                
                # Convert the tensor to a numpy array and display the aged face
                aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
                st.subheader("Aged Face")
                st.image(aged_face, caption='Aged Face', use_column_width=True)

if __name__ == '__main__':
    main()
