import streamlit as st
from PIL import Image
import os
import pandas as pd
from model_helper import SwinV2Classifier
from huggingface_hub import hf_hub_download

# Configuration
CLASS_NAMES = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']
MODEL_PATH = hf_hub_download(
    repo_id="Neelkumar/car_damage_detection",
    filename="best_swinv2_small.pth"
)

# Multiple examples per class
EXAMPLE_IMAGES = {
    'F_Breakage': [
        "examples/FB_5.jpg",
        "examples/FB_30.jpg"
    ],
    'F_Crushed': [
        "examples/FC_16.jpg",
        "examples/FC_149.jpg"
    ],
    'F_Normal': [
        "examples/FN_83.jpg",
        "examples/FN_124.jpg"
    ],
    'R_Breakage': [
        "examples/RB_50.jpg",
        "examples/RB_227.jpg"
    ],
    'R_Crushed': [
        "examples/RC_182.jpg",
        "examples/RC_188.jpg"
    ],
    'R_Normal': [
        "examples/RN_28.jpg",
        "examples/RN_53.jpg"
    ]
}

@st.cache_resource
def load_model():
    return SwinV2Classifier(MODEL_PATH, CLASS_NAMES)

def display_results(image_path, classifier):
    """Display prediction results for an image"""
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(image_path)
        st.image(image, caption="Selected Image", use_container_width =False)
    
    with col2:
        result = classifier.predict(image_path)
        st.subheader("Prediction Results")
        st.metric("Predicted Class", result["class"])
        st.metric("Confidence", f"{result['confidence']:.2%}")
        
        # Detailed probabilities
        with st.expander("See class probabilities"):
            for name, prob in result["probabilities"].items():
                st.progress(prob, text=f"{name}: {prob:.2%}")

def main():
    st.title("Car Damage Classifier")
    classifier = load_model()
    
    # Layout tabs
    tab1, tab2 = st.tabs(["Test Single Image", "Batch Test Examples"])
    
    with tab1:
        st.subheader("Test Options")
        
        # Option 1: Upload
        uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])
        
        # Option 2: Select from examples
        selected_class = st.selectbox("Choose class examples:", ["Select"] + CLASS_NAMES)
        
        if selected_class != "Select":
            selected_image = st.radio(
                "Choose example:",
                EXAMPLE_IMAGES[selected_class],
                format_func=lambda x: os.path.basename(x)
            )
            display_results(selected_image, classifier)
        
        if uploaded_file:
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            display_results(temp_path, classifier)
            os.remove(temp_path)

    
    with tab2:
        st.subheader("Test All Examples")
        if st.button("Run Batch Test"):
            results = []
            for class_name, img_paths in EXAMPLE_IMAGES.items():
                for img_path in img_paths:
                    result = classifier.predict(img_path)
                    results.append({
                        "Image": os.path.basename(img_path),
                        "True Class": class_name,
                        "Predicted": result["class"],
                        "Confidence": f"{result['confidence']:.2%}",
                        "Correct": result["class"] == class_name
                    })
            
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Calculate accuracy
            accuracy = df["Correct"].mean()
            st.metric("Overall Accuracy", f"{accuracy:.2%}")



if __name__ == "__main__":
    main()
