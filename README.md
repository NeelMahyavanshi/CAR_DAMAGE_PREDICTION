# CAR_DAMAGE_PREDICTION

An AI model that can classify images into 6 classes representing different types of car damage.

## Key Features & Benefits

*   **Image Classification:** Accurately classifies car damage into six distinct categories: F_Breakage, F_Crushed, F_Normal, R_Breakage, R_Crushed, and R_Normal.
*   **SwinV2 Architecture:** Leverages the SwinV2 transformer model for robust and efficient image processing.
*   **Easy to Use:**  Deployed as a Streamlit application for a user-friendly experience.
*   **Accessibility:**  Pre-trained model available on Hugging Face Hub for easy download and use.

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

*   **Python:** 3.7+
*   **pip:** Python package installer
*   **Git:** For cloning the repository.

The following Python libraries are required and listed in `requirements.txt`:

```
streamlit
torch>=2.0.0
torchvision
torchaudio
transformers>=4.40.0
accelerate
Pillow
matplotlib
```

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/NeelMahyavanshi/CAR_DAMAGE_PREDICTION.git
    cd CAR_DAMAGE_PREDICTION
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Pre-trained Model**
    The application automatically downloads the model from Hugging Face Hub, but ensure you have an internet connection for the first run.

## Usage Examples & API Documentation

To run the Streamlit application:

1.  **Navigate to the `streamlit_app` directory:**

    ```bash
    cd streamlit_app
    ```

2.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

This will launch the application in your web browser. You can upload images and get predictions from the model.

**Code Snippet (example from app.py):**

```python
import streamlit as st
from PIL import Image
from model_helper import SwinV2Classifier
from huggingface_hub import hf_hub_download

# Configuration
CLASS_NAMES = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']
MODEL_PATH = hf_hub_download(
    repo_id="Neelkumar/car_damage_detection",
    filename="best_swinv2_small.pth"
)

# Load model
classifier = SwinV2Classifier(MODEL_PATH, CLASS_NAMES)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, probability = classifier.predict(image)
    st.write(f"Prediction: {label} with probability: {probability:.4f}")
```

## Configuration Options

*   **`MODEL_PATH` (in `streamlit_app/app.py`):** Specifies the path to the pre-trained model. By default, it downloads from Hugging Face Hub. You can modify it to use a local file if needed.
*   **`CLASS_NAMES` (in `streamlit_app/app.py`):** List of class names used for the classification.
*   The device (`CPU` or `GPU`) is automatically selected within the `model_helper.py` depending on availability.

## Contributing Guidelines

We welcome contributions! Here's how you can contribute:

1.  **Fork the repository.**
2.  **Create a new branch:** `git checkout -b feature/your-feature`
3.  **Make your changes and commit them:** `git commit -am 'Add some feature'`
4.  **Push to the branch:** `git push origin feature/your-feature`
5.  **Create a pull request.**

Please ensure your code adheres to the project's style guidelines and includes appropriate tests.

## License Information

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   This project utilizes the SwinV2 model architecture from Hugging Face Transformers.
*   Streamlit is used for creating the user interface.
