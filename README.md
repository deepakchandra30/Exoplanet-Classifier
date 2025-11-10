# NCI-ComV: Exoplanet Classifier

**Project for the NASA Space Apps Hackathon — Challenge: "A World Away: Hunting for Exoplanets with AI"**

## What is this?

This project is an **AI-powered exoplanet classification tool** created for the NASA Space Apps Hackathon challenge, *A World Away*. Its goal is to help scientists and astronomy enthusiasts quickly and accurately classify exoplanet candidates using modern machine learning methods on real data from NASA's exoplanet archive.

## How to Use

1. **Clone the Repository**
    ```bash
    git clone https://github.com/Epochdev0/NCI-ComV.git
    cd NCI-ComV
    ```

2. **Set Up the Environment**
    - Create a virtual environment (recommended):
        ```bash
        python -m venv venv
        source venv/bin/activate         # macOS / Linux
        # OR
        .\venv\Scripts\activate          # Windows (PowerShell)
        ```
    - Install required dependencies:
        ```bash
        pip install -r requirements.txt
        ```

3. **Quick Test or Training**
    - Explore the ready-to-run Jupyter notebook (`exoplanet_quick_test_presets.ipynb`) to test the model and presets.
    - See `train_xgboost_model.py` for training a custom model or experimenting with other settings.

4. **Included Data**
    - The repository includes a sample NASA exoplanet CSV (`nasa_exoplanet_cumulative.csv`) for exploration and testing.
    - For your own experiments, update this file with new datasets as needed.

## Technologies Involved

- **Python**: Main programming language
- **Machine Learning**: XGBoost model for classification
- **Jupyter Notebook**: For interactive testing and experimentation
- **Shell scripting**: For setup and automation
- **Data**: NASA exoplanet archive (provided as CSV)

---
