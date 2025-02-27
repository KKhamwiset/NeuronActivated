# 🧠 Neuron Activated
A sophisticated machine learning web application built with Streamlit that empowers users to upload data, train custom ML models, and generate accurate predictions with minimal effort.

## ✨Features

- **Data Upload**: Support for CSV and Excel files
- **Data Exploration**: Preview data and get basic statistics
- **Feature Importance**: Visualize which features impact the model most
- **Predictions**: Make new predictions using the trained model

## 🏗️Project Structure

```
├── .devcontainer/              # Development container configuration
├── .github/workflows/          # CI/CD pipeline configurations
│   └── ci_test.yml             # Automated testing workflow
├── .streamlit/                 # Streamlit configuration files
├── assets/                     # Static assets
│   └── ml_prepare/             # ML preprocessing utilities
├── components/                 # Reusable UI components
│   ├── main_style.py           # Main styling configurations
│   └── sidebar.py              # Sidebar component
├── data/                       # Sample and user data storage
│   ├── income/                 # Income prediction dataset
│   │   ├── adult.data          # Training data
│   │   ├── adult.names         # Feature descriptions
│   │   ├── adult.test          # Test data
│   │   ├── index               # Data index
│   │   ├── old.adult.names     # Legacy feature descriptions
│   │   └── new_folder/         # Additional datasets
│   └── env_tensor/             # Environment tensor data
├── model_training/             # Model training scripts
├── pages/                      # Multi-page app views
├── app.py                      # Main Streamlit application
├── environment.yml             # Conda environment specification
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

## 🚀Installation

1. Clone this repository:
```bash
git clone https://github.com/KKhamwiset/NeuronActivated
cd your folder
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🎮Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Use the application:
   - Upload your dataset (CSV or Excel)
   - Explore your data
   - Select target and feature columns
   - Make predictions

## 🛠️Dependencies

- streamlit
- tensorflow
- pandas
- numpy
- scikit-learn
- matplotlib

## 🔄 CI/CD Pipeline
This project uses GitHub Actions for continuous integration and deployment. The pipeline automatically tests the application on each push to ensure functionality.

## Keeping the App Active

To prevent the app from going to sleep on hosting platforms, a keep-alive mechanism is included. You can use services like [uptimerobot](https://uptimerobot.com/) to ping your app regularly.

🙏 Acknowledgments

Special thanks to all the open-source libraries that made this possible
Inspired by the need for accessible machine learning tools