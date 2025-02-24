# ML Model App

A lightweight machine learning web application built with Streamlit that allows users to upload data, train a machine learning model, and make predictions.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Data Exploration**: Preview data and get basic statistics
- **Feature Importance**: Visualize which features impact the model most
- **Predictions**: Make new predictions using the trained model

## Project Structure

```
project_folder/
  ├── app.py                  # Main Streamlit application
  ├── components/             # Reusable UI components
  │   ├── __init__.py
  │   └── card.py             # Custom card component
  ├── requirements.txt        # Project dependencies
  └── README.md               # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/KKhamwiset/IS_Project
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

## Usage

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

## Dependencies

- streamlit
- tensorflow
- pandas
- numpy
- scikit-learn
- matplotlib

## Keeping the App Active

To prevent the app from going to sleep on hosting platforms, a keep-alive mechanism is included. You can use services like [uptimerobot](https://uptimerobot.com/) to ping your app regularly.
