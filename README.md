﻿# 🧠 Neuron Activated
Note : Dataset for CNN can be reach via Reference Page or [Fruit 10 classes](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data)


## ✨Features

- **Predictions**: Make new predictions using the trained model



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

3. Use the application

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
