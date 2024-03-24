{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f093f5c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Defaulting to user installation because normal site-packages is not writeable\n",
      "Requirement already satisfied: streamlit in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (1.32.2)\n",
      "Requirement already satisfied: flask in c:\\programdata\\anaconda3\\lib\\site-packages (2.2.2)\n",
      "Requirement already satisfied: scikit-learn in c:\\programdata\\anaconda3\\lib\\site-packages (1.2.2)\n",
      "Requirement already satisfied: pandas in c:\\programdata\\anaconda3\\lib\\site-packages (1.5.3)\n",
      "Requirement already satisfied: numpy in c:\\programdata\\anaconda3\\lib\\site-packages (1.24.3)\n",
      "Requirement already satisfied: tensorflow in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (2.16.1)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from streamlit) (5.2.0)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from streamlit) (1.7.0)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (8.0.4)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (23.0)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (9.4.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from streamlit) (4.25.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (11.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (2.29.0)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from streamlit) (13.7.1)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (4.6.3)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from streamlit) (3.1.42)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from streamlit) (0.8.1b0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (6.2)\n",
      "Requirement already satisfied: watchdog>=2.1.5 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (2.1.6)\n",
      "Requirement already satisfied: Werkzeug>=2.2.2 in c:\\programdata\\anaconda3\\lib\\site-packages (from flask) (2.2.3)\n",
      "Requirement already satisfied: Jinja2>=3.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from flask) (3.1.2)\n",
      "Requirement already satisfied: itsdangerous>=2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from flask) (2.0.1)\n",
      "Requirement already satisfied: scipy>=1.3.2 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn) (1.10.1)\n",
      "Requirement already satisfied: joblib>=1.1.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn) (1.2.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn) (2.2.0)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2022.7)\n",
      "Requirement already satisfied: tensorflow-intel==2.16.1 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow) (2.16.1)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.1.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=23.5.26 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (24.3.7)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.5.4)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: h5py>=3.10.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.10.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (18.1.1)\n",
      "Requirement already satisfied: ml-dtypes~=0.3.1 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.3.2)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.3.0)\n",
      "Requirement already satisfied: setuptools in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (67.8.0)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.4.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.62.1)\n",
      "Requirement already satisfied: tensorboard<2.17,>=2.16 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.16.2)\n",
      "Requirement already satisfied: keras>=3.0.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.1.0)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.31.0)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.17.3)\n",
      "Requirement already satisfied: toolz in c:\\programdata\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\programdata\\anaconda3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from Jinja2>=3.0->flask) (2.1.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (1.26.16)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2023.5.7)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.16.1->tensorflow) (0.38.4)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (22.1.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.18.0)\n",
      "Requirement already satisfied: namex in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.0.7)\n",
      "Requirement already satisfied: optree in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.10.0)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\hp\\appdata\\roaming\\python\\python311\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (0.7.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit flask scikit-learn pandas numpy tensorflow\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a4633b7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from flask import Flask, jsonify, request\n",
    "from tensorflow.keras.models import Sequential, load_model\n",
    "from tensorflow.keras.layers import Dense\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a99aac8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "iris = load_iris()\n",
    "X = iris.data\n",
    "y = iris.target\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "82f90672",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "dbec355d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "10e13bad",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\AppData\\Roaming\\Python\\Python311\\site-packages\\keras\\src\\layers\\core\\dense.py:88: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x2218e9b6c50>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = Sequential([\n",
    "    Dense(64, activation='relu', input_shape=(4,)),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dense(3, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3695a852",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "model.save('iris_model.h5')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ff5ab7d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    data = request.get_json(force=True)\n",
    "    predict_request = np.array(data['features'])\n",
    "    predict_request = predict_request.reshape(1, -1)\n",
    "    prediction = model.predict_classes(predict_request)\n",
    "    output = {'prediction': int(prediction[0])}\n",
    "    return jsonify(output)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "914233a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    st.title(\"Iris Classifier\")\n",
    "    sepal_length = st.slider(\"Sepal Length\", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))\n",
    "    sepal_width = st.slider(\"Sepal Width\", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))\n",
    "    petal_length = st.slider(\"Petal Length\", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))\n",
    "    petal_width = st.slider(\"Petal Width\", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))\n",
    "\n",
    "    features = [sepal_length, sepal_width, petal_length, petal_width]\n",
    "\n",
    "    if st.button(\"Predict\"):\n",
    "        response = requests.post(\"http://localhost:5000/predict\", json={\"features\": features})\n",
    "        prediction = response.json()['prediction']\n",
    "        species = iris.target_names[prediction]\n",
    "        st.write(f\"The predicted species is {species}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "43146361",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "INFO:werkzeug: * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    import sys\n",
    "    if 'ipykernel' in sys.modules:\n",
    "        # Running in IPython or Jupyter Notebook\n",
    "        app.run(port=5000)\n",
    "    else:\n",
    "        # Running in a standalone Python script\n",
    "        app.run(port=5000, debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6887df6c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
