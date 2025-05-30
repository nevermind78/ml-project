# ML Project - Spring 2025

## 📝 Overview

This project implements a machine learning pipeline with a **Flask API** deployed on **Render** and a **Streamlit app** for user interaction. The project is fully automated using GitHub Actions.

---

## 🚀 Deployed Applications

- **API on Render**: [https://ml-project-api.onrender.com](https://ml-project-api.onrender.com)
- **Streamlit App**: [https://ml-project25.streamlit.app](https://ml-project25.streamlit.app)

---

## 🗂 Project Structure

```file.txt
your-repo/
├── .github/
│ └── workflows/
│ └── ci_cd.yml 
├── api/ 
│ ├── api.py 
│ ├── requirements.txt 
│ └── runtime.txt 
├── app/ 
│ └── app.py 
├── models/ 
│ ├── logistic_regression.pkl 
│ ├── linear_svc.pkl 
│ └── knn.pkl 
├── train.py 
├── requirements.txt 
├── README.md 
└── .gitignore 

```

---

## 🛠 Setup and Installation

### Prerequisites

- Python 3.9
- Git
- Accounts on [GitHub](https://github.com), [Render](https://render.com), and [Streamlit Sharing](https://share.streamlit.io)

### Local Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Train the models:

   ```bash
   python train.py
   ```
5. Run the API locally:

   ```bash
   cd api
   python api.py
   ```
6. Run the Streamlit app locally:

   ```
   streamlit run app/app.py
   ```

---

## 🚀 Deployment

### 1. Deploy the API on Render

1. Go to [Render](https://render.com/) and create a new  **Web Service** .
2. Connect your GitHub repository.
3. Configure the service:
   * **Build Command** : `pip install -r requirements.txt`
   * **Start Command** : `python api.py`
4. Deploy the service.

### 2. Deploy the Streamlit App on Streamlit Sharing

1. Go to [Streamlit Sharing](https://share.streamlit.io/).
2. Connect your GitHub repository.
3. Specify the path to your Streamlit file (`app/app.py`).
4. Deploy the app.

---

## 🤖 GitHub Actions Workflow

The GitHub Actions workflow automates the following steps:

1. **Train Models** : Runs `train.py` to generate `.pkl` files.
2. **Deploy API** : Deploys the API to Render.
3. **Add Models to Repository** : Adds the `.pkl` files to the Git repository.

### `.github/workflows/ci_cd.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  train-models:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Create models directory
        run: |
          mkdir -p models

      - name: Train and save models
        run: |
          python train_models.py

      - name: Add models to Git
        run: |
          git config --global user.name "GitHub Actions"
          git config --global user.email "actions@github.com"
          git add models/logistic_regression.pkl models/linear_svc.pkl models/knn.pkl
          git commit -m "Update models via GitHub Actions"
          git push https://${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git main

  deploy-api:
    needs: train-models
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r api/requirements.txt

      - name: Deploy to Render using API
        run: |
          pip install requests
          SERVICE_ID="${{ secrets.RENDER_SERVICE_ID }}"
          API_TOKEN="${{ secrets.RENDER_TOKEN }}"
          curl -s -X POST \
            -H "Authorization: Bearer $API_TOKEN" \
            -H "Accept: application/json" \
            -H "Content-Type: application/json" \
            -d '{
                  "clearCache": "do_not_clear"
                }' \
            "https://api.render.com/v1/services/$SERVICE_ID/deploys"
```

---

## 📄 Additional Documentation

### Key Files

* **`train.py`** : Script to train and save models.
* **`api/api.py`** : Flask API code.
* **`app/app.py`** : Streamlit app code.
* **`requirements.txt`** : Dependencies for the local environment.
* **`api/requirements.txt`** : Dependencies for the API.

---

## 🙏 Acknowledgments

* **Render** for hosting the API.
* **Streamlit Sharing** for hosting the Streamlit app.
* **GitHub Actions** for automating the CI/CD pipeline.

---

## 📧 Contact

For questions or feedback, contact me at [your-email@example.com](https://mailto:your-email@example.com/).

---

### **How to Use This File**

1. Copy this content into a file named `README.md` in the root of your repository.
2. Replace placeholders (e.g., `your-username`, `your-repo`, `your-api-url.onrender.com`) with the appropriate values.
3. Push the file to your GitHub repository:
   ```bash
   git add README.md
   git commit -m "Add final project documentation"
   git push origin main
   ```

```

```
