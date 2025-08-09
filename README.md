# Risk Flag Prediction: An MLOps End-to-End Project

<img src="risk-flag-image.jpeg" alt="Project Banner" width="100%">

![Step 1](https://img.shields.io/badge/Step%201-Data%20Loading-blue)
![Step 2](https://img.shields.io/badge/Step%202-Preprocessing-orange)
![Step 3](https://img.shields.io/badge/Step%203-Experiment%20Tracking-yellow)
![Step 4](https://img.shields.io/badge/Step%204-Model%20Training-green)
![Step 5](https://img.shields.io/badge/Step%205-CI%2FCD%20Pipeline-lightgrey)
![Step 6](https://img.shields.io/badge/Step%206-Deployment-purple)
![Step 7](https://img.shields.io/badge/Step%207-Monitoring-red)

## 📜 Introduction

This project is an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict credit risk. The goal is to build, deploy, and monitor a machine learning model that classifies loan applicants as "High Risk" or "Low Risk" based on their financial and demographic data like cities, states.

The primary objective is to classify loan applicants as **High Risk** or **Low Risk** based on their financial and demographic data (e.g., cities, states).  
It demonstrates **MLOps best practices** such as:

- Automated CI/CD pipelines with **GitHub Actions**
- **Experiment tracking** and model registry via MLflow
- Containerization with **Docker**
- Deployment to AWS **(ECR + EC2)**
- Continuous monitoring and performance tracking using **grafana**

## 📈 Project Workflow & Architecture

```
.
├── .dvc
├── .github
│   └── workflows
│       └── cicd.yaml
├── artifacts
│   ├── city_global_mean.joblib
│   ├── city_mapping.joblib
│   ├── state_global_mean.joblib
│   └── state_mapping.joblib
├── data
│   ├── processed
│   │   ├── test_processed.csv
│   │   └── train_processed.csv
│   └── raw
│       ├── test.csv
│       └── train.csv
├── docs
├── flask_api
│   ├── main.py
│   └── templates
│       └── index.html
├── metrics
│   └── validation_metrics.json
├── models
│   └── lightgbm_model.joblib
├── notebooks
├── plots
│   └── confusion_matrix_validation.png
├── predictions
│   └── test_predictions.csv
├── reports
├── src
│   ├── data
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   └── modeling
│       ├── model_building.py
│       ├── model_evaluation.py
│       ├── model_prediction.py
│       ├── model_register.py
│       ├── predict.py
│       ├── train.py
│       └── __init__.py
├── .dvcignore
├── .env
├── .gitattributes
├── .gitignore
├── Dockerfile
├── dvc.lock
├── dvc.yaml
├── ingestion_errors.log
├── Makefile
├── mlflow_run_info.json
├── model_building.log
├── model_evaluation.log
├── model_prediction.log
├── model_registration_errors.log
├── params.yaml
├── Pipfile
├── Pipfile.lock
├── preprocessing.log
├── preprocessing_errors.log
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.py
```

The project follows a standard MLOps lifecycle, as depicted in the workflow below:

1.  **Model Development:** The data preprocessing and model training logic are defined in a Python script...
2.  **Experiment Tracking (MLflow):** Every model training run is tracked with MLflow to log parameters, metrics, and the model artifact.
3.  **Model Registry (MLflow):** The best-performing model is registered in the MLflow Model Registry, where it is versioned and managed.
4.  **CI/CD Trigger:** A push to the `main` branch of the GitHub repository triggers an automated CI/CD pipeline via GitHub Actions...
5.  **Deployment (AWS):** The pipeline builds a Docker image of the Flask API, pushes it to AWS ECR, and deploys it to a running AWS EC2 instance.
6.  **Real-Time Inference:** The deployed Flask API serves as a real-time inference endpoint, accepting user data and returning predictions.
7.  **Model Monitoring:** The architecture supports continuous monitoring of the deployed model, providing insights into its performance and data drift.

```
+----------------+       +-------------------+       +-----------------+
| Data Prep      | ----> | MLflow Tracking   | ----> | MLflow Registry |
| & Training     |       | (Parameters,      |       | (Versioned Model)|
| (Python Script)|       | Metrics, Artifacts)|       +-----------------+
+----------------+       +-------------------+               |
                                                               |
(git push to main)                                             |
       |                                                       |
       V                                                       |
+----------------+       +------------------+                  V
| GitHub Actions | ----> | Docker Build &   | ---->   +------------------+
| (CI/CD)        |       | Push to AWS ECR  |         | AWS EC2 Instance |
|                |       |                  |         | (Flask API)      |
+----------------+       +------------------+         +------------------+
                                                               |
                                                               V
                                                     +------------------+
                                                     | Real-Time        |
                                                     | Inference &      |
                                                     | Monitoring       |
                                                     +------------------+
```

## Getting Started

### Prerequisites

- Python 3.8+
- `pipenv` package manager. You can install it using `pip install pipenv`.
- [Docker](https://docs.docker.com/get-docker/) installed locally and on your AWS EC2 instance.
- An AWS account with an EC2 instance, ECR repository, and IAM credentials configured.
- An SSH key for connecting to your EC2 instance.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/gujeah/Risk-Flag-Prediction.git
    cd Risk-Flag-Prediction
    ```

2.  **Install dependencies using Pipenv:**

    ```bash
    pipenv install
    ```

    This command will read the `Pipfile` and `Pipfile.lock` to create a virtual environment and install all necessary packages.

3.  **Activate the virtual environment:**
    ```bash
    pipenv shell
    ```

### Set Up AWS & GitHub Secrets

configure the necessary secrets in GitHub repository. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`, `EC2_HOST`, `EC2_USERNAME`, and `SSH_PRIVATE_KEY`.

### Local Testing with Docker

To ensure your application works correctly before a full deployment, you can build and run it locally using Docker.

1.  **Build the Docker image:**

    ```bash
    docker build -t risk-flag-app .
    ```

2.  **Run the container:**

    ```bash
    docker run -d -p 5000:5000 --name my-risk-app risk-flag-app
    ```

3.  **Deployment:**

    - Flask app was deployed on AWS platform

4.  **Tools & Technologies**

    - Machine Learning: LightGBM, Pandas, Scikit-learn

    - Experiment Tracking: MLflow

    - Containerization: Docker

    - CI/CD: GitHub Actions

    - Cloud: AWS ECR, AWS EC2

    - API Framework: Flask
