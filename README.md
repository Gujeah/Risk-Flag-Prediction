# Risk Flag Prediction: An MLOps End-to-End Project- Am still finalising it to AWS, dear reviewr, give me 5 hours i will be done as soon as possible.

## 📜 Introduction

This project is an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict credit risk. The goal is to build, deploy, and monitor a machine learning model that classifies loan applicants as "High Risk" or "Low Risk" based on their financial and demographic data like cities, states.

This project serves as a practical demonstration of key MLOps principles, from experiment tracking and model registration to automated deployment and continuous monitoring, using industry-standard tools like MLflow, Docker, and GitHub Actions.

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

1.  **Model Development:** The data preprocessing and model training logic are defined in a Python script.
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

3.  **Access the web interface:**
    - Open your web browser and navigate to `http://localhost:5000`.
