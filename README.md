# Risk Flag Prediction: An MLOps End-to-End Project

## 📜 Introduction

This project is an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict credit risk. The goal is to build, deploy, and monitor a machine learning model that classifies loan applicants as "High Risk" or "Low Risk" based on their financial and demographic data.

This project serves as a practical demonstration of key MLOps principles, from experiment tracking and model registration to automated deployment and continuous monitoring, using industry-standard tools like MLflow, Docker, and GitHub Actions.

## 📈 Project Workflow & Architecture

The project follows a standard MLOps lifecycle, as depicted in the workflow below:

1.  **Model Development:** The data preprocessing and model training logic are defined in a Python script.
2.  **Experiment Tracking (MLflow):** Every model training run is tracked with MLflow to log parameters, metrics, and the model artifact.
3.  **Model Registry (MLflow):** The best-performing model is registered in the MLflow Model Registry, where it is versioned and managed.
4.  **CI/CD Trigger:** A push to the `main` branch of the GitHub repository triggers an automated CI/CD pipeline via GitHub Actions.
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
