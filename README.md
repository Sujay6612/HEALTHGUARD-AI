# HealthGuard AI

### Machine Learning-Based Heart Disease Risk Prediction & Patient Management System

HealthGuard AI is a full-stack healthcare web application that combines machine learning with secure user authentication, patient record management, and health analytics to support early heart disease risk assessment.

Unlike a standalone prediction model, the application provides an end-to-end workflow where users can securely register, receive personalized heart disease risk predictions, track their historical health records, compare prediction trends across multiple visits, and receive tailored health recommendations. An integrated administrative dashboard enables patient management, analytics, CSV export, and system administration.

The project demonstrates the integration of machine learning, backend development, database management, and web technologies into a single production-style application.

## Features

### Machine Learning

- Predicts heart disease risk using supervised machine learning.
- Automatically evaluates multiple classification algorithms and selects the best-performing model using GridSearchCV.
- Stores trained model artifacts and metadata for reproducibility.
- Provides prediction probability and confidence score.

### User Management

- Secure user registration and login.
- Passwords stored using secure hashing (Werkzeug `scrypt`).
- Session-based authentication and authorization.

### Patient Management

- Maintains prediction history for every registered patient.
- Allows users to track previous health assessments.
- Compares historical predictions to identify health trends over time.
- Generates personalized health recommendations based on prediction results.

### Administrative Dashboard

- Secure administrator login.
- View and search patient records.
- Export patient data as CSV.
- View application analytics and high-risk alerts.
- Manage administrator credentials.

### Database

- SQLite-based persistent storage.
- Automatic database initialization on first startup.
- Automatically creates the default administrator account if one does not exist.

### Web Application

- Flask-based backend.
- Responsive HTML templates using Jinja2.
- Organized project structure separating frontend, backend, machine learning, and data layers.

## System Architecture

HealthGuard AI follows a modular architecture that integrates a web interface, machine learning model, and database into a single end-to-end healthcare application.

```text
                    User
                      │
                      ▼
             Flask Web Application
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
 Authentication   ML Prediction   SQLite Database
        │             │             │
        └─────────────┼─────────────┘
                      ▼
            Personalized Dashboard
```

### Architecture Components

**Frontend**

- HTML templates rendered using Jinja2.
- Responsive interface for patient and administrator workflows.

**Backend**

- Flask handles routing, authentication, session management, database operations, and prediction requests.

**Machine Learning Layer**

- Loads the trained model, preprocessing artifacts, and metadata.
- Processes user inputs and generates prediction probabilities along with personalized health recommendations.

**Database Layer**

- SQLite stores user accounts, prediction history, and administrator settings.
- The application automatically initializes the database and seeds the default administrator account during first startup.

## Machine Learning Pipeline

The machine learning pipeline is designed to automate model training, evaluation, and selection while maintaining reproducibility through saved model metadata.

### Workflow

```text
Clinical Dataset
        │
        ▼
Data Loading
        │
        ▼
Feature Mapping & Validation
        │
        ▼
Missing Value Handling
        │
        ▼
Train-Test Split
        │
        ▼
Candidate Model Training
(Logistic Regression, Random Forest,
Gradient Boosting, Extra Trees,
HistGradientBoosting)
        │
        ▼
Hyperparameter Tuning
(GridSearchCV with 5-Fold Cross Validation)
        │
        ▼
Model Evaluation
(Accuracy, Precision, Recall,
F1-Score, ROC-AUC)
        │
        ▼
Best Model Selection
        │
        ▼
Save Model Artifacts
(model.pkl, scaler.pkl, model_metadata.json)
```

### Training Strategy

The training pipeline evaluates multiple supervised machine learning algorithms instead of relying on a single predefined model. Each candidate model undergoes hyperparameter tuning using **GridSearchCV** with **5-fold cross-validation**, and the best-performing model is automatically selected based on evaluation metrics.

### Evaluation Metrics

The selected model is evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

Using multiple evaluation metrics provides a more reliable assessment of model performance, particularly for healthcare classification tasks where accuracy alone may not be sufficient.

### Model Artifacts

After training, the application stores:

- **model.pkl** – Trained machine learning model
- **scaler.pkl** – Feature preprocessing artifact
- **model_metadata.json** – Training metadata, evaluation metrics, selected model information, dataset details, and feature configuration

This separation allows the Flask application to load trained artifacts directly without retraining the model during inference.

## Technology Stack

| Category                  | Technologies                         |
| ------------------------- | ------------------------------------ |
| **Programming Language**  | Python 3                             |
| **Backend Framework**     | Flask                                |
| **Machine Learning**      | Scikit-learn                         |
| **Data Processing**       | Pandas, NumPy                        |
| **Database**              | SQLite                               |
| **Frontend**              | HTML5, CSS3, Jinja2                  |
| **Authentication**        | Werkzeug Security (Password Hashing) |
| **Model Persistence**     | Pickle                               |
| **Hyperparameter Tuning** | GridSearchCV                         |
| **Version Control**       | Git, GitHub                          |

## Project Structure

```text
HEALTHGUARD-AI/
│
├── data/                  # Training dataset
├── models/                # Trained model, preprocessing artifacts, and metadata
├── screenshots/           # Application screenshots for documentation
├── static/                # CSS, images, and static assets
├── templates/             # HTML templates rendered by Flask
│
├── app.py                 # Main Flask application
├── train_model.py         # Model training and evaluation pipeline
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── setup.py               # Package configuration
├── pyproject.toml         # Python project configuration
├── MANIFEST.in            # Package manifest
├── .gitignore             # Git ignore rules
└── heartiq_records.db     # SQLite database
```

### Folder Overview

| Folder / File          | Purpose                                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **data/**              | Stores the heart disease dataset used for training the machine learning model.                                        |
| **models/**            | Contains the trained model, preprocessing artifacts, and training metadata.                                           |
| **templates/**         | HTML templates for authentication, prediction, patient history, and administrator pages.                              |
| **static/**            | Static assets including CSS and images.                                                                               |
| **screenshots/**       | Screenshots used in the project documentation.                                                                        |
| **app.py**             | Entry point of the Flask web application containing routing, authentication, prediction, and database operations.     |
| **train_model.py**     | Complete machine learning training pipeline with preprocessing, model selection, evaluation, and artifact generation. |
| **heartiq_records.db** | SQLite database used for storing user accounts, administrator settings, and prediction history.                       |

## Why This Project?

Many heart disease prediction projects stop after training a machine learning model. HealthGuard AI was built to go beyond that by providing a complete web application around the prediction process.

The application allows users to securely create an account, receive heart disease risk predictions, maintain a history of previous assessments, compare health trends over time, and view personalized health recommendations. It also includes an administrative dashboard for managing users, viewing analytics, and exporting patient records.

The goal of this project was not only to build an accurate prediction model, but also to understand how machine learning can be integrated into a complete software application.

## Getting Started

### Prerequisites

- Python 3.8 or above
- Git
- pip

### Clone the Repository

```bash
git clone https://github.com/Sujay6612/HEALTHGUARD-AI.git
cd HEALTHGUARD-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

After starting the server, open your browser and visit:

```
http://127.0.0.1:5000
```

## Using the Application

### Patient Workflow

1. Register a new account.
2. Log in using your credentials.
3. Enter the required health parameters.
4. Submit the prediction request.
5. View the predicted heart disease risk along with confidence and personalized recommendations.
6. Access previous prediction history to compare health trends over time.

### Administrator Workflow

The administrator can:

- View registered users
- Search patient records
- Monitor prediction statistics
- Export patient records as CSV
- Manage administrator credentials
- View high-risk alerts

## Dataset

This project is trained using the **Cleveland Heart Disease Dataset**, one of the most widely used benchmark datasets for heart disease prediction.

The training pipeline extracts the five clinical features used by the web application:

- Age
- Sex
- Resting Blood Pressure
- Cholesterol
- Resting ECG Results

During training, the application automatically validates feature names, handles missing values, normalizes the target labels, and prepares the dataset before model training.

Keeping the prediction form limited to five inputs makes the application easier to use while still demonstrating a complete machine learning workflow.

## Project Highlights

- End-to-end Flask web application integrating machine learning with a healthcare workflow.
- Automated model selection using GridSearchCV and multiple classification algorithms.
- Secure authentication with password hashing and session management.
- Persistent patient history using SQLite.
- Personalized health recommendations based on prediction results.
- Administrative dashboard with analytics, patient search, and CSV export.
- Automatic database initialization and administrator account seeding.

## Future Improvements

Some enhancements planned for future versions include:

- Email verification and password reset functionality.
- Support for additional clinical features to improve prediction quality.
- Interactive dashboards with charts and health trend visualizations.
- Role-based access control with multiple administrator levels.
- REST API for integration with external healthcare systems.
- Deployment using Docker and cloud infrastructure.
- Integration with PostgreSQL for production environments.

---

## Key Learnings

Building this project provided practical experience in combining machine learning with software engineering. Some of the key takeaways include:

- Designing an end-to-end machine learning application instead of only training a model.
- Integrating a trained model into a Flask web application.
- Implementing secure authentication using password hashing.
- Working with SQLite for persistent data storage.
- Building reusable training pipelines with automated model selection and evaluation.
- Structuring a Python project for maintainability and future scalability.

---

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project in accordance with the terms of the license.
