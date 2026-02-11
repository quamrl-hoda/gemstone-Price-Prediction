# Gemstone Price Prediction

This project is an end-to-end Machine Learning web application that predicts the price of a gemstone based on its characteristics (Carat, Cut, Color, Clarity, Depth, Table, and Dimensions).

## About the Data

**The dataset:** The goal is to predict `price` of given diamond (Regression Analysis).

There are 10 independent variables (including `id`):

*   `id` : unique identifier of each diamond
*   `carat` : Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
*   `cut` : Quality of Diamond Cut
*   `color` : Color of Diamond
*   `clarity` : Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
*   `depth` : The depth of diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface)
*   `table` : A diamond's table is the facet which can be seen when the stone is viewed face up.
*   `x` : Diamond X dimension
*   `y` : Diamond Y dimension
*   `z` : Diamond Z dimension

**Target variable:**
*   `price`: Price of the given Diamond.

**Dataset Source Link:** [Kaggle Playground Series s3e8](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

## Project Structure

The project follows a modular structure for better maintainability and scalability.

```
gemstonePricePrediction/
├── artifacts/              # Generated artifacts (data, models, logs)
├── config/                 # Configuration files (config.yaml)
├── research/               # Jupyter notebooks for experimentation
├── src/                    # Source code
│   └── gemstonePricePrediction/
│       ├── components/     # ML components (Ingestion, Transformation, Training)
│       ├── config/         # Configuration manager
│       ├── entity/         # Data classes for configuration
│       ├── pipeline/       # Pipeline stages
│       ├── utils/          # Utility functions
│       └── constants/      # Project constants
├── templates/              # HTML templates for the Flask app
├── app.py                  # Flask application entry point
├── main.py                 # Main script to run the ML pipeline
├── params.yaml             # Model hyperparameters
├── schema.yaml             # Data schema
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup script
```

## Machine Learning Pipeline

The pipeline consists of the following stages:

1.  **Data Ingestion**: Downloads the dataset from a source URL and extracts it.
2.  **Data Validation**: Validates the dataset against the defined schema.
3.  **Data Transformation**: Preprocesses the data (handling missing values, encoding categorical variables, scaling) and saves the preprocessor object.
4.  **Model Trainer**: Trains multiple regression models (Linear Regression, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost), evaluates them, and saves the best performing model.
    - **Best Model**: The final model used for prediction is a **Voting Regressor**, which is an ensemble of the tuned CatBoost, XGBoost, and tuned KNN models. This ensemble approach combines the strengths of individual models to achieve better generalization and accuracy.
5.  **Prediction**: Loads the saved model and preprocessor to make predictions on new data.

## How to Run the Project

### Prerequisites

*   Python 3.8+
*   Anaconda (recommended) or Virtualenv

### Setup

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment:**

    ```bash
    conda create -n mlenv python=3.8 -y
    conda activate mlenv
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Training the Model

To run the entire training pipeline (Ingestion -> Validation -> Transformation -> Training), execute:

```bash
python main.py
```

This will generate the trained model at `artifacts/model_trainer/model.pkl` and the preprocessor at `artifacts/data_transformation/preprocessor.pkl`.

### Running the Web Application

To start the Flask web application:

```bash
python app.py
```

The application will be accessible at `http://localhost:5000`. You can use the web interface to input gemstone details and get price predictions.

## UI Features

*   **Home Page**: A modern, premium interface for inputting gemstone data.
*   **Dataset Info**: Detailed information about the dataset and features.
*   **About**: Project context and goals.

## key Dependencies

*   Flask
*   Scikit-learn
*   Pandas
*   Numpy
*   XGBoost
*   CatBoost
*   Dill
