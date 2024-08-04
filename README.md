# Sales Prediction using Python and Machine Learning

This project involves developing a sales prediction model using Python and machine learning. The goal is to predict future sales based on various factors such as advertising spend on TV, Radio, and Newspaper.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sales prediction involves forecasting the future sales of a product based on historical data. This project uses advertising spend data to predict sales, helping businesses allocate their advertising budget more effectively.

## Dataset

The dataset used in this project contains information on advertising spending across different media (TV, Radio, Newspaper) and the corresponding sales. The dataset can be downloaded [here](#).

The dataset structure:
```
| Column      | Description                     |
|-------------|---------------------------------|
| Unnamed: 0  | Index                           |
| TV          | Advertising spend on TV         |
| Radio       | Advertising spend on Radio      |
| Newspaper   | Advertising spend on Newspaper  |
| Sales       | Sales                           |
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/sales-prediction.git
    cd sales-prediction
    ```

2. Install the required libraries:
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

## Usage

1. Load the dataset:
    ```python
    import pandas as pd

    data = pd.read_csv('sales_data.csv')
    data = data.drop(columns=['Unnamed: 0'])
    ```

2. Explore and visualize the data:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.pairplot(data)
    plt.show()

    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()
    ```

3. Split the data and train the model:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    X = data.drop('Sales', axis=1)
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

4. Evaluate the model:
    ```python
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    ```

5. Predict new data:
    ```python
    new_data = pd.DataFrame({
        'TV': [200.0],
        'Radio': [30.0],
        'Newspaper': [20.0]
    })
    predicted_sales = model.predict(new_data)
    print(predicted_sales)
    ```

## Model Training and Evaluation

The model is trained using the Linear Regression algorithm. The training and testing data are split in an 80-20 ratio. The model's performance is evaluated using Mean Squared Error (MSE) and R-squared metrics.

## Results

The model's Mean Squared Error (MSE) and R-squared values are printed after evaluation. You can use these metrics to assess the model's performance and accuracy in predicting sales based on advertising spend.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
