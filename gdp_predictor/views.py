
# gdp_predictor/views.py
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
import base64






def home(request):
    # Load your dataset (replace 'gdp_data.csv' with your actual file name)
    df = pd.read_csv('static/data.csv')

    # Combine GDP values from BUSINESS, AGRICULTURE, and IT SECTOR
    df['TOTAL_GDP'] = df['BUSINESS'] + df['AGRICULTURE'] + df['IT_SECTOR']

    # Split the data into features (X) and target variable (y)
    X = df[['Year']]
    y = df['TOTAL_GDP']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Plot the regression line
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.title('Total GDP Prediction using Linear Regression')
    plt.xlabel('Year')
    plt.ylabel('Total GDP')

    # Convert the plot to base64 for embedding in HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    img.close()

    # Display results on the webpage
    context = {
        'actual_total_gdp': list(y_test.values),
        'predicted_total_gdp': list(y_pred),
        'plot_url': f"data:image/png;base64,{plot_url}",
    }

    return render(request, 'home.html', context)

