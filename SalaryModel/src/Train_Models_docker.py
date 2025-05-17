import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle


# Initialize Spark Session
spark = SparkSession.builder.appName("SalaryModel").getOrCreate()

def load_data():
    return spark.read.csv('/app/src/Salary-Data.csv', header=True, inferSchema=True)

def preprocess_data(df):
    df = df.dropna()
    job_title_count = df.groupBy("Job Title").count().filter(col("count") > 25)
    frequent_titles = [row["Job Title"] for row in job_title_count.collect()]
    df = df.withColumn("Job Title", when(col("Job Title").isin(frequent_titles), col("Job Title")).otherwise("Others"))
    df = df.withColumn("Education Level", when(col("Education Level") == "Bachelor's Degree", "Bachelor's")
                       .when(col("Education Level") == "Master's Degree", "Master's")
                       .when(col("Education Level") == "phD", "PhD")
                       .otherwise(col("Education Level")))
    return df, frequent_titles

def encode_features(df):
   
    df = df.withColumn("Education Level",
        when(col("Education Level") == "High School", 0)
        .when(col("Education Level") == "Bachelor's", 1)
        .when(col("Education Level") == "Master's", 2)
        .when(col("Education Level") == "PhD", 3)
        .otherwise(None))

  
    df = df.withColumn("Gender", when(col("Gender") == "Male", 1).otherwise(0))

    
    job_title_dummies = df.select("Job Title").distinct().rdd.flatMap(lambda x: x).collect()
    for title in job_title_dummies:
        df = df.withColumn("Job Title_{}".format(title), when(col("Job Title") == title, 1).otherwise(0))

    df = df.drop("Job Title")

    feature_columns = [col for col in df.columns if col != "Salary"]

    return df, feature_columns, job_title_dummies

def train_and_save_models(x_train, y_train, x_test, y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    model_configs = {
        'Linear_Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Random_Forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10]
            }
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.001, 0.01, 0.1]
            }
        }
    }
    for model_name, config in model_configs.items():
        print(f"\n🔄 Training {model_name}...")
        grid_search = GridSearchCV(config['model'], config['params'], cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model.predict(x_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = r2 * 100  
        print(f"✅ {model_name} trained.")
        print(f"   🔧 Best Params : {best_params}")
        print(f"   📊 Test Metrics:")
        print(f"     - RMSE       : {rmse:.2f}")
        print(f"     - MAE        : {mae:.2f}")
        print(f"     - R² Score   : {r2:.2f}")
        print(f"     - Accuracy % : {accuracy:.2f}%")

        # Save model to /app/models
        with open(f"/app/src/{model_name}.pkl", 'wb') as f:
            pickle.dump(best_model, f)

        print(f"   ✅ {model_name} model saved.")
    

def main():
    
    print("*************** Loading and preprocessing data... ***************")
    df, frequent_titles = preprocess_data(load_data())
    
    print("*************** Data loaded and preprocessed. ***************")
    print("*************** Encoding features... ***************")
    encoded_df, feature_columns, job_title_columns = encode_features(df)

    pandas_df = encoded_df.toPandas()
    x_train, x_test, y_train, y_test = train_test_split(
        pandas_df[feature_columns], pandas_df['Salary'], test_size=0.25, random_state=42
    )

    # Save metadata with job_title_columns
    with open('preprocessing_metadata.pkl', 'wb') as f:
        pickle.dump({
            'feature_columns': feature_columns,
            'job_title_columns': job_title_columns,
            'frequent_titles': frequent_titles,
        }, f)
    df.show(10)
    train_and_save_models(x_train, y_train, x_test, y_test)
    print(" ************** Models trained and saved successfully. **************")

if __name__ == '__main__':
    main()
