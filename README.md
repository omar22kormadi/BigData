# Salary Prediction Big Data Pipeline

This project provides a big data pipeline for training and saving salary prediction models using Spark, Hadoop, and Docker. It includes a sample dataset and scripts for preprocessing and model training.

## Project Structure

```
docker-compose.yml
SalaryModel/
  src/
    Lasso.pkl
    Linear_Regression.pkl
    preprocessing_metadata.pkl
    Random_Forest.pkl
    requirements.txt
    Salary-Data.csv
    SVR.pkl
    Train_Models_docker.py
    XGBoost.pkl
```

## Components

- **Hadoop Cluster**: Namenode, Datanodes, ResourceManager for distributed storage and processing.
- **Spark Cluster**: Spark master and worker for distributed computation.
- **SalaryModel/src**: Contains data, model training script, and saved models.

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/)
- At least 4GB RAM available for Docker

### Setup

1. **Clone the repository** (if not already done):

   ```sh
   git clone https://github.com/omar22kormadi/BigData.git
   ```

2. **Start the cluster**:
   Open the folder where you can find docker-compose.yml file in cmd and build it

   ```sh
   docker-compose up --build
   ```

   This will start Hadoop and Spark services, and mount the `SalaryModel` directory into the containers.

3. **Access Spark and Hadoop UIs**:

   - Spark Master UI: [http://localhost:8080](http://localhost:8080)
   - Spark Worker UI: [http://localhost:8081](http://localhost:8081)
   - Hadoop Namenode UI: [http://localhost:9870](http://localhost:9870)
   - YARN ResourceManager UI: [http://localhost:8088](http://localhost:8088)

### Training Models

1. **Open a shell in the Spark master container**:

   ```sh
   docker exec -t spark-master bash
   ```

2. **Install Python dependencies** (if needed):

   ```sh
   pip install -r /app/src/requirements.txt
   ```

3. **Run the training script in Spark**:

   ```sh
   /opt/bitnami/spark/bin/spark-submit --master spark://spark-master:7077 /app/src/Train_Models_docker.py
   ```

   This will preprocess the data, train multiple regression models, and save them as `.pkl` files in `/app/src/`.

### Files

- `Salary-Data.csv`: Sample salary dataset.
- `Train_Models_docker.py`: Main script for preprocessing and training models.
- `requirements.txt`: Python dependencies.
- `*.pkl`: Saved trained models and preprocessing metadata.

## Notes

- The pipeline uses PySpark for distributed data processing and scikit-learn/XGBoost for model training.
- Models are saved in the `SalaryModel/src` directory.
- You can modify the dataset or script as needed for your use case.

---

**Author:** Amor Kormadi
**Contact:** amor.kormadi@polytechnicien.tn