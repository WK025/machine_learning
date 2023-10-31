# syntax=docker/dockerfile:1

# Sets the Base Image
FROM python:3.9-slim
WORKDIR /app

# Install Git to clone
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*
# Clone the repo
# RUN git clone https://github.com/streamlit/streamlit-example.git .

##### copy requirements file and install necessary packages

# ***CODE TO DO***
# ADD the requirements.txt into the ${LAMBDA_TASK_ROOT} directory in the container
COPY requirements.txt .

RUN pip3 install -r app/requirements.txt --target "app"

##### Copy function code to docker container

# ADD the app.py file into the app directory in the container
COPY app.py .

# ADD the data and pickle file into the app/data and app/saved_stacked_models directory in the container
COPY datasets/downsampled_dataset_after_feature_selection.csv ./datasets
COPY saved_stacked_models/StackedPickleSmoking.pkl ./saved_stacked_models
COPY saved_stacked_models/StackedPickleDrinking.pkl ./saved_stacked_models

# Expose the port used by Streamlit (8501 by default)
EXPOSE 8501

# Test a container to check that it is still working
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

##### SET THE COMMAND OF THE CONTAINER FOR THE LAMBDA HANDLER
# app (name of py file)
# handler (name of function to execute for lambda job)
CMD ["streamlit", "run", "app.py"]

# Configure a container that will run as an executable
# ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -t streamlit-app .
# docker images
# docker run -p 8501:8501 streamlit-app