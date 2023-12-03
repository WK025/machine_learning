# docker build -t streamlit .
# docker images
# docker run -p 8501:8501 streamlit
# docker rm $(docker ps -aq)
# docker rmi $(docker images -q)

# Sets the Base Image
FROM python:3.11.0-slim
WORKDIR /app

# Install Git to clone
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*
# Clone the repo
# RUN git clone https://github.com/WK025/machine_learning.git ./Project


##### copy requirements file and install necessary packages

# ADD the requirements.txt into the /app directory in the container
COPY requirements.txt /app

RUN pip3 install -r requirements.txt --target "streamlit"


##### Copy function code to docker container

# ADD the app.py file into the /app directory in the container
COPY app.py /app

# ADD the data and pickle file into the /app/datasets and /app/saved_stacked_models directory in the container
COPY datasets/downsampled_dataset_after_feature_selection.csv /app/datasets
COPY saved_models/* /app/saved_models

# Expose the port used by Streamlit (8501 by default)
EXPOSE 8501

# Test a container to check that it is still working
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

##### SET THE COMMAND OF THE CONTAINER FOR /app
# app (name of py file)
# handler (name of function to execute for lambda job)
# CMD ["streamlit", "run", "./app.py"]
# RUN ["chmod", "+x", "/docker/entrypoints/docker-entrypoint.sh"] \
    # ["chmod", "+x", "/docker/entrypoints/sidekiq-entrypoint.sh"]
# ENTRYPOINT ["./docker/entrypoints/docker-entrypoint.sh"]

# Configure a container that will run as an executable
ENTRYPOINT ["streamlit", "run"]
# ENTRYPOINT ["streamlit", "run", "./app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD [ "./app.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
# CMD ["streamlit", "run", "./app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
CMD ["./app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]