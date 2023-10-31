# syntax=docker/dockerfile:1

FROM python:3.8
WORKDIR /app


##### copy requirements file and install necessary packages

# ***CODE TO DO***
# ADD the requirements.txt into the ${LAMBDA_TASK_ROOT} directory in the container
COPY requirements.txt .

RUN pip3 install -r app/requirements.txt --target "app"

##### Copy function code to docker container

# ***CODE TO DO***
# ADD the app.py file into the ${LAMBDA_TASK_ROOT} directory in the container
COPY app.py .

# Expose the port used by Streamlit (8501 by default)
EXPOSE 8501

##### SET THE COMMAND OF THE CONTAINER FOR THE LAMBDA HANDLER
# app (name of py file)
# handler (name of function to execute for lambda job)
CMD ["streamlit", "run", "app.py"]

# docker build -t my-app .
# docker run -p 4000:80 my-app