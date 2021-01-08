#I specify the parent base image which is the python version 3.7
FROM python:3.7

LABEL maintainer="smr1020@gmail.com"

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED=1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /

# copy all files to home of container filesystem 
COPY . .

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# set app port
EXPOSE 5000 

# Run src.html when the container launches
CMD ["uvicorn", "src.html:app", "--host=0.0.0.0", "--port=5000"]