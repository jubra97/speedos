FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies to the local user directory (eg. /root/.local)
RUN pip install --user -r requirements.txt

# copy files
COPY src/scripts/run_docker.py .
COPY src/ src/

CMD [ "python", "./run_docker.py" ]