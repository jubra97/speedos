FROM python:3.6

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies to the local user directory (eg. /root/.local)
RUN pip install --user -r requirements.txt

# expose port of speed_viz
EXPOSE 8521

# copy files
COPY src/evaluation/run.py .
COPY src/ src/

COPY setup.py .
RUN python setup.py build_ext --inplace

CMD [ "python", "./run.py" ]