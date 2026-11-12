FROM jfhernandeza/ul-cs6524-cv-python-latex:lab01
COPY requirements.txt /tmp/lab3-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/lab3-requirements.txt
WORKDIR /workspace
