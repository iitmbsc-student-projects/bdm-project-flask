FROM continuumio/miniconda3
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple && pip3 cache purge && conda clean --all
EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["python3", "app.py"]
