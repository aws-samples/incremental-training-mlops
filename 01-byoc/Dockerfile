ARG BASE_IMG=${BASE_IMG}
FROM ${BASE_IMG} 
COPY code/* /
RUN pip install -r requirements.txt --no-cache-dir 
RUN apt-get update -y && apt-get install python3-soundfile -y  && rm -rf /var/lib/apt/lists/*
WORKDIR /
ENV PATH="/:${PATH}"



