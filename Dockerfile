FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python", "full_automation.py"]
