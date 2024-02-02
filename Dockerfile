FROM ubuntu:20.04

# Ban interactive shell during installation
ARG DEBIAN_FRONTEND=noninteractive

# Install required packages
WORKDIR /app
COPY . /app
RUN apt update && apt install -y python3.8 python3.8-dev
RUN apt install -y build-essential
RUN apt install -y openjdk-8-jdk
RUN apt install -y libyaml-cpp-dev
RUN apt install -y libgmp3-dev libmpfr-dev libmpfr-doc
RUN apt install -y gsl-bin libgsl0-dev
RUN apt install -y bison
RUN apt install -y flex
RUN apt install -y libglpk-dev

# Setup Verisig
RUN apt install -y git
RUN git clone https://github.com/Verisig/verisig
WORKDIR verisig/flowstar
RUN make
WORKDIR ../verisig-src
RUN ./gradlew installDist
WORKDIR /app

# Setup Python packages
RUN apt install -y python3-pip
RUN pip install --no-cache-dir -r requirements.txt

# Wait for manual execution of scripts
CMD ["tail", "-f", "/dev/null"]

