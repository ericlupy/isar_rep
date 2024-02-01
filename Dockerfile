FROM python:3.8-slim

# Setup working directory for Docker and install Python requirements
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# Setup Verisig
RUN git clone https://github.com/Verisig/verisig
RUN apt install libgmp3-dev libmpfr-dev libmpfr-doc libmpfr4 libmpfr4-dbg gsl-bin libgsl0-dev bison flex gnuplot-x11 libglpk-dev
WORKDIR /verisig/flowstar
RUN make
WORKDIR ../verisig-src
RUN ./gradlew installDist
WORKDIR /app

# Wait for manual execution of scripts
CMD ["tail", "-f", "/dev/null"]

