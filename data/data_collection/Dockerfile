FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    xvfb \
    x11vnc \
    python3-tk \
    python3-dev \
    scrot \
    xdotool \
    x11-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    xfce4 xfce4-goodies xfce4-terminal \
    dbus-x11 \
    xserver-xorg \
    xinit \
    x11-xserver-utils \
    firefox \
    && apt-get clean

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Set up virtual display
ENV DISPLAY=:99
ENV SCREEN_WIDTH=512
ENV SCREEN_HEIGHT=368
ENV SCREEN_DEPTH=24

# Copy your scripts
COPY synthetic_script.py /app/
COPY synthetic_mouse_path.py /app/
COPY record_script.py /app/
COPY start.sh /app/

RUN mkdir -p /usr/share/backgrounds/xfce
COPY background.png /usr/share/backgrounds/xfce/

WORKDIR /app
ENV PYTHONPATH=/app

RUN chmod +x /app/start.sh
#
#CMD ["/app/start.sh"]
#
#ENTRYPOINT []
