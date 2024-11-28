#!/bin/bash
# Clean up any existing X server locks
rm -f /tmp/.X99-lock
rm -f /tmp/.X11-unix/X99
## Create required directories
mkdir -p /root/.config/xfce4/xfconf/xfce-perchannel-xml
#
## Create desktop background config
cat <<EOF > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-desktop" version="1.0">
  <property name="backdrop" type="empty">
    <property name="screen0" type="empty">
      <property name="monitor0" type="empty">
        <property name="workspace0" type="empty">
          <property name="last-image" type="string" value="/usr/share/backgrounds/xfce/background.png"/>
          <property name="image-style" type="int" value="5"/>
        </property>
      </property>
      <property name="monitorscreen" type="empty">
        <property name="workspace0" type="empty">
          <property name="last-image" type="string" value="/usr/share/backgrounds/xfce/background.png"/>
          <property name="image-style" type="int" value="5"/>
        </property>
        <property name="workspace1" type="empty">
          <property name="last-image" type="string" value="/usr/share/backgrounds/xfce/background.png"/>
          <property name="image-style" type="int" value="5"/>
        </property>
        <property name="workspace2" type="empty">
          <property name="last-image" type="string" value="/usr/share/backgrounds/xfce/background.png"/>
          <property name="image-style" type="int" value="5"/>
        </property>
        <property name="workspace3" type="empty">
          <property name="last-image" type="string" value="/usr/share/backgrounds/xfce/background.png"/>
          <property name="image-style" type="int" value="5"/>
        </property>
      </property>
    </property>
  </property>
  <property name="desktop-icons" type="empty">
    <property name="file-icons" type="empty">
      <property name="show-filesystem" type="bool" value="false"/>
    </property>
  </property>
</channel>
EOF

# Create panel config (minimal panel)
cat <<EOF > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-panel" version="1.0">
  <property name="configver" type="int" value="2"/>
  <property name="panels" type="array">
    <value type="int" value="2" />
      <property name="panel-2" type="empty">
        <property name="plugin-ids" type="array">
        </property>
        <property name="position" type="string" value="p=0;x=4096;y=4096" />
        <property name="size" type="uint" value="0" />
      </property>
  </property>
  <property name="plugins" type="empty">
  </property>
</channel>
EOF

Xvfb :99 -screen 0 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH} &
sleep 2
XVFB_PID=$!

# Wait for X server to start
for i in $(seq 1 10); do
    if xdpyinfo >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

XFCE_PID=$!
mkdir -p /root/Desktop
cat <<EOF > /root/Desktop/xfce4-terminal.desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=Terminal
Exec=xfce4-terminal
Icon=utilities-terminal
Terminal=false
Categories=System;TerminalEmulator;
EOF
chmod +x /root/Desktop/xfce4-terminal.desktop

cat <<EOF > /root/Desktop/firefox.desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=Firefox
Exec=firefox
Icon=firefox
Terminal=false
Categories=Network;WebBrowser;
EOF
chmod +x ~/Desktop/firefox.desktop

startxfce4 &
sleep 5  # Give desktop time to start
xfce4-panel --quit
sleep 1  # Give desktop time to start

x11vnc -display :99 -passwd vncpassword -forever -nopw -listen 0.0.0.0 -rfbport 5900 &
## Keep the container running if no command is specified
if [ $# -eq 0 ]; then
    wait $XVFB_PID
else
    exec "$@"
fi
