#!/bin/bash

## Create required directories
#mkdir -p /root/.cache/sessions
mkdir -p /root/.config/xfce4/xfconf/xfce-perchannel-xml
#
## Create a default XFCE session
#cat > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-session.xml << EOF
#<?xml version="1.0" encoding="UTF-8"?>
#<channel name="xfce4-session" version="1.0">
#  <property name="general" type="empty">
#    <property name="AutoSave" type="bool" value="false"/>
#    <property name="SaveOnExit" type="bool" value="false"/>
#  </property>
#  <property name="sessions" type="empty">
#    <property name="Failsafe" type="empty">
#      <property name="IsFailsafe" type="bool" value="false"/>
#      <property name="Count" type="int" value="1"/>
#      <property name="Client0_Command" type="array">
#        <value type="string">xfce4-panel</value>
#      </property>
#      <property name="Client0_Priority" type="int" value="10"/>
#      <property name="Client1_Command" type="array">
#        <value type="string">xfdesktop</value>
#      </property>
#      <property name="Client1_Priority" type="int" value="15"/>
#    </property>
#  </property>
#</channel>
#EOF

# Start virtual display
#Xvfb :99 -screen 0 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH} -ac +extension GLX +render -noreset &

## Start D-Bus
#dbus-daemon --system --fork
#dbus-daemon --session --fork
#sleep 1

## Start desktop with specific settings
#export XDG_CONFIG_HOME=/root/.config
#export XDG_CACHE_HOME=/root/.cache
#export XDG_DATA_HOME=/root/.local/share
#
#mkdir -p $XDG_DATA_HOME
#mkdir -p $XDG_CACHE_HOME

# Start XFCE session
#startxfce4 --sm-client-disable &
## Create desktop background config
cat > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-desktop" version="1.0">
  <property name="backdrop" type="empty">
    <property name="screen0" type="empty">
      <property name="monitor0" type="empty">
        <property name="workspace0" type="empty">
          <property name="last-image" type="string" value="/usr/share/backgrounds/xfce/background.png"/>
        </property>
      </property>
    </property>
  </property>
</channel>
EOF

# Create panel config (minimal panel)
cat > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-panel" version="1.0">
  <property name="panels" type="empty">
    <property name="panel-1" type="empty">
      <property name="position" type="string" value="p=0;x=0;y=0"/>
      <property name="length" type="uint" value="1"/>
      <property name="position-locked" type="bool" value="true"/>
      <property name="plugin-ids" type="array">
      </property>
    </property>
  </property>
</channel>
EOF

Xvfb :99 -screen 0 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH} &
sleep 2
#xfconf-query -c xfce4-desktop -p /backdrop/screen0/monitor0/workspace0/last-image -s /usr/share/backgrounds/xfce/background.png
##xfconf-query -c xfce4-desktop -p /backdrop/screen0/monitor0/workspace0/last-image -s /root/Pictures/my_background.png
#
#for workspace in 0 1 2 3; do
#  xfconf-query -c xfce4-desktop -p /backdrop/screen0/monitorscreen/workspace${workspace}/last-image -s /usr/share/backgrounds/xfce/background.png
#done

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
</channel>
EOF

#cat <<EOF > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml
#<?xml version="1.0" encoding="UTF-8"?>
#<channel name="xfce4-panel" version="1.0">
#  <property name="plugin-ids" type="array">
#    <!-- No plugins -->
#  </property>
#  <property name="panels" type="empty">
#    <property name="panel-1" type="empty">
#      <property name="disable" type="bool" value="true"/>
#    </property>
#    <property name="panel-2" type="empty">
#      <property name="disable" type="bool" value="true"/>
#    </property>
#  </property>
#</channel>
#EOF

startxfce4 &
sleep 5  # Give desktop time to start
# Reload the XFCE panel configuration
# Add Terminal and Browser shortcuts to the desktop
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
xfce4-panel --quit

#xfdesktop --reload
#xfconf-query -c xfce4-desktop -l -v
#xfconf-query -c xfce4-desktop -l
#cat /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml


## Set a background color
#xsetroot -solid "#333333"
# Run the Python script
python3 synthetic_script.py
