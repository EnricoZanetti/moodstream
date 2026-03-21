FROM nodered/node-red:latest

RUN npm install node-red-contrib-influxdb

COPY settings.js /data/settings.js
COPY flows_cred.json /data/flows_cred.json
