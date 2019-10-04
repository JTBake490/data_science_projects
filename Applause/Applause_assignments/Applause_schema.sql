CREATE DATABASE applause;

\connect applause;

CREATE TABLE testers(
    testerId BIGSERIAL NOT NULL PRIMARY KEY,
    firstName VARCHAR(15) NOT NULL,
    lastName VARCHAR(15) NOT NULL,
    country char(2),
    lastLogin DATE
);

CREATE TABLE devices(
    deviceId BIGSERIAL NOT NULL PRIMARY KEY,
    deviceName varchar(20) NOT NULL
);

CREATE TABLE tester_device(
    testerId BIGSERIAL REFERENCES testers(testerId),
    deviceId BIGSERIAL REFERENCES devices(deviceId)
);

CREATE TABLE bugs(
    bugId BIGINT NOT NULL PRIMARY KEY,
    deviceId BIGSERIAL REFERENCES devices(deviceId),
    testerId BIGSERIAL REFERENCES testers(testerId)
);

\COPY testers FROM '../Applause_files/testers.csv' DELIMITER ',' CSV HEADER;
\COPY devices FROM '../Applause_files/devices_devicename.csv' DELIMITER ',' CSV HEADER;
\COPY tester_device FROM '../Applause_files/tester_device.csv' DELIMITER ',' CSV HEADER;
\COPY bugs FROM '../Applause_files/bugs.csv' DELIMITER ',' CSV HEADER;