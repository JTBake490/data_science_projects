/* The total bugs found for every tester that has ever tested a device */
SELECT firstName, lastName, COUNT(bugId) AS bugs_found
FROM testers 
INNER JOIN bugs
ON testers.testerId = bugs.testerId
GROUP BY  firstName, lastName
ORDER BY COUNT(bugId) DESC;

/* The total bugs found for each tester when given specific devices but withou specifing countries */
/* This example specifies the devices: iPhone 4 and iPhone 5 */
SELECT firstName, lastName, COUNT(bugId) AS bugs_found
FROM bugs 
INNER JOIN testers
ON bugs.testerid = testers.testerid
INNER JOIN devices
ON bugs.deviceId = devices.deviceId
WHERE deviceName IN ('iPhone 4', 'iPhone 5') /* case sensitive */
GROUP BY firstName, lastName
ORDER BY COUNT(bugId) DESC;

/* The total bugs found for each person given specific countries but without specifing specific devices */
/* This example specifies the countries: US and JP */
SELECT firstName, lastName, COUNT(bugId) AS bugs_found
FROM testers
INNER JOIN bugs
ON testers.testerId = bugs.testerId
WHERE country IN ('US', 'JP') /* case sensitive */
GROUP BY firstName, lastName
ORDER BY COUNT(bugId) DESC;

/* The total bugs for each tester given specific countries and specific devices */
/* This example specifices the countries: US and JP; The example specifices the devices: iPhone 4 and iPhone 5 */
SELECT firstName, lastName, COUNT(bugId) AS bugs_found
FROM bugs
INNER JOIN testers
ON bugs.testerId = testers.testerId
INNER JOIN devices
ON bugs.deviceId = devices.deviceId
WHERE country IN ('US', 'JP') /* case sensitive */ AND devicename IN ('iPhone 4', 'iPhone 5') /* case sensitive */
GROUP BY firstName, lastName
ORDER BY COUNT(bugId) DESC;