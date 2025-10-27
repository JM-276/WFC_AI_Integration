
// =============================
// Shopfloor Starter Cypher Pack
// Aligned with shopfloor_graph_schema.md
// =============================

// 0) Helpful indexes (optional — run once)
// CREATE CONSTRAINT machine_id IF NOT EXISTS FOR (m:Machine) REQUIRE m.id IS UNIQUE;
// CREATE CONSTRAINT sensor_id IF NOT EXISTS FOR (s:Sensor) REQUIRE s.id IS UNIQUE;
// CREATE CONSTRAINT operator_id IF NOT EXISTS FOR (o:Operator) REQUIRE o.id IS UNIQUE;
// CREATE CONSTRAINT workorder_id IF NOT EXISTS FOR (w:WorkOrder) REQUIRE w.id IS UNIQUE;
// CREATE CONSTRAINT maintlog_id IF NOT EXISTS FOR (l:MaintenanceLog) REQUIRE l.id IS UNIQUE;
// CREATE CONSTRAINT zone_id IF NOT EXISTS FOR (z:FacilityZone) REQUIRE z.id IS UNIQUE;
// CREATE CONSTRAINT batch_id IF NOT EXISTS FOR (b:ProductionBatch) REQUIRE b.id IS UNIQUE;

// 1) Machines with overdue work orders
// Params: (none)
MATCH (m:Machine)-[:HAS_WORK_ORDER]->(w:WorkOrder)
WHERE w.status = 'Overdue'
RETURN m.id AS machineId, w.id AS workOrderId, w.scheduledTime AS due
ORDER BY due ASC;

// 2) Machines with high sensor readings (e.g., vibration)
// Params: $threshold (float), $unit (string)
MATCH (m:Machine)-[:MONITORED_BY]->(s:Sensor)
WHERE s.reading > $threshold AND s.unit = $unit
RETURN m.id AS machineId, s.id AS sensorId, s.type AS sensorType, s.reading AS reading, s.timestamp AS at
ORDER BY reading DESC;

// 3) Current operator of a machine
// Params: $machineId (string)
MATCH (m:Machine {id: $machineId})- [r:OPERATED_BY]->(o:Operator)
RETURN m.id AS machineId, o.id AS operatorId, o.name AS operatorName, r.shift AS shift;

// 4) Sensors for machines in a specific zone
// Params: $zoneId (string)
MATCH (z:FacilityZone {id: $zoneId})<-[:LOCATED_IN]-(m:Machine)-[:MONITORED_BY]->(s:Sensor)
RETURN z.id AS zoneId, m.id AS machineId, s.id AS sensorId, s.type AS sensorType, s.reading AS reading, s.unit AS unit;

// 5) Machines with no recent maintenance (before cutoff)
// Params: $cutoff (date or datetime string)
MATCH (m:Machine)
WHERE m.lastMaintenance < datetime($cutoff)
RETURN m.id AS machineId, m.lastMaintenance AS lastMaintenance
ORDER BY lastMaintenance ASC;

// 6) Average energy usage per zone
MATCH (m:Machine)-[:LOCATED_IN]->(z:FacilityZone)
WITH z, avg(m.energyUsage) AS avgEnergy
RETURN z.id AS zoneId, z.name AS zoneName, round(avgEnergy,2) AS avgEnergy_kWh
ORDER BY avgEnergy_kWh DESC;

// 7) Path from operator to sensors (who operates machines monitored by which sensors)
MATCH (o:Operator)<-[:OPERATED_BY]-(m:Machine)-[:MONITORED_BY]->(s:Sensor)
RETURN o.id AS operatorId, o.name AS operatorName, m.id AS machineId, s.id AS sensorId, s.type AS sensorType;

// 8) Work order logs for a machine
// Params: $machineId (string)
MATCH (m:Machine {id: $machineId})-[:HAS_WORK_ORDER]->(w:WorkOrder)-[:HAS_LOG]->(l:MaintenanceLog)
RETURN m.id AS machineId, w.id AS workOrderId, l.id AS logId, l.issue AS issue, l.resolution AS resolution, l.duration AS hours
ORDER BY w.id, l.id;

// 9) Machines with active production batches at a given time
// Params: $now (datetime string)
MATCH (m:Machine)-[:PART_OF_BATCH]->(b:ProductionBatch)
WHERE datetime(b.startTime) <= datetime($now) AND datetime(b.endTime) >= datetime($now)
RETURN m.id AS machineId, b.id AS batchId, b.productType AS product, b.startTime AS start, b.endTime AS end;

// 10) Simple health snapshot: status, last maintenance, latest sensor reading
MATCH (m:Machine)-[:MONITORED_BY]->(s:Sensor)
WITH m, s ORDER BY s.timestamp DESC
WITH m, collect(s)[0] AS latest
RETURN m.id AS machineId, m.status AS status, m.lastMaintenance AS lastMaintenance,
       latest.id AS latestSensorId, latest.type AS sensorType, latest.reading AS latestReading, latest.unit AS unit, latest.timestamp AS at;
