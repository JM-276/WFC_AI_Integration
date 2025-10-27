A Cypher query template is a parameterized Cypher statement — think of it like a “prepared SQL statement” for Neo4j.

It’s a reusable query pattern where variables are substituted dynamically at runtime.
You write the query structure once, and inject real values later (from your code, API, or agent)

An MCP-style contract you can register in your tool broker. It defines 5 operations:

highVibrationMachines(threshold, unit)
overdueWorkOrders()
currentOperator(machineId)
sensorsByZone(zoneId)
dueForMaintenance(cutoff)

Each includes: description, input schema, the Cypher to run, output fields, and an example call—so your Secondary Agent can call them deterministically