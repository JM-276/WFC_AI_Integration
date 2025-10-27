# Factory Shopfloor LPG Schema

## Node Types

### Machine
- Properties:
  - id (string)
  - type (string)
  - status (string: Running, Idle, Maintenance)
  - lastMaintenance (date)
  - energyUsage (float: kWh)

### Sensor
- Properties:
  - id (string)
  - type (string: vibration, temperature, pressure)
  - reading (float)
  - unit (string)
  - timestamp (datetime)

### Operator
- Properties:
  - id (string)
  - name (string)
  - shift (string: Day, Night)
  - certifications (list of strings)

### WorkOrder
- Properties:
  - id (string)
  - status (string: Scheduled, InProgress, Completed, Overdue)
  - scheduledTime (datetime)
  - priority (string: High, Medium, Low)

### MaintenanceLog
- Properties:
  - id (string)
  - issue (string)
  - resolution (string)
  - duration (float: hours)

### FacilityZone
- Properties:
  - id (string)
  - name (string)
  - description (string)

### ProductionBatch
- Properties:
  - id (string)
  - productType (string)
  - startTime (datetime)
  - endTime (datetime)

## Relationship Types

### MONITORED_BY
- From: Machine
- To: Sensor
- Properties:
  - timestamp (datetime)

### OPERATED_BY
- From: Machine
- To: Operator
- Properties:
  - shift (string)

### HAS_WORK_ORDER
- From: Machine
- To: WorkOrder
- Properties:
  - assignedDate (datetime)

### HAS_LOG
- From: WorkOrder
- To: MaintenanceLog
- Properties:
  - createdDate (datetime)

### LOCATED_IN
- From: Machine
- To: FacilityZone
- Properties:
  - since (datetime)

### PART_OF_BATCH
- From: Machine
- To: ProductionBatch
- Properties:
  - assignedDate (datetime)
