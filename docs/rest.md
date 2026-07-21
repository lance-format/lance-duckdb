# Connecting to a Lance REST Namespace

This guide explains how to connect DuckDB to a Lance REST Namespace server (e.g. LanceDB Cloud or Enterprise) using the lance-duckdb extension.

## Overview

A Lance REST Namespace provides a centralized catalog for managing Lance tables. The REST Namespace API allows:

- Listing tables in a namespace
- Creating and dropping tables
- Describing tables (getting location and storage credentials)
- Querying tables through the `query_table` API
- Credential vending for accessing underlying storage (S3, GCS, Azure, etc.)

## Prerequisites

1. A Lance REST Namespace server endpoint available
2. The lance-duckdb extension built and available
3. Authentication credentials (API key, bearer token, etc.)

## ATTACH Syntax

```sql
ATTACH '<namespace_id>' AS <alias> (
    TYPE LANCE,
    ENDPOINT '<server_url>',
    HEADER '<header1>=<value1>;<header2>=<value2>'
);
```

### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `<namespace_id>` | The namespace identifier to connect to | Yes |
| `<alias>` | Local alias for the attached database | Yes |
| `ENDPOINT` | URL of the REST Namespace server | Yes |
| `HEADER` | Custom HTTP headers (semicolon-separated key=value pairs) | No |
| `DELIMITER` | Namespace delimiter (default: `$`) | No |
| `BEARER_TOKEN` | Bearer token for authentication | No |
| `API_KEY` | API key for authentication | No |

### Custom Headers

The `HEADER` option allows passing custom HTTP headers to the REST Namespace server. Multiple headers can be specified using semicolon separation:

```sql
HEADER 'x-lancedb-database=my_db;x-api-key=sk_xxx;x-custom-header=value'
```

## Example: Connecting to a REST Namespace Server

### 1. Connect from DuckDB

```sql
-- Load the Lance extension
LOAD 'lance.duckdb_extension';

-- Attach to the REST Namespace
ATTACH 'ns1' AS lance_ns (
    TYPE LANCE,
    ENDPOINT 'http://localhost:10024',
    HEADER 'x-lancedb-database=lance_ns;x-api-key=sk_localtest'
);

-- Switch to the attached database
USE lance_ns;
```

### 2. List Tables

```sql
SHOW TABLES;
```

Output:
```
┌─────────┐
│  name   │
│ varchar │
├─────────┤
│ 0 rows  │
└─────────┘
```

### 3. Create a Table

```sql
CREATE TABLE users (
    id INTEGER,
    name VARCHAR,
    email VARCHAR
);
```

### 4. Insert Data

```sql
INSERT INTO users VALUES
    (1, 'Alice', 'alice@example.com'),
    (2, 'Bob', 'bob@example.com'),
    (3, 'Charlie', 'charlie@example.com');
```

### 5. Query Data

For tables attached through a REST Namespace, `SELECT` is executed through the
Lance Namespace `query_table` API. DuckDB pushes the required projection,
supported `WHERE` predicates, and `LIMIT`/`OFFSET` pairs into the request and
reads the Arrow IPC response. Reads therefore do not require DuckDB to open the
underlying object-store location directly. Unsupported predicates remain in
DuckDB, and an `OFFSET` without a `LIMIT` is also evaluated by DuckDB.

```sql
-- Select all rows
SELECT * FROM users;
```

Output:
```
┌───────┬─────────┬─────────────────────┐
│  id   │  name   │        email        │
│ int32 │ varchar │       varchar       │
├───────┼─────────┼─────────────────────┤
│     1 │ Alice   │ alice@example.com   │
│     2 │ Bob     │ bob@example.com     │
│     3 │ Charlie │ charlie@example.com │
└───────┴─────────┴─────────────────────┘
```

```sql
-- Select with filter
SELECT * FROM users WHERE id > 1;
```

Output:
```
┌───────┬─────────┬─────────────────────┐
│  id   │  name   │        email        │
│ int32 │ varchar │       varchar       │
├───────┼─────────┼─────────────────────┤
│     2 │ Bob     │ bob@example.com     │
│     3 │ Charlie │ charlie@example.com │
└───────┴─────────┴─────────────────────┘
```

```sql
-- Aggregation
SELECT COUNT(*) as total_users FROM users;
```

Output:
```
┌─────────────┐
│ total_users │
│    int64    │
├─────────────┤
│           3 │
└─────────────┘
```

### 6. Using Fully Qualified Names

You can also use fully qualified table names without switching databases:

```sql
-- Create table with fully qualified name
CREATE TABLE lance_ns.main.my_table (col1 INTEGER, col2 VARCHAR);

-- Query with fully qualified name
SELECT * FROM lance_ns.main.my_table;
```
