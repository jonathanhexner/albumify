# How to Expand JSON Column into Separate Columns

## Method 1: Using `json_normalize()` (Recommended)

If your JSON column contains dictionaries:

```python
import pandas as pd
import json

# If JSON is stored as string
df['json_column'] = df['json_column'].apply(json.loads)

# Expand JSON column into separate columns
df_expanded = pd.json_normalize(df['json_column'])

# Combine with original DataFrame (drop original JSON column)
df = pd.concat([df.drop('json_column', axis=1), df_expanded], axis=1)
```

## Method 2: Using `apply(pd.Series)`

For simple dictionaries:

```python
# Expand JSON column
df_expanded = df['json_column'].apply(pd.Series)

# Combine with original DataFrame
df = pd.concat([df.drop('json_column', axis=1), df_expanded], axis=1)
```

## Method 3: Manual Expansion with `apply()`

For more control:

```python
# Extract specific keys
df['key1'] = df['json_column'].apply(lambda x: x.get('key1') if isinstance(x, dict) else None)
df['key2'] = df['json_column'].apply(lambda x: x.get('key2') if isinstance(x, dict) else None)
```

## Method 4: If JSON is Stored as String

First parse the JSON strings:

```python
import json

# Parse JSON strings
df['json_column'] = df['json_column'].apply(json.loads)

# Then use Method 1 or 2
df_expanded = pd.json_normalize(df['json_column'])
df = pd.concat([df.drop('json_column', axis=1), df_expanded], axis=1)
```

## Complete Example

```python
import pandas as pd
import json

# Sample data with JSON column
data = {
    'id': [1, 2, 3],
    'json_data': [
        '{"name": "Alice", "age": 30, "city": "NYC"}',
        '{"name": "Bob", "age": 25, "city": "LA"}',
        '{"name": "Charlie", "age": 35, "city": "Chicago"}'
    ]
}
df = pd.DataFrame(data)

# Step 1: Parse JSON strings (if needed)
df['json_data'] = df['json_data'].apply(json.loads)

# Step 2: Expand JSON column
df_expanded = pd.json_normalize(df['json_data'])

# Step 3: Combine
df_final = pd.concat([df.drop('json_data', axis=1), df_expanded], axis=1)

print(df_final)
# Output:
#    id     name  age      city
# 0   1    Alice   30       NYC
# 1   2      Bob   25        LA
# 2   3  Charlie   35   Chicago
```

## For Nested JSON

If JSON contains nested dictionaries, `json_normalize()` handles it automatically:

```python
# Nested JSON example
data = {
    'id': [1, 2],
    'json_data': [
        {'name': 'Alice', 'address': {'city': 'NYC', 'zip': '10001'}},
        {'name': 'Bob', 'address': {'city': 'LA', 'zip': '90001'}}
    ]
}
df = pd.DataFrame(data)

# Expand (nested keys become 'address.city', 'address.zip')
df_expanded = pd.json_normalize(df['json_data'])
df = pd.concat([df.drop('json_data', axis=1), df_expanded], axis=1)

# Result: columns are 'name', 'address.city', 'address.zip'
```

## Handling Missing/Invalid JSON

```python
import json

def safe_json_loads(x):
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return {}
    return x if isinstance(x, dict) else {}

df['json_column'] = df['json_column'].apply(safe_json_loads)
df_expanded = pd.json_normalize(df['json_column'])
```

