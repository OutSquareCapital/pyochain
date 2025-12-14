# Replacing While Loops and Stateful Iterations

This guide explores **how to think differently** about problems that traditionally use `while` and `for` loops with mutable state, and how to reformulate them with pyochain.

The goal is not to ban loops, but to **recognize the patterns** where mutable state actually hides a **data transformation**.

---

## The Problem with Mutable State

Let's take a concrete problem: parsing a log file to extract critical errors and calculate the average time between these errors.

### Classic Imperative Approach

```python
from datetime import datetime

def analyze_critical_errors(log_lines: list[str]) -> dict:
    critical_errors = []
    prev_timestamp = None
    time_diffs = []
    
    for line in log_lines:
        if "CRITICAL" in line:
            # Parse timestamp (format: "2024-01-15 10:30:45 CRITICAL ...")
            timestamp_str = line.split()[0] + " " + line.split()[1]
            timestamp = datetime.fromisoformat(timestamp_str)
            
            critical_errors.append(line)
            
            if prev_timestamp is not None:
                time_diffs.append((timestamp - prev_timestamp).total_seconds())
            
            prev_timestamp = timestamp
    
    return {
        "count": len(critical_errors),
        "avg_interval": sum(time_diffs) / len(time_diffs) if time_diffs else 0,
        "errors": critical_errors
    }
```

**Identified Problems:**

1. **Scattered state**: `critical_errors`, `prev_timestamp`, `time_diffs` mutate in different places
2. **Tangled logic**: parsing, filtering, calculating differences, aggregation → all in one loop
3. **Hard to test**: impossible to test individual steps
4. **Fragile**: forget `prev_timestamp = timestamp` and everything breaks silently

### The Real Problem: Confusing Steps

This function does **4 distinct things** that are hidden by mutable state:

1. **Filter** critical lines
2. **Parse** timestamps  
3. **Calculate** intervals between consecutive events
4. **Aggregate** results

Each of these steps is actually a **pure transformation** disguised as mutation.

### Reformulated: Separating Concerns

```python
from datetime import datetime
from collections.abc import Sequence
import pyochain as pc

def parse_timestamp(line: str) -> datetime:
    """Pure function: string → datetime"""
    parts = line.split()
    return datetime.fromisoformat(f"{parts[0]} {parts[1]}")

def calculate_intervals(timestamps: Sequence[datetime]) -> Sequence[float]:
    """Pure function: timestamps → intervals between consecutive events"""
    return (
        pc.Seq(timestamps)
        .iter()
        .adjacent()  # Pairs consecutive elements: [(t0, t1), (t1, t2), ...]
        .map(lambda pair: (pair[1] - pair[0]).total_seconds())
        .collect()
    )

def analyze_critical_errors(log_lines: Sequence[str]) -> dict:
    """Orchestration: compose pure functions into a pipeline"""
    timestamps = (
        pc.Seq(log_lines)
        .iter()
        .filter(lambda line: "CRITICAL" in line)  # Step 1: Filter
        .map(parse_timestamp)                     # Step 2: Parse
        .collect()
    )
    
    intervals = calculate_intervals(timestamps)   # Step 3: Calculate
    
    return {                                      # Step 4: Aggregate
        "count": timestamps.length(),
        "avg_interval": intervals.mean() if intervals.length() > 0 else 0,
        "errors": [line for line in log_lines if "CRITICAL" in line]
    }
```

**What changed:**

- **Zero mutable variables**: No `prev_timestamp`, no `time_diffs = []`
- **Testable steps**: `parse_timestamp` and `calculate_intervals` are pure functions
- **Clear flow**: Read the pipeline top-to-bottom, each line is one concept
- **Resilient**: No hidden state to forget updating

**Key insight:** The `prev_timestamp` pattern is actually **pairwise iteration** over consecutive elements. Use `.adjacent()` instead of maintaining state.

---

## Understanding Stateful Iteration Patterns

Most mutable-state loops fall into a few categories. Understanding **which category** you're in determines **which pyochain tool** to use.

### Category 1: "Look Behind" State (Adjacent Pairs)

**Symptom:** You keep `prev_value` to compare with `current_value`.

**Examples:**

- Detect changes in a sequence
- Calculate differences between consecutive elements  
- Find transitions or inflection points

**Tool:** `.adjacent()` - pairs consecutive elements

**Real-world example: Detecting price changes**

```python
# Imperative: track previous price
def find_price_increases(prices: list[float]) -> list[tuple[float, float]]:
    increases = []
    prev_price = None
    
    for price in prices:
        if prev_price is not None and price > prev_price:
            increases.append((prev_price, price))
        prev_price = price
    
    return increases

# Declarative: pairwise comparison
import pyochain as pc

def find_price_increases(prices: Sequence[float]) -> Sequence[tuple[float, float]]:
    return (
        pc.Seq(prices)
        .iter()
        .adjacent()                                   # [(p0, p1), (p1, p2), ...]
        .filter(lambda pair: pair[1] > pair[0])       # Keep only increases
        .collect()
    )
```

No `prev_price` variable. The pairwise structure makes the comparison explicit.

### Category 2: Cumulative State (Running Totals)

**Symptom:** You maintain `total`, `count`, `sum`, etc. that accumulates across iterations.

**Examples:**

- Running totals
- Cumulative sums
- Balance calculations

**Tool:** `.accumulate()` - applies a binary function cumulatively

**Real-world example: Bank account balance**

```python
from dataclasses import dataclass

@dataclass
class Transaction:
    amount: float
    description: str

# Imperative: track balance
def calculate_balances(transactions: list[Transaction], initial: float) -> list[float]:
    balances = []
    balance = initial
    
    for txn in transactions:
        balance += txn.amount
        balances.append(balance)
    
    return balances

# Declarative: accumulate
import pyochain as pc

def calculate_balances(
    transactions: Sequence[Transaction], 
    initial: float
) -> Sequence[float]:
    return (
        pc.Seq(transactions)
        .iter()
        .map(lambda txn: txn.amount)
        .insert_left(initial)                     # Start with initial balance
        .accumulate(lambda balance, amount: balance + amount)
        .skip(1)                                  # Remove initial value
        .collect()
    )
```

The `balance` mutation is replaced by `.accumulate()` which makes the "running total" pattern explicit.

### Category 3: Complex State Machines

**Symptom:** You maintain multiple state variables that evolve together in complex ways.

**Examples:**

- Fibonacci sequence
- Game state simulation
- Parsing with context

**Tool:** `Iter.unfold()` - generates values from evolving state

**Real-world example: Fibonacci with limits**

```python
# Imperative: multiple state variables
def fibonacci_until(limit: int) -> list[int]:
    result = []
    a, b = 0, 1
    
    while a <= limit:
        result.append(a)
        a, b = b, a + b
    
    return result

# Declarative: unfold state
import pyochain as pc

type FibState = tuple[int, int]

def fib_generator(state: FibState) -> pc.Option[tuple[int, FibState]]:
    a, b = state
    if a > 100:
        return pc.NONE
    return pc.Some((a, (b, a + b)))

def fibonacci_until(limit: int) -> Sequence[int]:
    def generator(state: FibState) -> pc.Option[tuple[int, FibState]]:
        a, b = state
        if a > limit:
            return pc.NONE
        return pc.Some((a, (b, a + b)))
    
    return pc.Iter.unfold(seed=(0, 1), generator=generator).collect()
```

The `while` loop with `a, b = b, a + b` is **state evolution**. `unfold` makes this explicit: given state, produce `(value, next_state)`.

**Another example: Simulating resource consumption**

```python
from dataclasses import dataclass

@dataclass
class Task:
    memory_mb: int
    duration_sec: int

# Imperative: simulate with mutable state
def simulate_memory_usage(tasks: list[Task], total_memory: int) -> list[dict]:
    timeline = []
    available_memory = total_memory
    time = 0
    
    for task in tasks:
        if task.memory_mb > available_memory:
            # Wait for some memory to free up (simplified)
            available_memory = total_memory
        
        available_memory -= task.memory_mb
        time += task.duration_sec
        
        timeline.append({
            "time": time,
            "available": available_memory,
            "task": task.memory_mb
        })
    
    return timeline

# Declarative: unfold simulation state
import pyochain as pc

type SimState = tuple[int, int]  # (available_memory, time)

def simulate_memory_usage(
    tasks: Sequence[Task], 
    total_memory: int
) -> Sequence[dict]:
    def step(state: SimState, task: Task) -> SimState:
        available, time = state
        
        # Reset if not enough memory
        if task.memory_mb > available:
            available = total_memory
        
        return (available - task.memory_mb, time + task.duration_sec)
    
    def make_snapshot(state: SimState, task: Task) -> dict:
        return {
            "time": state[1],
            "available": state[0],
            "task": task.memory_mb
        }
    
    initial_state = (total_memory, 0)
    
    return (
        pc.Seq(tasks)
        .iter()
        .scan(
            initial_state,
            lambda state, task: pc.Some(
                (make_snapshot(new_state := step(state, task), task), new_state)[1]
            )
        )
        .collect()
    )
```

Wait, that's getting complex. Let me show a cleaner pattern using `.map()` with stateful objects when appropriate.

Actually, this reveals an important point: **not all state is bad**. Sometimes a simple loop **is** clearer.

### Category 4: Early Termination

**Symptom:** You `break` out of a loop when a condition is met.

**Examples:**

- "Find first N matching items"
- "Process until threshold"
- "Stop when error encountered"

**Tools:** `.take_while()`, `.scan()` (with `NONE` return)

**Real-world example: API rate limiting**

```python
import time
from dataclasses import dataclass

@dataclass
class APIRequest:
    endpoint: str
    priority: int

# Imperative: process with rate limit
def process_requests(requests: list[APIRequest], max_per_minute: int) -> list[str]:
    processed = []
    count = 0
    
    for req in requests:
        if count >= max_per_minute:
            break
        
        # Make request (simplified)
        result = f"Called {req.endpoint}"
        processed.append(result)
        count += 1
    
    return processed

# Declarative: take while under limit
import pyochain as pc

def process_requests(
    requests: Sequence[APIRequest], 
    max_per_minute: int
) -> Sequence[str]:
    return (
        pc.Seq(requests)
        .iter()
        .take(max_per_minute)                        # Simple limit
        .map(lambda req: f"Called {req.endpoint}")
        .collect()
    )
```

For conditional termination:

```python
# Process high-priority requests until total cost exceeds budget
def process_within_budget(
    requests: Sequence[APIRequest], 
    budget: float
) -> Sequence[str]:
    def under_budget(state: float, req: APIRequest) -> pc.Option[float]:
        cost = req.priority * 0.01  # Simplified cost model
        new_total = state + cost
        
        if new_total > budget:
            return pc.NONE  # Stop iteration
        
        return pc.Some(new_total)
    
    return (
        pc.Seq(requests)
        .iter()
        .scan(0.0, under_budget)                     # Accumulate cost, stop when over
        .map(lambda cost: f"Processed (total: ${cost:.2f})")
        .collect()
    )
```

The `break` statement is replaced by returning `NONE` from the scan function.

---

## Real-World Case Study: ETL Pipeline

Let's build something more substantial: processing CSV files to detect anomalies.

**Problem:** Given sales data CSVs, find days where revenue dropped >20% from the previous day, and calculate recovery time.

### Imperative Approach

```python
import csv
from pathlib import Path
from datetime import datetime

def find_anomalies_imperative(csv_path: Path) -> list[dict]:
    anomalies = []
    prev_revenue = None
    drop_start = None
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            date = datetime.fromisoformat(row["date"])
            revenue = float(row["revenue"])
            
            if prev_revenue is not None:
                drop_pct = (prev_revenue - revenue) / prev_revenue * 100
                
                if drop_pct > 20:
                    # Anomaly detected
                    if drop_start is None:
                        drop_start = (date, prev_revenue)
                elif drop_start is not None:
                    # Recovery
                    recovery_days = (date - drop_start[0]).days
                    anomalies.append({
                        "start_date": drop_start[0],
                        "baseline": drop_start[1],
                        "recovery_days": recovery_days
                    })
                    drop_start = None
            
            prev_revenue = revenue
    
    return anomalies
```

**State tracking:**

- `prev_revenue`: look-behind
- `drop_start`: multi-step state machine
- Nested conditionals mixing detection and recovery logic

### Declarative Approach

```python
import csv
import pyochain as pc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DayData:
    date: datetime
    revenue: float

@dataclass
class Drop:
    date: datetime
    baseline: float
    current: float
    drop_pct: float

def parse_csv(csv_path: Path) -> Sequence[DayData]:
    """Pure: CSV → structured data"""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [
            DayData(
                date=datetime.fromisoformat(row["date"]),
                revenue=float(row["revenue"])
            )
            for row in reader
        ]

def detect_drops(data: Sequence[DayData]) -> Sequence[Drop]:
    """Pure: revenue data → drops"""
    return (
        pc.Seq(data)
        .iter()
        .adjacent()                                    # Pairs of consecutive days
        .map(lambda pair: Drop(
            date=pair[1].date,
            baseline=pair[0].revenue,
            current=pair[1].revenue,
            drop_pct=(pair[0].revenue - pair[1].revenue) / pair[0].revenue * 100
        ))
        .filter(lambda drop: drop.drop_pct > 20)       # Significant drops
        .collect()
    )

def group_into_incidents(drops: Sequence[Drop]) -> Sequence[dict]:
    """Pure: drops → incidents with recovery time"""
    # Group consecutive drops into single incidents
    return (
        pc.Seq(drops)
        .iter()
        .partition_by(lambda drop: drop.drop_pct > 20) # Split on non-drops
        .filter(lambda group: pc.Seq(group).length() > 0)
        .map(lambda group: {
            "start_date": pc.Seq(group).first().date,
            "baseline": pc.Seq(group).first().baseline,
            "recovery_days": len(group)  # Simplified: each drop = 1 day
        })
        .collect()
    )

def find_anomalies(csv_path: Path) -> Sequence[dict]:
    """Orchestration: compose the pipeline"""
    return (
        parse_csv(csv_path)
        .pipe(detect_drops)
        .pipe(group_into_incidents)
    )
```

**Benefits:**

1. **Each function does one thing**: parse, detect, group
2. **Testable in isolation**: can test `detect_drops` with mock data
3. **No hidden state**: all state is in function parameters or return values
4. **Composable**: can reuse `detect_drops` for other analyses

---

## When to Use Loops vs Pyochain

Not every loop needs pyochain. Use this decision tree:

### Use pyochain when

1. **Data transformation pipeline**: filtering, mapping, grouping, aggregating
2. **Clear stages**: each step is a distinct transformation
3. **Reusability**: steps might be useful elsewhere
4. **Complex logic**: easier to test small pure functions
5. **Lazy evaluation helps**: working with large or infinite sequences

### Use plain loops when

1. **Heavy external mutation**: updating complex objects, I/O operations
2. **Simple one-offs**: trivial iterations that won't be maintained
3. **Performance critical**: profiling shows chaining overhead matters
4. **Genuinely clearer**: when forced functional style obscures intent

**Example from the wild** (from `EXAMPLES.md`):

```python
# Building a string with external context
def generate_literal() -> None:
    literal_content: str = Text.CONTENT
    for name in get_palettes().iter_keys().sort().unwrap():
        literal_content += f'    "{name}",\n'
    literal_content += Text.END_CONTENT
    # ... write to file
```

This is fine. The loop is simple, the mutation is local and obvious, and forcing it into `.reduce()` would be less readable.

---

## Summary: Recognizing Patterns

When you see mutable state in a loop, ask:

| Pattern | State Variables | Pyochain Tool |
|---------|----------------|---------------|
| **Look-behind** | `prev_value` | `.adjacent()` |
| **Running total** | `sum`, `count`, `total` | `.accumulate()` |
| **State machine** | Multiple variables evolving together | `Iter.unfold()` |
| **Conditional accumulation** | State + early termination | `.scan()` |
| **Limit iterations** | `if count > N: break` | `.take()`, `.take_while()` |
| **Process pairs/groups** | Complex index math | `.windows()`, `.partition()`, `.batch()` |

The key mental shift: **Don't think about how to maintain state. Think about what transformation you're describing.**
