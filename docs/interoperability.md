# Interoperability & Chaining Guide

Pyochain is designed for fluent API usage. Most types can be converted into others seamlessly, allowing you to chain operations without breaking the flow.

## Shared Features

To enable this flexibility, Pyochain provides two core traits (mixins) that extend all types with powerful chaining and conditional logic capabilities.

### Pipeable trait

All pyochain types implement `Pipeable`, providing universal methods to continue a chain:

#### `.into(func, *args, **kwargs) -> R`

Convert `Self` to any type `R` via a function, maintaining fluent chaining.

Conceptually, this replaces `f(x, args, kwargs)` with `x.into(f, args, kwargs)`.

This is particularly useful when you need to pass the result to a function you don't control (like a library function), or to convert to a type not native to Pyochain.

```python
>>> import pyochain as pc
>>> import json
>>> # Flow is broken, nested function calls, read from middle -> right -> left -> right
>>> json.dumps(dict(pc.Dict({"id": 1, "name": "Alice"}).map_keys(str.upper)))
'{"ID": 1, "NAME": "Alice"}'
>>> # Fluent chaining with .into(), read left -> right
>>> pc.Dict({"id": 1, "name": "Alice"}).map_keys(str.upper).into(lambda d: json.dumps(dict(d)))
'{"ID": 1, "NAME": "Alice"}'
```

#### `.inspect(func, *args, **kwargs) -> Self`

Pass `Self` to a function for side effects (logging, debugging, metrics) without breaking the chain. The instance is returned unchanged.

```python
>>> import pyochain as pc
>>> pc.Seq([1, 2, 3]).inspect(print).iter().map(lambda x: x * 2).collect()
Seq(1, 2, 3)
Seq(2, 4, 6)
```

### Checkable trait

Collections (`Seq`, `Vec`, `Set`, `SetMut`, `Dict`) and iterators (`Iter`) implement `Checkable`, providing conditional chaining based on truthiness (usually emptiness):

#### `.then(func, *args, **kwargs) -> Option[R]`

Call **func** and wrap result in `Some` only if the instance is truthy.

```python
>>> import pyochain as pc
>>> pc.Seq([1, 2, 3]).then(lambda s: s.sum())
Some(6)
>>> pc.Seq([]).then(lambda s: s.sum())
NONE
```

#### `.then_some() -> Option[Self]`

Wrap the instance in `Some` if truthy, otherwise `NONE`.

```python
>>> import pyochain as pc
>>> pc.Seq([1, 2, 3]).then_some()
Some(Seq(1, 2, 3))
>>> pc.Seq([]).then_some()
NONE
```

#### `.ok_or(err) -> Result[Self, E]`

Wrap in `Ok` if truthy, otherwise wrap the error in `Err`.

```python
>>> import pyochain as pc
>>> pc.Seq([1, 2, 3]).ok_or("empty list")
Ok(Seq(1, 2, 3))
>>> pc.Seq([]).ok_or("empty list")
Err('empty list')
```

#### `.ok_or_else(func, *args, **kwargs) -> Result[Self, E]`

Wrap in `Ok` if truthy, otherwise call **func** and wrap result in `Err` (lazy evaluation).

```python
>>> import pyochain as pc
>>> pc.Seq([1, 2, 3]).ok_or_else(lambda _: "empty")
Ok(Seq(1, 2, 3))
>>> pc.Seq([]).ok_or_else(lambda _: "empty")
Err('empty')
```

## Conversion & Interoperability Map

The following graph illustrates all the built-in ways to convert between types in Pyochain.

- Types are grouped by category.
- Arrows color and direction represent conversion paths.
- Arrow labels represent methods.

```mermaid
---
config:
  layout: elk
---
flowchart TB
 subgraph Collections["📦 Collections (Eager)"]
    direction LR
        Seq["<b>Seq[T]</b><br>immutable<br>tuple"]
        Vec["<b>Vec[T]</b><br>mutable<br>list"]
        Set["<b>Set[T]</b><br>immutable<br>frozenset"]
        SetMut["<b>SetMut[T]</b><br>mutable<br>set"]
  end
 subgraph Lazy["⛓️ Lazy"]
    direction LR
        Iter["<b>Iter[T]</b><br>lazy iterator<br>Iterator"]
  end
 subgraph DictGroup["🔑 Dictionary"]
    direction LR
        Dict["<b>Dict[K,V]</b><br>mutable<br>dict"]
  end
 subgraph OptionGroup["🎁 Option Types"]
    direction LR
        Option["<b>Option[T]</b>"]
        Some["<b>Some[T]</b>"]
        NONE["<b>NONE</b>"]
  end
 subgraph ResultGroup["✅ Result Types"]
    direction LR
        Result["<b>Result[T,E]</b>"]
        Ok["<b>Ok[T]</b>"]
        Err["<b>Err[E]</b>"]
  end
 subgraph External["🌐 External Types"]
    direction LR
        AnyType["<b>Any Type</b><br>via .into(func)"]
  end
    Option -.-> Some & NONE
    Result -.-> Ok & Err
    Collections -- ".iter()" --> Lazy
    Lazy -- ".collect()" --> Collections
    Lazy -- ".collect(Dict)" --> DictGroup
    DictGroup -- ".iter() → Item[K,V]<br>.keys_iter() → K<br>.values_iter() → V" --> Lazy
    OptionGroup -- ".iter()" --> Lazy
    ResultGroup -- ".iter()" --> Lazy
    DictGroup -- ".get_item(key)<br>.insert(key, val)<br>.remove(key)" --> OptionGroup
    DictGroup -- ".try_insert(key, val)" --> ResultGroup
    Collections -- ".then(func)<br>.then_some()" --> OptionGroup
    Lazy -- ".then(func)<br>.then_some()" --> OptionGroup
    DictGroup -- ".then(func)<br>.then_some()" --> OptionGroup
    Collections -- ".ok_or(err)<br>.ok_or_else(func)" --> ResultGroup
    Lazy -- ".ok_or(err)<br>.ok_or_else(func)" --> ResultGroup
    DictGroup -- ".ok_or(err)<br>.ok_or_else(func)" --> ResultGroup
    OptionGroup -- ".ok_or(err)<br>.ok_or_else(func)" --> ResultGroup
    ResultGroup -- ".ok()<br>.err()" --> OptionGroup
    OptionGroup L_OptionGroup_ResultGroup_2@<-- ".transpose()" --> ResultGroup
    Collections -- ".into(func)" --> External
    Lazy -- ".into(func)" --> External
    DictGroup -- ".into(func)" --> External
    OptionGroup -- ".into(func)" --> External
    ResultGroup -- ".into(func)" --> External

     Seq:::collectionsStyle
     Vec:::collectionsStyle
     Set:::collectionsStyle
     SetMut:::collectionsStyle
     Iter:::iterStyle
     Dict:::dictStyle
     Option:::optionStyle
     Some:::optionStyle
     NONE:::optionStyle
     Result:::resultStyle
     Ok:::resultStyle
     Err:::resultStyle
     AnyType:::externalStyle
     Collections:::collectionsStyle
     OptionGroup:::optionStyle
     ResultGroup:::resultStyle
    classDef collectionsStyle fill:#1e88e5,stroke:#0d47a1,stroke-width:2px,color:#fff
    classDef iterStyle fill:#43a047,stroke:#1b5e20,stroke-width:2px,color:#fff
    classDef dictStyle fill:#fb8c00,stroke:#e65100,stroke-width:2px,color:#fff
    classDef optionStyle fill:#fdd835,stroke:#f57f17,stroke-width:2px,color:#000
    classDef resultStyle fill:#e53935,stroke:#b71c1c,stroke-width:2px,color:#fff
    classDef externalStyle fill:#9e9e9e,stroke:#424242,stroke-width:2px,color:#fff
    style Seq color:none
    style Vec color:none
    style Set color:none
    style SetMut color:none
    style Dict fill:#FF6D00,color:none,stroke:#FF6D00
    style Option color:#FFFFFF,fill:transparent,stroke:#FFD600
    style Some color:#FFFFFF,fill:transparent,stroke:#FFD600
    style NONE color:#FFFFFF,fill:transparent,stroke:#FFD600
    style Result fill:#D50000
    style Ok fill:#D50000
    style Err fill:#D50000
    style Collections fill:#000000,color:none,stroke:#2962FF
    style Lazy fill:#000000,stroke:#00C853
    style DictGroup fill:#000000,color:none,stroke:#FF6D00
    style OptionGroup fill:#000000,color:#FFFFFF,stroke:#FFD600
    style ResultGroup fill:#000000,stroke:#D50000
    style External fill:#000000
    linkStyle 0 stroke:#666,stroke-width:1px,stroke-dasharray:3,fill:none
    linkStyle 1 stroke:#666,stroke-width:1px,stroke-dasharray:3,fill:none
    linkStyle 2 stroke:#666,stroke-width:1px,stroke-dasharray:3,fill:none
    linkStyle 3 stroke:#666,stroke-width:1px,stroke-dasharray:3,fill:none
    linkStyle 4 stroke:#1e88e5,stroke-width:2.5px,fill:none
    linkStyle 5 stroke:#43a047,stroke-width:2.5px,fill:none
    linkStyle 6 stroke:#43a047,stroke-width:2.5px,fill:none
    linkStyle 7 stroke:#fb8c00,stroke-width:2.5px,fill:none
    linkStyle 8 stroke:#fdd835,stroke-width:2.5px,fill:none
    linkStyle 9 stroke:#e53935,stroke-width:2.5px,fill:none
    linkStyle 10 stroke:#fb8c00,stroke-width:2.5px,fill:none
    linkStyle 11 stroke:#fb8c00,stroke-width:2.5px,fill:none
    linkStyle 12 stroke:#1e88e5,stroke-width:2.5px,fill:none
    linkStyle 13 stroke:#43a047,stroke-width:2.5px,fill:none
    linkStyle 14 stroke:#fb8c00,stroke-width:2.5px,fill:none
    linkStyle 15 stroke:#1e88e5,stroke-width:2.5px,fill:none
    linkStyle 16 stroke:#43a047,stroke-width:2.5px,fill:none
    linkStyle 17 stroke:#fb8c00,stroke-width:2.5px,fill:none
    linkStyle 18 stroke:#fdd835,stroke-width:2.5px,fill:none
    linkStyle 19 stroke:#e53935,stroke-width:2.5px,fill:none
    linkStyle 20 stroke:#9c27b0,stroke-width:2.5px,fill:none
    linkStyle 21 stroke:#1e88e5,stroke-width:2.5px,fill:none
    linkStyle 22 stroke:#43a047,stroke-width:2.5px,fill:none
    linkStyle 23 stroke:#fb8c00,stroke-width:2.5px,fill:none
    linkStyle 24 stroke:#fdd835,stroke-width:2.5px,fill:none
    linkStyle 25 stroke:#e53935,stroke-width:2.5px,fill:none

    L_OptionGroup_ResultGroup_2@{ animation: none }
```
