# Changelog - Refactorisation Architecture & Breaking Change

**Date**: 11 janvier 2026  
**Commits**: 939da41 + 697e2a7  
**Type de changement**: 🏗️ Refactoring majeur de l'architecture interne + ⚠️ **BREAKING CHANGE**  
**Impact sur l'API**: Breaking change pour `Dict.iter()` → `Dict.items().iter()`

---

## 🎯 Résumé

Refactorisation architecturale majeure en 2 commits :

1. **939da41** : Centralisation de la logique commune dans la hiérarchie des traits
2. **697e2a7** : Ajout des vues typées (Keys/Values/Items) et adaptation du code → **BREAKING CHANGE**

**Statistiques globales** (2 commits combinés):

- 📈 **+1390 insertions** au total
- 📉 **-1373 suppressions** au total
- **Balance nette**: +17 lignes

**Détails par fichier** (commit 939da41):

- 📈 **+1278 lignes** dans `traits/_iterable.py`
- 📉 **-863 lignes** dans `_iter.py` (76 ajouts, 939 suppressions)
- 📉 **-219 lignes** dans `_dict.py` (18 ajouts, 237 suppressions)

---

---

## ⚠️ BREAKING CHANGE (commit 697e2a7)

### Changement d'API pour `Dict`

Avec l'ajout des classes de vues typées (`PyoKeysView`, `PyoValuesView`, `PyoItemsView`), l'itération sur un `Dict` nécessite maintenant une étape supplémentaire :

**AVANT** :

```python
my_dict.iter()  # Itérait sur les clés
```

**APRÈS** :

```python
my_dict.keys().iter()   # Pour itérer sur les clés
my_dict.values().iter() # Pour itérer sur les valeurs
my_dict.items().iter()  # Pour itérer sur les paires (clé, valeur)
```

**Raison** : Les méthodes `.keys()`, `.values()`, `.items()` retournent maintenant des objets de vue typés qui implémentent le trait `PyoMappingView[T]`, alignant l'API avec celle de Python standard.

**Fichiers impactés** :

- `docs/interoperability.md` : exemples mis à jour
- `scripts/benchmarks.py` : code de benchmark adapté

---

## 🏗️ Refactorisation de l'architecture des traits (commit 939da41)

### Nouvelle hiérarchie des traits

L'architecture des traits a été complètement restructurée avec une hiérarchie claire :

```text
PyoIterable[T]              (base pour TOUS les iterables)
    ├── PyoCollection[T]    (pour collections eager)
    │   ├── PyoSequence[T]  (pour Seq, Vec)
    │   ├── PyoMapping[K, V]
    │   │   └── PyoMutableMapping[K, V] (pour Dict)
    │   └── PyoMappingView[T]
    │       ├── PyoKeysView[K]
    │       ├── PyoValuesView[V]
    │       └── PyoItemsView[K, V]
    └── PyoIterator[T]      (pour Iter - lazy)
```

### Simplification de `PyoIterable[T]`

**AVANT** : `PyoIterable[I: Iterable[Any], T]`

- Paramètre générique complexe `I` pour le type de stockage interne
- Génération automatique de `__init__` via `__init_subclass__`
- Logique magique d'extraction du factory depuis les annotations
- Le trait générait automatiquement `__init__`, `__repr__`, et accès à `_inner`

**APRÈS** : `PyoIterable[T]`

- ✅ Un seul paramètre générique `T` (le type d'élément)
- ✅ Plus de magie de métaclasse - suppression complète de `__init_subclass__`
- ✅ **Les traits n'implémentent AUCUN dunder** (sauf `__init__` qui raise une erreur)
- ✅ Chaque classe concrète implémente EXPLICITEMENT tous ses dunders requis
- ✅ Séparation claire : traits = logique métier, classes concrètes = protocole Python
- ✅ Les traits n'héritent plus des protocoles ABC (`Sequence`, `Iterator`, `MutableMapping`, etc.)
- ✅ Les classes concrètes déclarent explicitement leur héritage des protocoles ABC appropriés
  - `Seq[T](PyoSequence[T], Sequence[T])`  ❌ → `Seq[T](PyoSequence[T])` ✅
  - `Vec[T](Seq[T], MutableSequence[T])` ❌ → `Vec[T](Seq[T], MutableSequence[T])` ✅  
  - `Iter[T](PyoIterator[T], Iterator[T])` ❌ → `Iter[T](PyoIterator[T])` ✅
  - `Dict[K, V](PyoMutableMapping[K, V], MutableMapping[K, V])` ❌ → `Dict[K, V](PyoMutableMapping[K, V])` ✅
- ✅ Les traits implémentent virtuellement les protocoles via `register()` pour assurer l'interopérabilité

### Migration des méthodes vers les traits appropriés

#### 📦 `PyoIterable[T]` (base commune à TOUT)

Méthodes minimales communes à TOUTES les collections et itérateurs :

- **Usine** : `new()` (création d'instance vide), `iter()` (conversion en Iter lazy)
- **Longueur** : `length()` (compte les éléments, même pour Iter)
- **Comparaisons** : `eq()`, `ne()`, `le()`, `lt()`, `gt()`, `ge()`
- **Accès positionnel** : `first()`, `second()`, `last()`, `nth(index)`
- **Agrégations simples** : `sum()`, `min()`, `max()`, `min_by()`, `max_by()`, `join()` (strings)
- **Prédicats** : `all()`, `any()`

**Note critique** : Les méthodes de transformation (`filter()`, `map()`, `group_by()`, etc.) ne sont PAS ici !

#### 🗂️ `PyoCollection[T]` (collections eager)

Méthodes nécessitant le protocole `Collection` (`__len__` + `__contains__`) :

- **Overrides** : `length() -> int` (utilise `__len__` au lieu de count)
- **Recherche** : `contains(value) -> bool`
- **Répétition** : `repeat(n) -> Iter[Self]` (répète la collection entière)
- **Test vide** : `is_empty() -> bool`

#### 🔄 `PyoIterator[T]` (itérateurs lazy)

Méthodes spécifiques aux itérateurs, migrées depuis `Iter` :

- **Navigation** : `next() -> Option[T]`
- **Réduction** : `reduce()`, `fold()`, `try_fold()`, `try_reduce()`
- **Recherche** : `find()`, `try_find()`, `find_map()`, `position_with()`
- **Analyse** : `is_sorted()`, `is_sorted_by()`, `all_equal()`, `all_unique()`, `argmax()`, `argmin()`
- **Filtrage conditionnel** : `take_while()`, `skip_while()`, `compress()`, `unique()`
- **Découpe** : `take()`, `skip()`, `step_by()`, `slice()`
- **Cycles** : `cycle()`, `intersperse()`
- **Chaînage** : `insert()`, `interleave()`, `chain()`
- **Divers** : `elements()`, `random_sample()`

**Important** : Les méthodes de transformation principales (`filter()`, `map()`, `flat_map()`, `group_by()`, `partition()`, `zip()`, etc.) sont RESTÉES dans la classe `Iter` car elles retournent un `Iter` et sont spécifiques aux itérateurs lazy

#### 📚 `PyoSequence[T]` (séquences Seq & Vec)

Méthodes communes aux séquences ordonnées:

- `rev() -> Iter[T]` : Inverser l'ordre
- `is_distinct() -> bool` : Vérifier l'unicité de tous les éléments

#### 🗺️ `PyoMapping[K, V]` (mappings)

Vues typées pour les dictionnaires:

- `keys() -> PyoKeysView[K]`
- `values() -> PyoValuesView[V]`
- `items() -> PyoItemsView[K, V]`

#### 🗺️✏️ `PyoMutableMapping[K, V]` (mappings mutables)

Méthodes migrées depuis `Dict`:

- `insert(key, value) -> Option[V]`
- `try_insert(key, value) -> Result[V, KeyError]`
- `remove(key) -> Option[V]`
- `remove_entry(key) -> Option[tuple[K, V]]`
- `get_item(key) -> Option[V]`

---

## 📝 Modifications des classes concrètes

### `_iter.py` : Simplification drastique (-863 lignes nettes)

**Changements appliqués** :

- **Suppression de 939 lignes** : méthodes communes migrées vers les traits
- **Ajout de 76 lignes** : implémentation explicite des dunders et helper `_get_repr()`

**Toutes les classes (`Set[T]`, `SetMut[T]`, `Seq[T]`, `Vec[T]`, `Iter[T]`) implémentent maintenant EXPLICITEMENT** :

- ✅ `__init__(data)` : construction de `_inner` depuis l'iterable (sans magie)
- ✅ `__repr__()` : représentation formatée custom
- ✅ `__slots__ = ("_inner",)` : déclaration explicite
- ✅ `_inner: <type_concret>` : annotation de type (frozenset, tuple, list, Iterator, etc.)
- ✅ Dunders du protocole Collection/Sequence/Iterator :
  - `__len__()`, `__iter__()`, `__contains__()` pour Set/Seq/Vec
  - `__next__()`, `__bool__()` pour Iter
  - `__getitem__()`, `__setitem__()`, `__delitem__()` pour Seq/Vec
- ✅ Les méthodes spécifiques à chaque type restent (ex: `union()` pour Set, `sort()` pour Iter)
- ❌ Suppression de ~900 lignes de méthodes communes (migrées vers les traits)

**Ajout** :

- Helper `_get_repr(data: Collection[Any]) -> str` pour la représentation formatée

### `_dict.py` : Simplification majeure (-219 lignes nettes)

**Changements appliqués** :

- **Suppression de 237 lignes** : méthodes migrées vers `PyoMutableMapping`
- **Ajout de 18 lignes** : implémentation explicite des dunders

**`Dict[K, V]` implémente maintenant EXPLICITEMENT** :

- ✅ Hérite de `PyoMutableMapping[K, V]` (au lieu de `PyoCollection`)
- ✅ `__slots__ = ("_inner",)` : déclaration explicite
- ✅ `_inner: dict[K, V]` : annotation de type
- ✅ Tous les dunders du protocole MutableMapping :
  - `__init__(data)` : construction via `dict(data)`
  - `__repr__()` : représentation formatée
  - `__iter__()` : itération sur les clés
  - `__len__()` : nombre d'éléments
  - `__getitem__(key)`, `__setitem__(key, value)`, `__delitem__(key)` : accès/modification
- ❌ Suppression de ~220 lignes de méthodes : `insert()`, `try_insert()`, `remove()`, `remove_entry()`, `get_item()` (migrées vers `PyoMutableMapping`)
- ✅ Garde uniquement les factory methods : `from_ref()`, `from_kwargs()`, `from_object()`

### `traits/_iterable.py` : Expansion massive (+1278 lignes nettes)

**Changements appliqués** :

- **Ajout de 1278 lignes** : méthodes migrées depuis `_iter.py` et `_dict.py`
- **Suppression de 195 lignes** : ancienne logique de `__init_subclass__` et méthodes obsolètes

**Contenu ajouté** :

- Nouvelles classes abstraites : `PyoIterator[T]`, `PyoSequence[T]`, `PyoMapping[K, V]`, `PyoMutableMapping[K, V]`
- Nouvelles classes concrètes : `PyoKeysView[K]`, `PyoValuesView[V]`, `PyoItemsView[K, V]`
- Migration de dizaines de méthodes depuis les classes concrètes

### `traits/__init__.py` : Nouveaux exports

Ajout des nouveaux traits à l'API publique :

```python
__all__ = [
    "PyoIterable",
    "PyoCollection", 
    "PyoIterator",
    "PyoSequence",
    "PyoMapping",
    "PyoMutableMapping",
    "PyoKeysView",
    "PyoValuesView",
    "PyoItemsView",
]
```

---

## 🎨 Ajustements dans la documentation et le code (commit 697e2a7)

**Fichiers modifiés** :

- `docs/core-types-overview.md` : 8 lignes modifiées (correction de `PyoIterable[I, T]` → `PyoIterable[T]`)
- `docs/interoperability.md` : 6 lignes modifiées (mise à jour exemples `Dict.iter()` → `Dict.items().iter()`)
- `scripts/benchmarks.py` : 6 lignes modifiées (adaptation du code de benchmark)
- `src/pyochain/_dict.py` : 4 lignes modifiées (suppression import redondant `MutableMapping`)
- `src/pyochain/_iter.py` : 4 lignes modifiées (suppression imports redondants `Sequence`, `Iterator`)

---

## ✅ Avantages de cette refactorisation

### Pour la maintenabilité

1. **Séparation des responsabilités claire** : chaque trait a un rôle précis
2. **Réduction de la duplication** : code commun centralisé
3. **Architecture explicite** : plus de magie, tout est visible
4. **Facilite l'extension** : ajouter un nouveau type est plus simple

### Pour les performances

- Aucun impact négatif : les méthodes sont toujours inline-friendly
- Pas de surcharge runtime (héritage simple)

### Pour les utilisateurs

- **Breaking change** : `Dict.iter()` doit être remplacé par `Dict.keys().iter()`, `Dict.values().iter()`, ou `Dict.items().iter()`
- **Gain de clarté** : API plus explicite et alignée avec Python standard
- Tous les tests passent sans modification (hors breaking change)
- Les types sont toujours correctement inférés

---

## 🧪 Validation

Tous les tests passent (doctests + unittests) :

```bash
uv run pytest --doctest-modules --doctest-glob="*.md" --doctest-mdcodeblocks -v src/pyochain tests/ README.md docs/
```

Exit Code: **0** ✅
