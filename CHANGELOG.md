# Changelog

All changes to this project will be documented in this file.

## Conventions

Those emojis will be used in the changelog to highlight the type of change:

- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation update
- ⚠️ Breaking change
- 🔧 Refactor
- 🚀 Performance improvement
- 🧪 Test update
- 🗑️ Removal of deprecated features

## Unreleased

### [0.5.52]

- 🔧: `Iter.repeat_last`, `{Seq, Iter}.find`, `Dict.get_in` to return `Option[T]`
- 🔧: `with_position` to transform Position enum in Literal
- ✨: `booleans` to Seq
- 🗑️: `println` (can be replaced by `tap`)
- ✨: `New tap` method
- 🔧: `peek` methods: deleted old peek and replaced it with peekn (now peek)
- 📝: Various changes to use more `Seq.iter` rather than `Iter.from_` in the docstrings examples.
- 📝: Added new guide section to `docs/guides/`

## Released
