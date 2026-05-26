# Internal scripts

## TODO

- [ ] Add a centralized script to launch all checks (lint, tests, docstrings, site, etc..) and at each fail, stop, and once relaunched, re-start from there until everything is ok.
- [ ] Recheck the reference documentation, see if there's anything missing, if nested folders could help, etc...
- [ ] Add a script to ensure that a given wrapper (e.g `Vec` for `list`) correctly exposes all the methods of the original type, manage rules for false positives (e.g `from_keys` instead of `fromkeys`).
- [ ] Find a way (with AST?) to ensure that we always call C-level code from the `_inner` types.

### Note

Once we fully move to Rust, we don't have ABCs heritance giving us "free" methods.

However macros are fairly simple to implement.

But we must track somewhere what to generate for whom both at the implementation level, at the ABC registration level, and at the
stubs level.

ABCs registration for both pyochain ABCs (meta-classes?) and for standard lib need to be studied to see what we should concretely do here.

Then this step should obviously be kept in sync with the stubs. Orchestrating all of this could become a complex pipeline of transpilation and introspection between typeshed, Rust macros, etc...
