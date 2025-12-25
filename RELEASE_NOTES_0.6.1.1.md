# 🚀 pyochain 0.6.1.1 — Release Notes

**Date:** 2025-12-25

## ✨ Summary

This is a minor patch release that adds CI/CD infrastructure for automated documentation deployment and package publishing. No API changes or breaking changes are included in this release.

---

## 🎯 New Features

### 📚 Documentation Deployment Automation

- **GitHub Actions workflow for documentation:** Added automated deployment of documentation to GitHub Pages on every push to master branch
  - Uses MkDocs to build and deploy documentation
  - Automatically triggered on push to master or manual workflow dispatch
  - Deployed to: <https://outsquarecapital.github.io/pyochain/>

### 📦 Package Publishing Automation

- **GitHub Actions workflow for PyPI publishing:** Added automated package publishing to PyPI on release
  - Triggered automatically when a new release is published on GitHub
  - Uses `uv build` and `uv publish` for streamlined publishing
  - Includes proper permissions configuration for secure publishing

---

## 🔧 Improvements

- **CI/CD Infrastructure:** Complete CI/CD setup for documentation and package distribution
- **Development Workflow:** Improved development workflow with automated deployments

---

## 🔄 Breaking Changes

**None** — This is a purely additive release with no API changes.

---

## ⚠️ Migration Guide

No migration needed. This release is fully backward compatible with 0.6.1.0.

---

**View all releases:** [https://github.com/OutSquareCapital/pyochain/releases](https://github.com/OutSquareCapital/pyochain/releases)

---
