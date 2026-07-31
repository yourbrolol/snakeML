# Contributing

First of all, thank you for considering contributing to this project! Every contribution, whether it's fixing a typo, reporting a bug, or implementing a new feature, is appreciated.

## Getting Started

1. Fork the repository.
2. Clone your fork.

```bash
git clone https://github.com/yourbrolol/snakeML.git
cd snakeML
```

3. Create a new branch.

```bash
git checkout -b feature/my-feature
```

4. Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Development

### Code Style

- Follow PEP 8.
- Use meaningful variable and function names.
- Keep functions reasonably small.
- Add docstrings to public APIs.
- Avoid unnecessary dependencies.

### Testing

Before submitting a pull request, make sure all tests pass.

```bash
python -m unittest discover
```

If you add new functionality, please add corresponding tests whenever possible.

---

## Pull Requests

Before opening a pull request:

- Run all tests.
- Update documentation if necessary.
- Keep commits focused and descriptive.

A good pull request should explain:

- What changed
- Why it changed
- Any limitations or breaking changes

---

## Reporting Bugs

When reporting a bug, please include:

- Python version
- Operating system
- Project version or commit hash
- Steps to reproduce
- Expected behavior
- Actual behavior
- Full traceback (if applicable)

---

## Feature Requests

Feature requests are welcome!

Please explain:

- The problem you're trying to solve
- Your proposed solution
- Possible alternatives

---

## Commit Messages

Use clear commit messages and conventional commits.

Example:

```
feat(transformers): implement Attention().
Implement Attention() class including forward and backward.
```

In general:

```
type(scope(s)-through a comma): thing-done; other-things-done.
Optional detailed description.
Optional footer.
```

---

## Documentation

If you modify the public API, please update the documentation and examples accordingly.

---

## Questions

If you're unsure about a contribution, feel free to open an issue before starting work.

---

## Looking for something to work on?

Check out `tasks.md` for features, improvements, and bugs that need attention.

---

Thank you for helping improve SnakeML!

The maintainer is currently busy with other projects, so responses may occasionally take some time. Pull requests are always welcome and will be reviewed and merged whenever possible.