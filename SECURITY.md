# Security Policy

## Supported Versions

mloda is pre-1.0 and ships frequently. Security fixes are applied to the
latest released version on [PyPI](https://pypi.org/project/mloda/) and the
`main` branch. We do not backport fixes to older releases — please upgrade to
the latest version before reporting.

| Version        | Supported          |
| -------------- | ------------------ |
| Latest release | :white_check_mark: |
| Older releases | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.** This keeps users protected while a fix is
prepared.

Instead, use one of these private channels:

1. **GitHub Private Vulnerability Reporting** (preferred) — open the
   [Security tab](https://github.com/mloda-ai/mloda/security/advisories/new)
   and click **Report a vulnerability**. This keeps the report, discussion, and
   coordinated disclosure in one place.
2. **Email** — write to **security@mloda.ai**. Encrypt with our PGP key on
   request if the details are sensitive.

Please include as much of the following as you can:

- The type of issue (e.g. injection, path traversal, deserialization,
  dependency vulnerability).
- The affected version, module, or file path (`file.py:line` if known).
- Step-by-step reproduction or a proof of concept.
- The potential impact and any suggested mitigation.

## What to Expect

mloda is an early-stage project maintained by a small team, so we can't commit
to fixed response times yet. That said:

- We aim to acknowledge reports as soon as we reasonably can.
- We'll assess the severity and keep you updated as we work toward a fix.
- We're happy to credit you in the advisory once it's published, unless you
  prefer to remain anonymous.

We follow coordinated disclosure: we ask that you give us a reasonable window
to release a fix before any public disclosure. Thank you for helping keep mloda
and its users safe.
