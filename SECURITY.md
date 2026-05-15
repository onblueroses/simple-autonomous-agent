# Security Policy

## Supported Versions

This project is pre-1.0. Only the latest released minor version receives
security fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a Vulnerability

Open a GitHub issue labelled `security:` describing the issue in general
terms (no exploit payload in the public issue). The maintainer will reply
with a private channel for details. If the issue is severe, request a
private security advisory directly from the repository's Security tab.

Please do not disclose the vulnerability publicly until a fix has shipped.

## Disclosure Policy

Coordinated disclosure within 90 days of report, or sooner if a fix is
released earlier. Credit will be given in the changelog unless the reporter
requests anonymity.

## Scope

This policy covers the `simple-autonomous-agent` package only. Issues in
upstream dependencies (`openai`, `pyyaml`) should be reported to those
projects directly. Issues in the example scripts under `examples/` are
out of scope; they are illustrative demos, not part of the supported
library surface.
