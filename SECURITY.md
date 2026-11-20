# Security Policy

## Supported Versions

Security updates and bug fixes are actively provided for the following versions of `al3x38-intent-guard`:

| Version | Supported          |
| ------- | ------------------ |
| 1.1.3   | :white_check_mark: |
| 1.1.x   | :white_check_mark: |
| < 1.1.0 | :x:                |

---

## Reporting a Vulnerability

Because `al3x38-intent-guard` is designed as a behavioral guardrail and security verification layer for LLM agents, we take potential bypasses, prompt-injection vectors, and core vulnerabilities seriously.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them privately using one of the following methods:
1. **GitHub Private Vulnerability Reporting:** Use the **"Report a vulnerability"** button under the **Security** tab of this repository (`[https://github.com](https://github.com/alexirk97-ship-it/al3x38-intent-guard)`).
2. **Direct Message (VK):** Reach out directly to the maintainer (`Alex`) via private messages on VK: `https://vk.com/al3x38`


### What to Include in Your Report
* A clear description of the vulnerability or guardrail bypass vector.
* Steps to reproduce (minimal Python snippet, input prompt, or vector configuration).
* The version of `al3x38-intent-guard` and Python environment used.

### Response Timeline
* **Acknowledgment:** Within 48 hours of receiving the report.
* **Fix & Release:** Critical bypasses or security flaws will be patched and published to PyPI as a priority update.
