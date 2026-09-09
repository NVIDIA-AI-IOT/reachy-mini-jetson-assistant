# Contributing to Reachy Mini Jetson Assistant

We welcome contributions from the community. To get started, please follow the guidelines below.

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the [Developer Certificate of Origin (DCO)](https://developercertificate.org/):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Signing Your Work

To certify your contribution under the DCO, add a "Signed-off-by" line to each
commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

You can do this automatically with `git commit -s`.

**By signing off, you confirm that you have the right to submit this work under
the Apache 2.0 license used by this project.**

## NVIDIA Maintainer IP Review

Before merging a contribution, NVIDIA maintainers must complete the required internal [IP review process](https://nv/ip_review_process). The review must cover source provenance, applicable licenses, retained copyright and attribution notices, third-party code or assets, dependency changes, and any corresponding-source or redistribution obligations. External contributors are not expected to have access to NVIDIA's internal review system.

## How to Contribute

1. **Fork** the repository and create a feature branch from `main`.
2. **Make your changes** — keep commits focused and well-described.
3. **Test** your changes on the target hardware (Jetson Orin Nano recommended).
4. **Sign off** every commit (`git commit -s`).
5. **Open a pull request** against `main` with a clear description of the change.

## Code Style

- Python 3.10+
- Follow existing code conventions in the repository.
- Add SPDX headers to new files:
  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

## Reporting Issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce (if applicable)
- Hardware and software environment (Jetson model, JetPack version, Python version)

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
