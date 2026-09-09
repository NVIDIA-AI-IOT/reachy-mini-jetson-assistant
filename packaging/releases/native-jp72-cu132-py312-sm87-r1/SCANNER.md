# Approved scanner status

Status: **PENDING**.

An SPDX 2.3 SBOM has been generated locally, but no tool on this Jetson has been identified as NVIDIA’s organization-approved license and security scanner. Running an arbitrary public scanner and labelling it “NVIDIA approved” would not satisfy the release gate.

Availability check on the audited host: `syft`, `grype`, `trivy`, `osv-scanner`, `cdxgen`, `scancode`, `scancode-toolkit`, `blackduck`, `detect`, and `fossology` were not present in `PATH`. This is only a tool-availability observation; it does not identify which scanner or policy NVIDIA OSRB requires.

Before submission, the release owner or OSRB reviewer must provide the approved scanner name, access method, policy/ruleset, and required invocation. Run it against both wheel files, the unpacked wheel trees, and `sbom.spdx.json`; archive its complete unmodified output in this directory.

Record these fields:

- Scanner name:
- Scanner version:
- NVIDIA approval or policy reference:
- Ruleset/database version:
- Invocation:
- Scan timestamp:
- Input wheel SHA-256 values:
- License result:
- Vulnerability result:
- Exceptions/waivers:
- Reviewer:
- Report filename and SHA-256:

The required result is no unreviewed license finding, no vulnerability above the organization’s release threshold, an approved disposition for every exception, and an OSRB reviewer sign-off. Until those fields and the report are present, this bundle is not submission-ready.
