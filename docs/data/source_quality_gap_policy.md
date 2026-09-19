# Source Quality and Gap Policy

Acquisition success is not defined only by a successful download.

Each source should be checked for:

- valid content type
- expected schema
- timestamp coverage
- duplicate timestamps
- missing intervals
- unexpected HTML/error responses
- source frequency
- provenance metadata

Short gaps may be handled downstream according to the Stage 2 preprocessing
policy, while larger gaps must remain visible rather than being silently
fabricated.
