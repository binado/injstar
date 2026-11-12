"""BAYESTAR localization.

Packs simulated SNR series, response tensors, locations and horizons into the
arguments of ``ligo.skymap.core.toa_phoa_snr`` and runs one call per detected
event in a worker pool.

Design: "BAYESTAR numerical interface" in docs/design.md; the audited
signature is in docs/audit.md.
Not implemented yet.
"""
