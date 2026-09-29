"""Vendored subset of the FLock validator's ``video_inconsistency`` module.

Copied unchanged from FLock-io/FLock-validator (branch ``feat/video-inconsistency-task``) so
the trainer scripts can generate data with the validator's own synthesiser and calibrate
against the validator's exact scorer. Only the pure-Python pieces are vendored: issue types,
manifest models, the synthesiser, video I/O, output parsing and scoring. The sandbox and
FedLedger plumbing live in the validator repo. See VENDORED.md for the source commit.

Submissions must never import ``validator.*``: the sandbox cannot read it.
"""
