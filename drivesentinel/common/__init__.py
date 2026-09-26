"""
Shared vocabulary for the multi-stage branches.

`schema` defines the one data structure every branch sees; `splits` defines the
group-holdout utilities they are all validated with.  Neither imports torch, so
both are cheap to test and cheap to import from an adapter.
"""
