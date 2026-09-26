"""
Dataset adapters: the only code allowed to know a dataset's quirks.

Each adapter turns files on disk into `common.schema.Recording` objects and
records, in `Recording.provenance`, every decision it had to INFER rather than
read. That provenance is what `docs/data_notes_<id>.md` is generated from, and it
is the difference between "the adapter corrected the polarity" and "the adapter
did something to the signs".
"""
