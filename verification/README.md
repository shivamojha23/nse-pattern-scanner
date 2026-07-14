# Verification Archive

This folder is an archived record of the transition verification scripts (`verify_phase1.py` through `verify_phase5.py`, and `smoke_test.py`) that were used to ensure bit-for-bit accuracy during the incremental refactoring process of Phases 1 through 5.

These scripts compare the outputs of the old monolithic logic against the new modular logic. They rely on legacy database states and pickled `Before_*` and `After_*` data dumps which have since been purged from the repository.

**Note:** These scripts are preserved solely for historical documentation and auditability of the refactoring process. They have been superseded by the permanent, official test suite located in the `/tests/` directory at the repository root.
