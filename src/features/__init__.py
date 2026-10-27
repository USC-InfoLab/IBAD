"""Dataset adapters that turn raw mobility data into activity documents.

Every adapter returns a :class:`~src.features.documents.Documents`, so adding a
dataset means adding one module here and one entry in ``BUILDERS``.
"""

from src.features import geolife, nhts, numosim
from src.features.documents import Documents

BUILDERS = {
    "nhts": nhts.build_documents,
    "numosim": numosim.build_documents,
    "geolife": geolife.build_documents,
}


def build_documents(dataset: str, cfg: dict) -> Documents:
    """Build the document matrix for ``dataset`` using a template-discovery config."""
    if dataset not in BUILDERS:
        raise KeyError(f"Unknown dataset {dataset!r}; expected one of {list(BUILDERS)}")
    return BUILDERS[dataset](
        cfg["datasets"][dataset], cfg["n_slots"], cfg["slot_minutes"]
    )


__all__ = ["Documents", "BUILDERS", "build_documents"]
