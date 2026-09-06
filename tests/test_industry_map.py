import numpy as np
from emporion.structure.industry_map import INDUSTRY_TO_ICIO, SECTOR_FALLBACK, ICIO_CODES, exposure, exposure_matrix


def test_every_mapping_uses_valid_codes_and_sums_to_one():
    for table in (INDUSTRY_TO_ICIO, SECTOR_FALLBACK):
        for k, w in table.items():
            assert set(w) <= set(ICIO_CODES), k
            assert abs(sum(w.values()) - 1.0) < 1e-9, k


def test_fallback_and_matrix():
    assert exposure("Major Banks") == {"64T66": 1.0}
    assert exposure("  Medicinal Chemicals and Botanical Products ") == {"21": 1.0}
    assert exposure("Unknown thing", "Technology") == {"26": 0.5, "62T63": 0.5}
    assert exposure("Unknown thing", "") == {}
    M = exposure_matrix(["Semiconductors", "Unknown"], ["Technology", ""])
    assert M.shape == (2, len(ICIO_CODES)) and M[0].sum() == 1.0 and M[1].sum() == 0.0
