"""STOXX Europe 600 supersector ETFs (iShares, Xetra) -> ISIC (OECD ICIO 2018, 45 sectors) exposure map, weights sum to 1.
Hand-built from the ICB supersector definitions at the industry level; coarse by design (same status as sector_map.py for the US).
Yahoo symbols verified 2026-09-08. Benchmark: iShares STOXX Europe 600 (EXSA.DE)."""
ETF_EXPOSURE_EU = {
    "EXV1.DE": {"64T66": 1.0},                                                     # Banks
    "EXH2.DE": {"64T66": 1.0},                                                     # Financial Services
    "EXH5.DE": {"64T66": 1.0},                                                     # Insurance
    "EXH1.DE": {"05T06": 0.5, "19": 0.5},                                          # Oil & Gas
    "EXV3.DE": {"26": 0.6, "62T63": 0.4},                                          # Technology
    "EXV5.DE": {"29": 1.0},                                                        # Automobiles & Parts
    "EXV6.DE": {"24": 0.6, "07T08": 0.3, "17T18": 0.1},                            # Basic Resources
    "EXV7.DE": {"20": 1.0},                                                        # Chemicals
    "EXH9.DE": {"35": 0.9, "36T39": 0.1},                                          # Utilities
    "EXH4.DE": {"28": 0.30, "27": 0.15, "25": 0.10, "30": 0.15, "69T75": 0.10, "77T82": 0.10, "49": 0.05, "52": 0.05},  # Industrial Goods & Services
    "EXH3.DE": {"10T12": 1.0},                                                     # Food & Beverage
    "EXV4.DE": {"21": 0.6, "86T88": 0.2, "31T33": 0.2},                            # Health Care (pharma, services, medtech)
    "EXH8.DE": {"45T47": 1.0},                                                     # Retail
    "EXV8.DE": {"41T43": 0.5, "23": 0.5},                                          # Construction & Materials
    "EXV2.DE": {"61": 1.0},                                                        # Telecommunications
    "EXV9.DE": {"51": 0.4, "55T56": 0.4, "90T93": 0.2},                            # Travel & Leisure
    "EXH6.DE": {"58T60": 1.0},                                                     # Media
    "EXH7.DE": {"13T15": 0.4, "31T33": 0.2, "10T12": 0.2, "22": 0.2},              # Personal & Household Goods (luxury, durables, tobacco)
    "EXI5.DE": {"68": 1.0},                                                        # Real Estate
}
BENCHMARK_EU = "EXSA.DE"
# economy weights for pan-European events (STOXX 600 country weights, rounded; UK carried separately where an event is UK-only)
EU_WEIGHTS = {"DEU": 0.20, "FRA": 0.22, "GBR": 0.26, "NLD": 0.09, "ITA": 0.06, "ESP": 0.05, "SWE": 0.06, "CHE": 0.06}
