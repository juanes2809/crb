# Plots

- `crb_standard_regions.png` — **CRB(ρ, φ, h)** (se estiman los tres)
- `crb_standard_regions_fixed_h.png` — **CRB(ρ, φ)** (solo esos dos; h fija en la geometría)

```bash
.venv/bin/python regenerate_crb_standard_regions.py --mode both
.venv/bin/python regenerate_crb_standard_regions.py --mode with-h
.venv/bin/python regenerate_crb_standard_regions.py --mode fixed-h
```
