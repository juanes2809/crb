# Plots

- `crb_fd_baseline_32.png` — **CRB(ρ, φ)** por diferencias finitas sobre
  `simulator.py`, con la grilla de referencia de 32×32 píxeles y matriz de
  información de Fisher 2×2. Es la figura de referencia del repositorio.

```bash
python3 compute_crb_fd_baseline.py              # grilla 32×32 (baseline)
python3 compute_crb_fd_baseline.py --force      # ignora la caché y recomputa
python3 compute_crb_fd_baseline.py --cam-pixel-dim 64
```

Los resultados se cachean en `plots/crb_fd_baseline_<N>.pkl` (ignorado por git).

No hay variante con `h`: el simulador no expone la altura del objeto como
entrada, así que σ_h no está definida. Ver §5 de
[`../docs/reporte_consolidado.pdf`](../docs/reporte_consolidado.pdf).

Las figuras históricas de las cuatro ramas de trabajo (incluidas las del CRB de
3 parámetros que ya no es calculable) están conservadas en
[`../docs/figs_consolidado/`](../docs/figs_consolidado/) y comentadas una por
una en el catálogo del reporte consolidado.
