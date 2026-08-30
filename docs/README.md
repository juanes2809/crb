# Documentación CRB

- [`reporte_consolidado.tex`](reporte_consolidado.tex) →
  [`reporte_consolidado.pdf`](reporte_consolidado.pdf) — **empieza por aquí**.
  Consolida todo lo que se intentó en las cuatro ramas de trabajo y sus
  resultados: la cronología, las tablas numéricas de cada intento, los hallazgos
  de las revisiones, y el estado final (simulador PyTorch, baseline de 32×32,
  CRB por diferencias finitas de 2 parámetros).
- [`sustento_teorico_crb.tex`](sustento_teorico_crb.tex) →
  [`sustento_teorico_crb.pdf`](sustento_teorico_crb.pdf) — sustento teórico del
  CRB de la implementación.
- [`figs_consolidado/`](figs_consolidado/) — figuras de las cuatro ramas,
  conservadas aquí porque esas ramas se retiran. El catálogo con qué muestra
  cada una está en la §6 del reporte consolidado.

Los PDF se compilan con [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd docs
tectonic -X compile reporte_consolidado.tex
tectonic -X compile sustento_teorico_crb.tex
```
