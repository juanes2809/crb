# Documentación CRB

- [`reporte_consolidado.tex`](reporte_consolidado.tex) →
  [`reporte_consolidado.pdf`](reporte_consolidado.pdf) — **empieza por aquí**.
  Consolida todo lo que se intentó en las cuatro ramas de trabajo y sus
  resultados: la cronología, las tablas numéricas de cada intento, los hallazgos
  de las revisiones, y el estado final (simulador PyTorch, baseline de 32×32,
  CRB por diferencias finitas de 2 parámetros).
- [`forward_completo.tex`](forward_completo.tex) →
  [`forward_completo.pdf`](forward_completo.pdf) — notas didácticas del *forward*
  completo del pipeline actual, de principio a fin: colocación de la faceta,
  radiometría de dos rebotes, oclusión, binning temporal, modelo de Poisson,
  información de Fisher, jacobiano por diferencias finitas, cota de Cramér–Rao y
  elipses de incertidumbre, con la justificación de cada decisión y las fórmulas
  verificadas contra el código.
- [`sustento_teorico_crb.tex`](sustento_teorico_crb.tex) →
  [`sustento_teorico_crb.pdf`](sustento_teorico_crb.pdf) — sustento teórico del
  CRB de la implementación.
- [`figs_consolidado/`](figs_consolidado/) — figuras de las cuatro ramas,
  conservadas aquí porque esas ramas se retiran. El catálogo con qué muestra
  cada una está en la §6 del reporte consolidado.
- [`figs_forward/`](figs_forward/) — figuras auxiliares de
  `forward_completo.pdf`, con el script `make_aux_figs.py` que las regenera
  (`python3 docs/figs_forward/make_aux_figs.py` desde la raíz del repositorio).

Los PDF se compilan con [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd docs
tectonic -X compile reporte_consolidado.tex
tectonic -X compile forward_completo.tex
tectonic -X compile sustento_teorico_crb.tex
```
