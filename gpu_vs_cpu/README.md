# CPU (numpy original) vs GPU (`simulator.py`): las funciones a suavizar, lado a lado

Pose: ρ = 1.00 m, φ = 60° (x = 0.5000, y = 0.8660), `facet.obj`, w = 0.5,
N = 32, FOV = 0.25 m, Δt = 3.9e-10 s (cΔt = 11.69 cm), T = 95 bins,
32768 triángulos.  Figuras generadas por `plot_gpu_vs_cpu.py` a partir de las réplicas de
`plot_simulator_components.py` (`--backend cpu|gpu`).  Los mapas por píxel usan el centroide de la faceta como
"triángulo representativo"; el transitorio usa la malla completa.

**Sanity check** de la réplica GPU: `max|réplica − simulator.simulation(...)|` = **5.55e-17**
(relativo 3.1e-16; torch CPU float64, `hide_walls=True`, `add_noise=False`, con el mismo
`roll(+1, axis=1)` de `orient_transient_measurement`).  La réplica numpy y el simulador real son la misma función.

## Funciones a suavizar

| Función dura | CPU original | GPU `simulator.py` | ¿Igual? | Medido (φ=60°) |
|---|---|---|---|---|
| **ceil-binning** `arrival_bin = ceil((d1+d2)/(cΔt))` | `np.ceil(...).astype(int)`, sin bounds | `torch.ceil(...).long()`, `valid = noc & (bin>0) & (bin≤T)` | **Misma función.** Sólo la GPU descarta bins fuera de rango (aquí 0 descartados: bins 17…34 de 95). | píxeles con bin distinto: **0**; max\|Δdistance\| = 2.7e-15 m |
| **Heaviside-oclusión** `noc = 1{xint>0}` | `m = Δy/Δx` sin máscara; `xint>0` | `m = where(\|Δx\|>eps, Δy/Δx, nan)`; `noc = isfinite(xint) & (xint>0)` | **Misma función** (Heaviside en `xint=0`). La GPU sólo enmascara `\|Δx\|≤eps` (aquí 0 px). | píxeles que cambian de estado: **0** (802 vs 802 px ven la faceta); max\|Δxint\| = 3.6e-16 m |
| **clamps** `dot_k = max(0, cos_k)` | `np.maximum(0, ·)` | `torch.clamp(min=0)` | **Misma función.** Lo que cambia es el *argumento*: la normal de la faceta (theta). | max\|Δdot2\| = **3.568e-03**, max\|Δdot4\| = **1.0e-15** (φ=60°); max\|Δdot2\| = **2.475e-02**, max\|Δdot4\| = 3.3e-16 (φ=30°) |

Por qué `xint`, `noc`, `arrival_bin`, `d2` y `dot4` coinciden exactamente: `theta` rota la faceta alrededor de su
propio centro (la malla está centrada en el eje z antes de trasladarla a `v1`), así que el centroide es el mismo en
ambos backends y todo lo que sólo depende de la geometría píxel–centroide es idéntico.  La diferencia aparece en lo
que depende de la **normal** (`dot1`, `dot2`) y, en la malla completa, en la posición de cada triángulo individual.

## Diferencias de implementación

| # | Aspecto | CPU original | GPU `simulator.py` | Cifra medida |
|---|---|---|---|---|
| 1 | `theta` (rotación en z) | `theta = -cos(∠(u, v1)) = -cos φ` **usado como ángulo** (número en [-1,0] rad) | `theta = φ + 3π/2` (ángulo geométrico) | φ=60°: theta CPU = -0.5000 rad vs GPU ≡ -30.0°; normal CPU (-0.479, -0.878, +0.001) (error **1.35°**) vs GPU (-0.500, -0.866, +0.001) (error 0.05°), esperada (-0.500, -0.866, +0.000). φ=30°: normal CPU (-0.762, -0.648, +0.001) (error **10.38°**) vs GPU (-0.866, -0.500, +0.001) (error 0.05°). dot1: 0.7830 vs 0.7832 (60°), 0.7704 vs 0.7832 (30°); dot3 igual (0.6211 / 0.6211). |
| 2 | Índice de píxel | `coord = (bin-1)N² + (jx **- 1**)N + iy` → imagen corrida 1 columna a la izquierda; jx=0 envuelve a la columna N-1 del bin anterior (196203 depósitos) | `coord = (bin-1)·N² + jx·N + iy` (sin `-1`) | desplazamiento medido (mínimo max\|Δ\| sobre columnas comunes): `y_cpu[:, jx] ≈ y_gpu_raw[:, jx+1]` (**+1 col.**, residuo 3.6e-03); CPU-corregido vs GPU-raw: +0 col. (residuo 3.6e-03, sólo por theta; max\|Δ max_t y\| = 1.742e-03) |
| 3 | Acumulación | `y[coord] += intensity` (con índices repetidos se queda el último valor) | `index_add_` (suma todos) | en el original los índices no se repiten dentro de un triángulo (0 duplicados; energía perdida 4.9e-16); el bug es latente y sólo muerde al vectorizar sobre triángulos, que es lo que hace la GPU |
| 4 | Máscaras `eps` | ninguna: `Δx=0` → `m=±inf`, `xint=-0.0` → píxel ocluido en silencio | `\|Δx\|>eps`, `clamp(d1s,d2s ≥ eps)`, `isfinite(xint)` | píxeles afectados en esta pose: 0 |
| 5 | Bounds de bin | ninguno (un objeto fuera de bounds → `IndexError` o envuelve) | `0 < bin ≤ T` | descartados en esta pose: 0 (bins 17…34 de 95) |
| 6 | `roll` final | ninguno (la variante de `utils/crb_fix.py` hace `roll(-1, axis=-1)` en t) | `orient_transient_measurement`: `np.roll(y, +1, axis=1)` en jx | la salida GPU final queda corrida **una columna a la derecha** del índice correcto (`y_cpu_corregido[:, jx] ≈ y_gpu[:, jx+1]`); frente al original CPU (corrido a la izquierda) el desplazamiento total es **+2 columnas** (residuo 3.6e-03). Nota: la correlación cruzada de perfiles de columna (usada en el review CPU) da aquí +1 porque la imagen es casi constante por columnas; la medida por mínimo max\|Δ\| es la fiable. |

Transitorio: `max_t y` difiere entre CPU-original y GPU-final en max\|Δ\| = 1.778e-01 (máximos 0.1791 vs
0.1778); energía total 722.6971 vs 723.1124 (Δ = +0.06 %, por la
orientación de la faceta).  Comparando lo comparable (CPU corregido sin roll vs GPU sin roll) la diferencia
píxel-a-píxel-a-bin es max\|Δy\| = 3.623e-03, atribuible sólo al `theta`.
Histograma del píxel central (iy=16, jx=24): max\|Δ\| = 5.968e-03 (mismo índice, pero por los
desplazamientos de columna opuestos corresponde a píxeles físicos distintos).

Tiempos en esta máquina (sin GPU): bucle numpy por triángulo 9.9 s (CPU original) /
5.9 s (réplica GPU), `simulator.simulation` vectorizado en torch-CPU 1.7 s.

## Figuras

- `binning.png` — `arrival_bin` CPU | GPU | Δ y el corte 1D de la escalera del `ceil` superpuesto.
- `occlusion.png` — `xint` y `noc` CPU | GPU | Δ, con los píxeles que cambian de estado.
- `clamps.png` — `dot2` y `dot4` con clamp, filas 1-2 φ=60°, filas 3-4 φ=30°, con normales y `dot1`, `dot3` anotados.
- `transient.png` — `max_t y` CPU | GPU | Δ, histograma del píxel central y perfiles por columna con los lags medidos.
- `hard_functions.png` — `max(0,x)`, `ceil(x)`, `1{x>0}`: idénticas en ambos backends.

## Conclusión

Las tres no-linealidades duras que habría que suavizar para un forward diferenciable (`ceil` del binning,
Heaviside de la oclusión, `clamp` de los cosenos) son **exactamente las mismas funciones** en los dos simuladores;
la GPU no cambia la física, sólo dónde y cómo se evalúa.  Lo que sí cambia son los bugs: la GPU corrige el `theta`
(normal de la faceta bien orientada: 0.05° de error frente a 1.35° / 10.38° de la CPU en
φ=60°/30°), el índice `-1`, la acumulación con `index_add_`, y añade máscaras `eps` y bounds de bin, a costa de un
`roll(+1)` final de orientación que hay que tener en cuenta al comparar con la salida original.  Es además mucho más
rápida (vectorizada por chunks de triángulos; ~60× en GPU según la medición del repositorio, y aquí
6× incluso en torch-CPU).  **El simulador GPU es el preferible**: misma
física, bugs corregidos, y es la única implementación cuya salida coincide con la réplica al nivel de redondeo
(5.6e-17).

## Suavizado del binning: gaussiana vs super-gaussiana

Script: `plot_binning_smoothing.py`.  Pulso de orden p, `s_p(u) ∝ exp(−(u²/2)^p)`, `u=(t−t0)/τ`; peso por bin `S_j = F_p(u_hi) − F_p(u_lo)` con `F_p(u) = ½ + ½·sign(u)·P(1/(2p), (u²/2)^p)`; p=1 es la gaussiana actual (`P(½,u²/2) = erf(|u|/√2)`), p=2 la super-gaussiana de cima plana.  Comparación con el mismo τ y con la misma std (`std_2 = 0.822·τ_2`).  Geometría real de la faceta (ρ=1, φ=60°, N=32), bin central j=23.

Verificaciones: p=1 vía `gammainc` vs `erf`: max|Δ| = 2.2e-15; Σ_j S_j = 1 con max|Σ−1| = 1.1e-16 para ambos pulsos y los tres τ; con τ→0 (Δt/50, Δt/200) L1→0 para ambos (ver `pulse_comparison.txt`).

**Bin promedio** Σ_j j·S_j(t0) sobre los píxeles reales (τ = Δt/√12), RMS frente a la diagonal centrada t0/Δt+½: escalera dura 0.2960 (1/√12); gaussiana (p=1) 0.0434; super-gauss p=2, mismo τ 0.0572; super-gauss p=2, misma std 0.0175.

| pulso | τ base | criterio | τ_p/Δt | std/Δt | L1 [Δt] | L∞ | max\|dS_j/dt0\| [1/Δt] | Σ_j S_j min | Σ_j S_j max |
|---|---|---|---|---|---|---|---|---|---|
| p=1 | Δt/√12 | — | 0.2887 | 0.2887 | 0.4606 | 0.5003 | 1.3786 | 1.0000000000 | 1.0000000000 |
| p=2 | Δt/√12 | mismo τ | 0.2887 | 0.2373 | 0.3992 | 0.5000 | 1.3512 | 1.0000000000 | 1.0000000000 |
| p=2 | Δt/√12 | misma std | 0.3511 | 0.2887 | 0.4855 | 0.5000 | 1.1109 | 1.0000000000 | 1.0000000000 |
| p=1 | Δt/3 | — | 0.3333 | 0.3333 | 0.5314 | 0.5013 | 1.1841 | 1.0000000000 | 1.0000000000 |
| p=2 | Δt/3 | mismo τ | 0.3333 | 0.2741 | 0.4609 | 0.5000 | 1.1702 | 1.0000000000 | 1.0000000000 |
| p=2 | Δt/3 | misma std | 0.4054 | 0.3333 | 0.5606 | 0.5000 | 0.9621 | 1.0000000000 | 1.0000000000 |
| p=1 | Δt | — | 1.0000 | 1.0000 | 1.2625 | 0.6587 | 0.2229 | 1.0000000000 | 1.0000000000 |
| p=2 | Δt | mismo τ | 1.0000 | 0.8222 | 1.2324 | 0.6282 | 0.3239 | 1.0000000000 | 1.0000000000 |
| p=2 | Δt | misma std | 1.2163 | 1.0000 | 1.3634 | 0.6864 | 0.2367 | 1.0000000000 | 1.0000000000 |

Cocientes SG/gauss: τ=Δt/√12, mismo τ: L1 ×0.867, max|dS/dt0| ×0.980; τ=Δt/√12, misma std: L1 ×1.054, max|dS/dt0| ×0.806; τ=Δt/3, mismo τ: L1 ×0.867, max|dS/dt0| ×0.988; τ=Δt/3, misma std: L1 ×1.055, max|dS/dt0| ×0.812; τ=Δt, mismo τ: L1 ×0.976, max|dS/dt0| ×1.453; τ=Δt, misma std: L1 ×1.080, max|dS/dt0| ×1.062.

Figuras: `binning_smooth_overlay.png` (escalera real vs bin promedio; energía en el bin 23; derivada) y `pulse_gauss_vs_supergauss.png` (formas; S_j vs caja para los tres τ; pico de la derivada vs τ; conservación de masa).

1. ¿Aproxima mejor la caja la super-gaussiana?  Sólo en apariencia.  Con el MISMO τ, L1 baja 13 % (τ ≤ Δt/3) — pero únicamente porque su std es 0.822τ: es un pulso más estrecho, no uno 'más caja'.  Con la MISMA std, L1 es PEOR en 6 % en los tres τ: para τ ≪ Δt, L1 ≈ 2·E|t−t0| y una densidad de cima plana tiene más desviación media absoluta por unidad de std (E|u|/std = 0.841 frente a 0.798 de la gaussiana).  'Elevar al cuadrado' no acerca S_j a la caja a igualdad de anchura; el ancho de la transición lo fija la std, no el exponente.  Donde la SG (misma std) SÍ gana es en el bin promedio Σ_j j·S_j: su rizado respecto a la diagonal es 0.0175 frente a 0.0434 de la gaussiana (la escalera se 'derrite' de forma más uniforme porque la función característica de la SG es menor en la frecuencia 1/Δt).
2. ¿A qué costo en la derivada?  Para τ ≤ Δt/3 el pico de dS_j/dt0 es s_p(0)/τ_p y la SG NO es más picuda: 0.98× la gaussiana con el mismo τ y 0.81× con la misma std (la cima plana reparte la misma área con menor máximo).  Para τ = Δt, cuando los dos bordes del bin caen en los flancos, los flancos empinados sí se notan: 1.45× (mismo τ) y 1.06× (misma std).  El costo real está en la FORMA de la derivada: una meseta que cae como exp(−u⁴/4), con gradiente ≈0 fuera de |u|≳2 (zonas muertas más anchas y de borde más abrupto) y segunda derivada mayor; para un optimizador o para un CRB evaluado con t0 en medio de un bin y τ pequeño, la información de Fisher queda confinada a ventanas más estrechas y el Hessiano peor condicionado.
3. Recomendación para el CRB: la GAUSSIANA (p=1).  La caja no es física, es el artefacto numérico de ceil(); lo que hay que modelar es la respuesta temporal real (pulso láser de ps + jitter del SPAD + electrónica), que es aproximadamente gaussiana (con una cola exponencial de difusión en SPADs reales), no una caja.  La gaussiana (i) es la única con interpretación física directa (τ = σ del IRF, o Δt/√12 si sólo se quiere reproducir la varianza de la cuantización), (ii) tiene derivada suave, sin mesetas ni zonas muertas, con Fisher bien condicionado en todo t0, y (iii) la super-gaussiana sólo compra parecerse más al artefacto.  Si el objetivo fuera reproducir exactamente la caja, el límite p→∞ es un pulso uniforme y S_j se vuelve un trapecio: se recupera la derivada discontinua que se quería evitar.
