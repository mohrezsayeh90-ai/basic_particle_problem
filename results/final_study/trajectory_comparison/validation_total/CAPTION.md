# Validation total

**Caption.** Numerical flow comparison and selected particle trajectories. (a–c) Tangential velocity profiles at three channel sections. (d–f) PINN versus FEM velocity components and pressure, using all saved independent field-validation samples; pressure gauges are aligned by separate mean subtraction. Dashed diagonals indicate identity. (g–i) Termination-time parity, common-time coverage, and complete paired particle paths for four selected baseline inlet seeds, η₀ = −0.65, −0.4333, −0.2167, and 0. Colors match across these three panels. Solid trajectories are PINN + ForceNet and dashed trajectories are FEM + Stokes; dots mark three equally spaced physical times on each PINN path and arrows indicate motion. The dashed 95% coverage line is a visual reference.

**Scope.** Particle panels show 4 of 7 baseline seeds, selected explicitly as requested. The other three seeds (coverage 96.2%, 87.4%, and 73.6%) are omitted from this figure only. This selected illustration does not establish accuracy across all particle trajectories; the full comparison remains in the adjacent `paper_figures/` directory. Coverage measures common integration duration, not position accuracy. Selected synchronous RMS errors span 2.29–7.82% of channel width; geometric RMS errors span 0.32–1.56%. No simulation data were changed or trajectories extrapolated. This is a numerical FEM comparison, not experimental validation.

نسخهٔ ترکیبی شامل همهٔ نقاط مقایسهٔ میدان و فقط چهار مسیر منتخب از هفت مسیر پایه است. سه مسیر با پوشش زمانی ۹۶٫۲، ۸۷٫۴ و ۷۳٫۶ درصد فقط از این شکل کنار گذاشته شده‌اند؛ نتایج کامل در مخزن موجود است. پوشش زمانی معیار دقت مکانی نیست. خطای زمانی چهار مسیر منتخب تا ۷٫۸۲٪ عرض کانال است.

Formats: vector PDF/SVG, 600 dpi PNG/TIFF. The plotted profile values and selected/excluded path metrics are in `data/`; input hashes and selection details are in `manifest.json`.

Reproduce from the repository root with a fresh output directory:

```bash
python make_validation_total.py --out results/validation_total_new
```
