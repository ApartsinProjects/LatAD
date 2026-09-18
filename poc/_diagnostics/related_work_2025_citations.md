# Recent related-work citations (validated 2026-09-17, Crossref + arXiv)

All three real, title+authors match the resolved id (no hallucination). Ready to add to IoT2.html.

| Suggested anchor | Citation | DOI / arXiv | Relevance / positioning |
|---|---|---|---|
| `r-birihanu` | Birihanu, E.; Lendák, I. Explainable correlation-based anomaly detection for Industrial Control Systems. *Frontiers in Artificial Intelligence* **2025**, 7, 1508821. | 10.3389/frai.2024.1508821 | **Closest recent work:** builds a Latent Correlation Matrix (Pearson) + multivariate Gaussian to flag ICS anomalies, including hidden ones, with correlation-structure explainability. Contrast: we factorize a *learned latent* density over correlation *communities* and combine by Higher Criticism, rather than a single global correlation matrix. |
| `r-islam` | Islam, M.S.; Carden, J. Product-Aware Deep Autoencoders for Robust Process Monitoring in Multi-Product Cyber-Physical Systems. *arXiv* **2026**, arXiv:2606.00052. | arXiv:2606.00052 | Global AEs create detection blind spots on multi-product CPS; product-grade-specific AEs recover them (Extended TEP). Aligns with our regime/coverage argument (a global model dilutes rare regimes); we address it with regime-latent factorization rather than per-product models. |
| `r-graphmoe` | Huang, X.; Chen, W.; Hu, B.; Mao, Z. Graph Mixture of Experts and Memory-augmented Routers for Multivariate Time Series Anomaly Detection. *Proc. AAAI Conf. Artif. Intell.* **2025**, 39(16), 17476-17484. | 10.1609/aaai.v39i16.33921 | Plug-in graph MoE routing hierarchical graph info into entity representations for GNN MTS detectors. Contrast: we route per-*community* density experts by cohesion-weighted Higher Criticism, not a learned MoE gate. |

Notes: Birihanu DOI year-prefix 2024 (Frontiers), publication year 2025 — cite as 2025. Islam & Carden is a preprint (no peer-reviewed venue). Graph-MoE: prefer AAAI DOI (arXiv 2412.19108 is the same paper).
