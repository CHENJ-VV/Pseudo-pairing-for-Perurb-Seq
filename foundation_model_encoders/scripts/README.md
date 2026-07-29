# Encoder scripts

These files preserve the original tokenization, embedding extraction, scVI/HVG, and UMAP procedures. Machine-specific constants were replaced by `PPFM_*` environment variables supplied by `../launcher.py`.

Run the scripts through the launcher rather than editing them:

```bash
python ../launcher.py run --config ../configs/encoders.yaml --model scgpt
```

Direct execution is also possible after exporting all required variables, but the YAML launcher is the supported interface. `runtime_config.py` contains the environment parsing and repository-path insertion utilities.

| Script | Stage |
|---|---|
| `Geneformer_tokenize_emb.py` | Chunked Geneformer tokenization and embeddings |
| `scGPT_token_emb.py` | scGPT embeddings |
| `scCello_token_emb.py` | scCello tokenization and embeddings |
| `Scimilarity_tokenize.py` | SCimilarity embeddings |
| `scVI_token_emb.py` | HVG-PCA and/or scVI embeddings |
| `UMAP_plot.py` | UMAP visualization of generated embeddings |
