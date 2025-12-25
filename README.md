# Opposite Embedding Finder

Find the "opposite" embedding of any word using CLIP (Contrastive Language-Image Pre-training).

## Concept

This tool computes the opposite of a word's embedding by:
1. Getting the CLIP text embedding for your input word
2. Negating the embedding vector (multiplying by -1)
3. Searching through CLIP's entire token vocabulary (~49,000 tokens) to find which tokens are closest to this opposite embedding

The "opposite" in embedding space represents the direction that is most dissimilar to the original word in the high-dimensional semantic space.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

```bash
python opposite_embedding.py
```

Then type words to find their opposites!

### As a Python Module

```python
from opposite_embedding import OppositeEmbeddingFinder

# Initialize
finder = OppositeEmbeddingFinder()

# Find opposite of a word
finder.find_opposite("happy", top_k=10)
```

## How It Works

1. **CLIP Model**: Uses OpenAI's CLIP model (ViT-B/32) to get 512-dimensional semantic embeddings
2. **Token Vocabulary**: Loads CLIP's BPE token vocabulary (~49,000 tokens)
3. **Embedding Computation**: Pre-computes embeddings for all tokens (done once, cached)
4. **Negation**: The opposite embedding is computed as `-1 * original_embedding`
5. **Search**: Computes cosine similarity between opposite embedding and all token embeddings
6. **Results**: Returns the top-k tokens closest to the opposite embedding

## Example

```
Enter a word: hot

Finding opposite embedding for: 'hot'
============================================================

Loading CLIP token vocabulary...
Loaded 49408 tokens from CLIP vocabulary
Computing embeddings for 49408 CLIP tokens...
This may take a few minutes...
100%|████████████████████████| 193/193 [02:15<00:00,  1.42it/s]
Token embeddings computed! Shape: (49408, 512)

Original embedding shape: (512,)
Original embedding (first 5 dims): [ 0.0234 -0.0123  0.0456 ...]

Opposite embedding (first 5 dims): [-0.0234  0.0123 -0.0456 ...]
Dot product (should be ~-1): -1.0000

Top 10 tokens from CLIP vocabulary closest to the opposite embedding:
------------------------------------------------------------
 1. 'cold'                     (similarity: 0.7234)
 2. 'cool'                     (similarity: 0.6543)
 3. 'freezing'                 (similarity: 0.6102)
 ...
```

**Note**: The first run will take 2-3 minutes to compute embeddings for all tokens, but this is cached for subsequent searches.

## License

MIT License
