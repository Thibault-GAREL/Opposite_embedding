# Opposite Embedding Finder

Find the "opposite" embedding of any word using CLIP (Contrastive Language-Image Pre-training).

## Concept

This tool computes the opposite of a word's embedding by:
1. Getting the CLIP text embedding for your input word
2. Negating the embedding vector (multiplying by -1)
3. Finding which words in our vocabulary are closest to this opposite embedding

The "opposite" in embedding space represents the direction that is most dissimilar to the original word.

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

1. **CLIP Model**: Uses OpenAI's CLIP model to get semantic embeddings
2. **Negation**: The opposite embedding is computed as `-1 * original_embedding`
3. **Search**: Finds nearest neighbors in vocabulary to the opposite embedding
4. **Results**: Shows which words are semantically opposite in the embedding space

## Example

```
Enter a word: hot

Finding opposite embedding for: 'hot'
============================================================

Original embedding shape: (512,)
Original embedding (first 5 dims): [ 0.0234 -0.0123  0.0456 ...]

Opposite embedding (first 5 dims): [-0.0234  0.0123 -0.0456 ...]
Dot product (should be ~-1): -1.0000

Top 10 words closest to the opposite embedding:
------------------------------------------------------------
 1. cold                 (similarity: 0.7234)
 2. cool                 (similarity: 0.6543)
 3. winter               (similarity: 0.5432)
 ...
```

## License

MIT License
