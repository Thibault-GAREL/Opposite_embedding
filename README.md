# CLIP Embedding Tools

Two tools for exploring word embeddings using CLIP (Contrastive Language-Image Pre-training):
1. **Opposite Embedding Finder** - Find semantically opposite words
2. **Embedding Arithmetic** - Perform vector operations (e.g., "man + kingdom ≈ king")

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Tool 1: Opposite Embedding Finder

Find the "opposite" of a word by negating its embedding vector.

**Interactive Mode:**
```bash
python opposite_embedding.py
```

**As a Python Module:**
```python
from opposite_embedding import OppositeEmbeddingFinder

finder = OppositeEmbeddingFinder()
finder.find_opposite("happy", top_k=10)
# Finds tokens closest to -1 * embedding("happy")
```

### Tool 2: Embedding Arithmetic

Perform vector addition, subtraction, and complex operations.

**Interactive Mode:**
```bash
python embedding_arithmetic.py
```

**Commands:**
- `add man kingdom` - Add two word vectors
- `sub king man` - Subtract vectors
- `calc king - man + woman` - Complex expressions

**As a Python Module:**
```python
from embedding_arithmetic import EmbeddingArithmetic

arithmetic = EmbeddingArithmetic()

# Add vectors
arithmetic.add_vectors("man", "kingdom")

# Subtract vectors
arithmetic.subtract_vectors("king", "man")

# Complex operations
arithmetic.complex_arithmetic("king - man + woman")
```

## How It Works

1. **CLIP Model**: Uses OpenAI's CLIP model (ViT-B/32) to get 512-dimensional semantic embeddings
2. **Token Embeddings**: Directly accesses CLIP's token embedding layer (instant - no encoding needed!)
3. **Negation**: The opposite embedding is computed as `-1 * original_embedding`
4. **Search**: Computes cosine similarity between opposite embedding and all ~49K token embeddings
5. **Results**: Returns the top-k tokens closest to the opposite embedding

**Key optimization**: Instead of encoding 49K tokens as text (slow), we directly use the model's token embedding weights (instant!)

## Examples

### Example 1: Opposite Embedding

```
Enter a word: hot

Finding opposite embedding for: 'hot'
============================================================

Extracting token embeddings from CLIP model...
Got 49408 token embeddings directly from model!

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

### Example 2: Vector Addition

```
Enter command: add man kingdom

============================================================
Vector Addition: 'man' + 'kingdom' = ?
============================================================

'man' embedding (first 5 dims): [ 0.0123 -0.0456  0.0789 ...]
'kingdom' embedding (first 5 dims): [ 0.0234  0.0567 -0.0123 ...]

Result embedding (first 5 dims): [ 0.0357  0.0111  0.0666 ...]

Top 10 tokens closest to 'man' + 'kingdom':
------------------------------------------------------------
 1. 'king'                     (similarity: 0.8123)
 2. 'prince'                   (similarity: 0.7654)
 3. 'monarch'                  (similarity: 0.7234)
 ...
```

### Example 3: Complex Arithmetic

```
Enter command: calc king - man + woman

============================================================
Complex Arithmetic: king - man + woman
============================================================

Starting with 'king'
  - 'man'
  + 'woman'

Result embedding (first 5 dims): [ 0.0234 -0.0123  0.0456 ...]

Top 10 tokens closest to the result:
------------------------------------------------------------
 1. 'queen'                    (similarity: 0.8234)
 2. 'princess'                 (similarity: 0.7543)
 3. 'monarch'                  (similarity: 0.7123)
 ...
```

**Note**: Token embeddings are extracted directly from the model (instant!). First operation extracts them, then all subsequent operations use cached embeddings.

## License

MIT License
