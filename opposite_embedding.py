#!/usr/bin/env python3
"""
Opposite Embedding Finder using CLIP (with cached embeddings)
Finds the opposite embedding of a given token/word by searching CLIP's full token vocabulary.
Uses pre-computed embeddings cached to disk for high-quality results.
"""

import torch
import clip
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import os


class OppositeEmbeddingFinder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None, cache_file: str = "clip_token_embeddings.npz"):
        """
        Initialize the CLIP model for finding opposite embeddings.

        Args:
            model_name: CLIP model to use (default: "ViT-B/32")
            device: Device to run on (cuda/cpu), auto-detected if None
            cache_file: File to cache token embeddings (default: "clip_token_embeddings.npz")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_file = cache_file

        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Load tokenizer and vocabulary
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.vocabulary = list(self.tokenizer.encoder.keys())
        print(f"CLIP vocabulary size: {len(self.vocabulary)} tokens")

        # Load or compute token embeddings
        self.token_embeddings = None
        self.load_or_compute_embeddings()

    def load_or_compute_embeddings(self):
        """Load embeddings from cache or compute them if cache doesn't exist."""
        if os.path.exists(self.cache_file):
            print(f"\nLoading pre-computed embeddings from {self.cache_file}...")
            data = np.load(self.cache_file, allow_pickle=True)
            self.token_embeddings = data['embeddings']
            cached_vocab = data['vocabulary'].tolist()

            # Verify vocabulary matches
            if cached_vocab == self.vocabulary:
                print(f"✓ Loaded {len(self.token_embeddings)} cached embeddings!")
                print(f"  Shape: {self.token_embeddings.shape}")
                return
            else:
                print("⚠ Vocabulary mismatch - recomputing embeddings...")

        print(f"\nCache not found. Computing embeddings for {len(self.vocabulary)} tokens...")
        print("This is a ONE-TIME operation that takes 2-3 minutes.")
        print("Subsequent runs will load instantly from cache!\n")
        self.compute_all_token_embeddings()
        self.save_embeddings()

    def compute_all_token_embeddings(self, batch_size: int = 64):
        """
        Compute embeddings for all tokens through the full CLIP transformer.

        Args:
            batch_size: Number of tokens to process in each batch
        """
        embeddings = []

        # Process tokens in batches
        num_batches = (len(self.vocabulary) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(self.vocabulary), batch_size),
                      total=num_batches,
                      desc="Encoding tokens"):
            batch_tokens = self.vocabulary[i:i + batch_size]

            try:
                # Tokenize batch
                text_inputs = clip.tokenize(batch_tokens, truncate=True).to(self.device)

                # Encode through full CLIP model
                with torch.no_grad():
                    text_features = self.model.encode_text(text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                embeddings.append(text_features.cpu().numpy())

            except Exception as e:
                print(f"\nWarning: Failed to encode batch at index {i}: {e}")
                # Add zero vectors for failed tokens
                embeddings.append(np.zeros((len(batch_tokens), 512)))

        # Combine all batches
        self.token_embeddings = np.vstack(embeddings)
        print(f"\n✓ Computed embeddings! Shape: {self.token_embeddings.shape}")

    def save_embeddings(self):
        """Save computed embeddings to disk."""
        print(f"\nSaving embeddings to {self.cache_file}...")
        np.savez_compressed(
            self.cache_file,
            embeddings=self.token_embeddings,
            vocabulary=np.array(self.vocabulary, dtype=object)
        )
        print(f"✓ Saved! Future runs will load instantly from cache.")

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get CLIP embedding for a text input.

        Args:
            text: Input text/word

        Returns:
            Normalized embedding vector
        """
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def get_opposite_embedding(self, text: str) -> np.ndarray:
        """
        Get the opposite embedding by negating the original embedding.

        Args:
            text: Input text/word

        Returns:
            Opposite (negated) embedding vector
        """
        embedding = self.get_text_embedding(text)
        opposite = -embedding
        # Normalize
        opposite = opposite / np.linalg.norm(opposite)
        return opposite

    def find_nearest_tokens(self, embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the nearest tokens to a given embedding from CLIP's vocabulary.

        Args:
            embedding: Target embedding vector
            top_k: Number of top results to return

        Returns:
            List of (token, similarity_score) tuples
        """
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Compute cosine similarities
        similarities = np.dot(self.token_embeddings, embedding)

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            token = self.vocabulary[idx]
            # Clean up the token display
            token_clean = token.replace('</w>', '')
            results.append((token_clean, float(similarities[idx])))

        return results

    def find_opposite(self, word: str, top_k: int = 10):
        """
        Find and display the opposite embedding of a word.

        Args:
            word: Input word
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Finding opposite embedding for: '{word}'")
        print(f"{'='*60}\n")

        # Get original embedding
        original_emb = self.get_text_embedding(word)
        print(f"Original embedding shape: {original_emb.shape}")
        print(f"Original embedding (first 5 dims): {original_emb[:5]}")

        # Get opposite embedding
        opposite_emb = self.get_opposite_embedding(word)
        print(f"\nOpposite embedding (first 5 dims): {opposite_emb[:5]}")

        # Verify they are opposite
        dot_product = np.dot(original_emb, opposite_emb)
        print(f"Dot product (should be ~-1): {dot_product:.4f}")

        # Find nearest tokens to opposite embedding
        print(f"\nTop {top_k} tokens from CLIP vocabulary closest to the opposite embedding:")
        print("-" * 60)
        results = self.find_nearest_tokens(opposite_emb, top_k)

        for i, (token, score) in enumerate(results, 1):
            print(f"{i:2d}. '{token}' {' ' * (25 - len(token))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results


def main():
    """Main function for interactive use."""
    # Initialize the finder
    finder = OppositeEmbeddingFinder()

    print("\n" + "="*60)
    print("CLIP Opposite Embedding Finder (Cached)")
    print("="*60)
    print("\nThis tool finds the 'opposite' of a word's embedding by")
    print("negating its CLIP embedding vector and searching through")
    print("all 49K CLIP tokens with pre-computed high-quality embeddings.")
    print("\nNote: Uses cached full-transformer embeddings for best results!")
    print("\nType 'quit' or 'exit' to stop.\n")

    # Interactive loop
    while True:
        word = input("Enter a word: ").strip()

        if word.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not word:
            print("Please enter a valid word.")
            continue

        try:
            finder.find_opposite(word, top_k=10)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
