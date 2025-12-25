#!/usr/bin/env python3
"""
Opposite Embedding Finder using CLIP
Finds the opposite embedding of a given token/word by searching CLIP's token vocabulary.
"""

import torch
import clip
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class OppositeEmbeddingFinder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP model for finding opposite embeddings.

        Args:
            model_name: CLIP model to use (default: "ViT-B/32")
            device: Device to run on (cuda/cpu), auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Get CLIP's token vocabulary
        print("Loading CLIP token vocabulary...")
        self.token_vocabulary = self._load_clip_vocabulary()
        print(f"Loaded {len(self.token_vocabulary)} tokens from CLIP vocabulary")

        self.token_embeddings = None

    def _load_clip_vocabulary(self) -> List[str]:
        """Load CLIP's BPE token vocabulary."""
        # Access the tokenizer's encoder (token -> id mapping)
        tokenizer = clip.simple_tokenizer.SimpleTokenizer()

        # Get all tokens from the encoder
        # The encoder is a dict mapping byte-pair tokens to IDs
        tokens = list(tokenizer.encoder.keys())

        # Clean up tokens - remove special byte-pair encoding artifacts
        cleaned_tokens = []
        for token in tokens:
            # Replace the special </w> marker that indicates end of word
            clean_token = token.replace('</w>', '')
            # Only include tokens that have actual content
            if clean_token and len(clean_token) > 0:
                cleaned_tokens.append(clean_token)

        return cleaned_tokens

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

    def compute_token_embeddings(self, batch_size: int = 256):
        """
        Pre-compute embeddings for all CLIP tokens.

        Args:
            batch_size: Number of tokens to process in each batch
        """
        if self.token_embeddings is not None:
            return

        print(f"Computing embeddings for {len(self.token_vocabulary)} CLIP tokens...")
        print("This may take a few minutes...")

        embeddings = []

        # Process tokens in batches for efficiency
        for i in tqdm(range(0, len(self.token_vocabulary), batch_size)):
            batch_tokens = self.token_vocabulary[i:i + batch_size]

            # Get embeddings for this batch
            batch_embeddings = []
            for token in batch_tokens:
                try:
                    emb = self.get_text_embedding(token)
                    batch_embeddings.append(emb)
                except Exception:
                    # If a token can't be embedded, use zeros
                    batch_embeddings.append(np.zeros(512))

            embeddings.extend(batch_embeddings)

        self.token_embeddings = np.array(embeddings)
        print(f"Token embeddings computed! Shape: {self.token_embeddings.shape}")

    def find_nearest_tokens(self, embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the nearest tokens to a given embedding from CLIP's vocabulary.

        Args:
            embedding: Target embedding vector
            top_k: Number of top results to return

        Returns:
            List of (token, similarity_score) tuples
        """
        self.compute_token_embeddings()

        # Compute cosine similarities
        similarities = np.dot(self.token_embeddings, embedding)

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.token_vocabulary[idx], float(similarities[idx])) for idx in top_indices]

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
    print("CLIP Opposite Embedding Finder")
    print("="*60)
    print("\nThis tool finds the 'opposite' of a word's embedding by")
    print("negating its CLIP embedding vector and searching through")
    print("CLIP's entire token vocabulary (~49K tokens) to find which")
    print("tokens are closest to that opposite direction in embedding space.")
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
