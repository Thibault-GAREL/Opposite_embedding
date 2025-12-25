#!/usr/bin/env python3
"""
Opposite Embedding Finder using CLIP
Finds the opposite embedding of a given token/word.
"""

import torch
import clip
import numpy as np
from typing import List, Tuple


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

        # Common vocabulary for finding nearest neighbors
        self.vocabulary = self._load_vocabulary()
        self.vocab_embeddings = None

    def _load_vocabulary(self) -> List[str]:
        """Load a vocabulary of common English words."""
        # Basic vocabulary - can be extended
        vocab = [
            # Emotions & States
            "happy", "sad", "joy", "sorrow", "love", "hate", "peace", "war",
            "calm", "angry", "excited", "bored", "confident", "afraid",
            "strong", "weak", "brave", "cowardly", "kind", "cruel",

            # Physical Properties
            "hot", "cold", "warm", "cool", "big", "small", "large", "tiny",
            "tall", "short", "wide", "narrow", "thick", "thin", "heavy", "light",
            "fast", "slow", "hard", "soft", "rough", "smooth", "sharp", "dull",

            # Directions & Positions
            "up", "down", "left", "right", "forward", "backward", "inside", "outside",
            "above", "below", "high", "low", "near", "far", "close", "distant",

            # Time
            "day", "night", "morning", "evening", "early", "late", "new", "old",
            "young", "ancient", "modern", "past", "future", "present",

            # Quality
            "good", "bad", "beautiful", "ugly", "clean", "dirty", "bright", "dark",
            "loud", "quiet", "rich", "poor", "full", "empty", "wet", "dry",
            "alive", "dead", "awake", "asleep", "open", "closed", "easy", "difficult",

            # Abstract
            "truth", "lie", "success", "failure", "victory", "defeat", "freedom", "slavery",
            "order", "chaos", "simple", "complex", "knowledge", "ignorance",

            # Nature
            "summer", "winter", "spring", "autumn", "sun", "moon", "fire", "water",
            "earth", "sky", "mountain", "valley", "sea", "desert", "forest", "plain",

            # Colors
            "white", "black", "light", "dark", "bright", "dim",

            # Actions
            "give", "take", "push", "pull", "start", "stop", "build", "destroy",
            "create", "destroy", "attack", "defend", "grow", "shrink", "rise", "fall",
        ]
        return vocab

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

    def compute_vocabulary_embeddings(self):
        """Pre-compute embeddings for all vocabulary words."""
        if self.vocab_embeddings is not None:
            return

        print(f"Computing embeddings for {len(self.vocabulary)} vocabulary words...")
        embeddings = []
        for word in self.vocabulary:
            emb = self.get_text_embedding(word)
            embeddings.append(emb)
        self.vocab_embeddings = np.array(embeddings)
        print("Vocabulary embeddings computed!")

    def find_nearest_words(self, embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the nearest words to a given embedding.

        Args:
            embedding: Target embedding vector
            top_k: Number of top results to return

        Returns:
            List of (word, similarity_score) tuples
        """
        self.compute_vocabulary_embeddings()

        # Compute cosine similarities
        similarities = np.dot(self.vocab_embeddings, embedding)

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.vocabulary[idx], float(similarities[idx])) for idx in top_indices]

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

        # Find nearest words to opposite embedding
        print(f"\nTop {top_k} words closest to the opposite embedding:")
        print("-" * 60)
        results = self.find_nearest_words(opposite_emb, top_k)

        for i, (near_word, score) in enumerate(results, 1):
            print(f"{i:2d}. {near_word:20s} (similarity: {score:.4f})")

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
    print("negating its CLIP embedding vector and finding which words")
    print("are closest to that opposite direction in the embedding space.")
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
