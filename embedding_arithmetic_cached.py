#!/usr/bin/env python3
"""
Embedding Arithmetic with Cached Full Token Vocabulary
Perform vector operations with all 49K CLIP tokens using pre-computed embeddings.
Example: man + kingdom ≈ king

Features:
- Full 49K token vocabulary from CLIP
- Pre-computes embeddings through full transformer (one-time: 2-3 minutes)
- Saves to disk for instant loading on subsequent runs (~1 second)
- High quality results with complete vocabulary coverage
"""

import torch
import clip
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import os


class EmbeddingArithmeticCached:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None, cache_file: str = "clip_token_embeddings.npz"):
        """
        Initialize the CLIP model for embedding arithmetic with cached token embeddings.

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

    def find_nearest_tokens(self, embedding: np.ndarray, top_k: int = 10, exclude_tokens: List[str] = None) -> List[Tuple[str, float]]:
        """
        Find the nearest tokens to a given embedding from the full vocabulary.

        Args:
            embedding: Target embedding vector
            top_k: Number of top results to return
            exclude_tokens: List of tokens to exclude from results

        Returns:
            List of (token, similarity_score) tuples
        """
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Compute cosine similarities with all token embeddings
        similarities = np.dot(self.token_embeddings, embedding)

        # Get top k (get more than needed in case we need to filter)
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]

        results = []
        exclude_set = set([t.lower() for t in exclude_tokens]) if exclude_tokens else set()

        for idx in top_indices:
            token = self.vocabulary[idx]
            # Clean up the token display
            token_clean = token.replace('</w>', '')

            # Skip excluded tokens
            if token_clean.lower() in exclude_set:
                continue

            results.append((token_clean, float(similarities[idx])))

            if len(results) >= top_k:
                break

        return results

    def add_vectors(self, word1: str, word2: str, top_k: int = 10):
        """
        Add two word embeddings and find nearest tokens.
        Example: man + kingdom ≈ king

        Args:
            word1: First word
            word2: Second word
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Vector Addition: '{word1}' + '{word2}' = ?")
        print(f"{'='*60}\n")

        # Get embeddings
        emb1 = self.get_text_embedding(word1)
        emb2 = self.get_text_embedding(word2)

        print(f"'{word1}' embedding (first 5 dims): {emb1[:5]}")
        print(f"'{word2}' embedding (first 5 dims): {emb2[:5]}")

        # Add vectors
        result_emb = emb1 + emb2
        print(f"\nResult embedding (first 5 dims): {result_emb[:5]}")

        # Find nearest tokens (exclude input words)
        print(f"\nTop {top_k} tokens closest to '{word1}' + '{word2}':")
        print("-" * 60)
        results = self.find_nearest_tokens(result_emb, top_k, exclude_tokens=[word1, word2])

        for i, (token, score) in enumerate(results, 1):
            print(f"{i:2d}. '{token}' {' ' * (25 - len(token))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results

    def subtract_vectors(self, word1: str, word2: str, top_k: int = 10):
        """
        Subtract two word embeddings and find nearest tokens.
        Example: king - man ≈ queen (conceptually)

        Args:
            word1: First word (minuend)
            word2: Second word (subtrahend)
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Vector Subtraction: '{word1}' - '{word2}' = ?")
        print(f"{'='*60}\n")

        # Get embeddings
        emb1 = self.get_text_embedding(word1)
        emb2 = self.get_text_embedding(word2)

        print(f"'{word1}' embedding (first 5 dims): {emb1[:5]}")
        print(f"'{word2}' embedding (first 5 dims): {emb2[:5]}")

        # Subtract vectors
        result_emb = emb1 - emb2
        print(f"\nResult embedding (first 5 dims): {result_emb[:5]}")

        # Find nearest tokens (exclude input words)
        print(f"\nTop {top_k} tokens closest to '{word1}' - '{word2}':")
        print("-" * 60)
        results = self.find_nearest_tokens(result_emb, top_k, exclude_tokens=[word1, word2])

        for i, (token, score) in enumerate(results, 1):
            print(f"{i:2d}. '{token}' {' ' * (25 - len(token))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results

    def complex_arithmetic(self, operations: str, top_k: int = 10):
        """
        Perform complex vector arithmetic with multiple operations.
        Example: "king - man + woman"

        Args:
            operations: String with operations like "king - man + woman"
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Complex Arithmetic: {operations}")
        print(f"{'='*60}\n")

        # Parse the operations
        import re
        # Split by + and - while keeping the operators
        tokens = re.split(r'(\+|\-)', operations)
        tokens = [t.strip() for t in tokens if t.strip()]

        if not tokens:
            print("Error: No valid operations found")
            return []

        # Get all words involved for exclusion
        words = [t for t in tokens if t not in ['+', '-']]

        # Start with the first word
        result_emb = self.get_text_embedding(tokens[0])
        print(f"Starting with '{tokens[0]}'")

        # Process operations
        i = 1
        while i < len(tokens):
            if i + 1 < len(tokens):
                operator = tokens[i]
                word = tokens[i + 1]

                emb = self.get_text_embedding(word)

                if operator == '+':
                    result_emb = result_emb + emb
                    print(f"  + '{word}'")
                elif operator == '-':
                    result_emb = result_emb - emb
                    print(f"  - '{word}'")

                i += 2
            else:
                break

        print(f"\nResult embedding (first 5 dims): {result_emb[:5]}")

        # Find nearest tokens (exclude input words)
        print(f"\nTop {top_k} tokens closest to the result:")
        print("-" * 60)
        results = self.find_nearest_tokens(result_emb, top_k, exclude_tokens=words)

        for i, (token, score) in enumerate(results, 1):
            print(f"{i:2d}. '{token}' {' ' * (25 - len(token))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results


def main():
    """Main function for interactive use."""
    arithmetic = EmbeddingArithmeticCached()

    print("\n" + "="*60)
    print("CLIP Embedding Arithmetic (Cached Full Vocabulary)")
    print("="*60)
    print("\nPerform vector operations on word embeddings!")
    print(f"Using full CLIP vocabulary: {len(arithmetic.vocabulary)} tokens")
    print("\nCommands:")
    print("  add <word1> <word2>       - Add two vectors")
    print("  sub <word1> <word2>       - Subtract vectors (word1 - word2)")
    print("  calc <expression>         - Complex expression (e.g., 'king - man + woman')")
    print("  quit/exit                 - Exit program")
    print("\nExamples:")
    print("  add man kingdom")
    print("  sub king man")
    print("  calc king - man + woman")
    print()

    while True:
        try:
            user_input = input("Enter command: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            parts = user_input.split(None, 1)
            if len(parts) == 0:
                continue

            command = parts[0].lower()

            if command == 'add' and len(parts) > 1:
                words = parts[1].split()
                if len(words) >= 2:
                    arithmetic.add_vectors(words[0], words[1])
                else:
                    print("Error: Need two words for addition. Example: add man kingdom")

            elif command == 'sub' and len(parts) > 1:
                words = parts[1].split()
                if len(words) >= 2:
                    arithmetic.subtract_vectors(words[0], words[1])
                else:
                    print("Error: Need two words for subtraction. Example: sub king man")

            elif command == 'calc' and len(parts) > 1:
                expression = parts[1]
                arithmetic.complex_arithmetic(expression)

            else:
                print("Unknown command. Type 'quit' to exit or use: add/sub/calc")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
