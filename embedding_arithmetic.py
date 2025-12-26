#!/usr/bin/env python3
"""
Embedding Arithmetic using CLIP
Perform vector operations like: word1 + word2 = ?
Example: man + kingdom ≈ king
"""

import torch
import clip
import numpy as np
from typing import List, Tuple


class EmbeddingArithmetic:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP model for embedding arithmetic.

        Args:
            model_name: CLIP model to use (default: "ViT-B/32")
            device: Device to run on (cuda/cpu), auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.token_embeddings = None
        print("Model loaded! Ready for embedding arithmetic.")

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

    def get_token_embeddings(self):
        """
        Get the token embedding weights directly from CLIP's model.
        This is instant - no need to encode each token individually!

        Returns:
            Token embeddings from the model's embedding layer
        """
        if self.token_embeddings is not None:
            return self.token_embeddings

        print("Extracting token embeddings from CLIP model...")

        # Get the token embedding layer from CLIP's transformer
        token_embedding_layer = self.model.token_embedding.weight

        # Get embeddings and normalize them
        embeddings = token_embedding_layer.detach().cpu().numpy()

        # Normalize embeddings (same as CLIP does for text)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.token_embeddings = embeddings / norms

        print(f"Got {len(self.token_embeddings)} token embeddings directly from model!")
        return self.token_embeddings

    def find_nearest_tokens(self, embedding: np.ndarray, top_k: int = 10, exclude_tokens: List[str] = None) -> List[Tuple[str, float]]:
        """
        Find the nearest tokens to a given embedding from CLIP's vocabulary.

        Args:
            embedding: Target embedding vector
            top_k: Number of top results to return
            exclude_tokens: List of tokens to exclude from results

        Returns:
            List of (token, similarity_score) tuples
        """
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Get token embeddings from the model (instant!)
        token_embeds = self.get_token_embeddings()

        # Compute cosine similarities
        similarities = np.dot(token_embeds, embedding)

        # Get top k (get more than needed in case we need to filter)
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]

        # Map back to token strings
        tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        decoder = {v: k for k, v in tokenizer.encoder.items()}  # id -> token

        results = []
        exclude_set = set(exclude_tokens) if exclude_tokens else set()

        for idx in top_indices:
            token_str = decoder.get(idx, f"<unk_{idx}>")
            # Clean up the token display
            token_str_clean = token_str.replace('</w>', '')

            # Skip excluded tokens
            if token_str_clean.lower() in exclude_set:
                continue

            results.append((token_str_clean, float(similarities[idx])))

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
    arithmetic = EmbeddingArithmetic()

    print("\n" + "="*60)
    print("CLIP Embedding Arithmetic")
    print("="*60)
    print("\nPerform vector operations on word embeddings!")
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
