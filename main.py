# pip install transformers torch sentence-transformers numpy scikit-learn loguru pyyaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import yaml
import hashlib
import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Configuration Loader
# ===============================
class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        self.model_name = data.get("model_name", "gpt2")
        self.embedding_model_name = data.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.alpha = data.get("alpha", 0.2)
        self.novelty_threshold = data.get("novelty_threshold", 0.9)
        self.max_iterations = data.get("max_iterations", 10)
        self.max_length = data.get("max_length", 128)
        self.temperature = data.get("temperature", 0.7)

# ===============================
# MiniLM Client for Generation
# ===============================
class MiniLMClient:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def chat(self, prompt, max_length=128, temperature=0.7):
        outputs = self.generator(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=temperature
        )
        return outputs[0]["generated_text"]

# ===============================
# Novelty Filter with Embeddings
# ===============================
class NoveltyFilter:
    def __init__(self, threshold, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.hash_memory = set()
        self.memory_embeddings = []
        self.threshold = threshold
        self.embed_model = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def is_novel(self, text):
        # hash check
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self.hash_memory:
            return False
        self.hash_memory.add(h)

        # semantic check
        vec = self.embed_model.encode([text])
        if self.memory_embeddings:
            sims = cosine_similarity(vec, np.vstack(self.memory_embeddings)).max()
            if sims > self.threshold:
                return False
        self.memory_embeddings.append(vec)
        return True

# ===============================
# EMA Scorer
# ===============================
class EMAScorer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema = None

    def update(self, score):
        if self.ema is None:
            self.ema = score
        else:
            self.ema = self.alpha * score + (1 - self.alpha) * self.ema
        return self.ema

# ===============================
# Adaptive Difficulty
# ===============================
class DifficultyController:
    def __init__(self):
        self.level = 1

    def adjust(self, ema):
        if ema > 0.85:
            self.level += 1
        elif ema < 0.4:
            self.level = max(1, self.level - 1)

# ===============================
# Benchmark Engine
# ===============================
class SelfEvolvingBenchmark:
    def __init__(self, config):
        self.client = MiniLMClient(config.model_name)
        self.novelty = NoveltyFilter(config.novelty_threshold, config.embedding_model_name)
        self.scorer = EMAScorer(config.alpha)
        self.difficulty = DifficultyController()
        self.max_iterations = config.max_iterations
        self.embed_model = SentenceTransformer(config.embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.max_length
        self.temperature = config.temperature

    def generate_question(self):
        prompt = f"Generate a completely novel reasoning question. Difficulty level: {self.difficulty.level}."
        return self.client.chat(prompt, max_length=self.max_length, temperature=self.temperature)

    def answer_question(self, question):
        return self.client.chat(question, max_length=self.max_length, temperature=self.temperature)

    def evaluate(self, question, answer):
        # semantic evaluation using embeddings
        q_vec = self.embed_model.encode([question])
        a_vec = self.embed_model.encode([answer])
        score = float(cosine_similarity(q_vec, a_vec)[0][0])
        return score

    def run(self):
        for i in range(self.max_iterations):
            logger.info(f"\nIteration {i+1}")
            question = self.generate_question()

            while not self.novelty.is_novel(question):
                logger.info("Regenerating question (not novel)...")
                question = self.generate_question()

            answer = self.answer_question(question)
            score = self.evaluate(question, answer)
            ema = self.scorer.update(score)
            self.difficulty.adjust(ema)

            logger.info(f"Difficulty Level: {self.difficulty.level}")
            logger.info(f"Semantic Score: {score:.3f}")
            logger.info(f"EMA: {ema:.3f}")
            logger.info(f"Question: {question[:80]}...")
            logger.info(f"Answer: {answer[:80]}...")
            logger.info("-" * 50)

# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    logger.info("Starting Self-Evolving Benchmark with MiniLM (High-Accuracy Version)...")
    config = Config()
    benchmark = SelfEvolvingBenchmark(config)
    benchmark.run()