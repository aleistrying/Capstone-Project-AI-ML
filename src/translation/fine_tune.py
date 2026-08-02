"""
Fine-tune Helsinki-NLP MarianMT models on movie-domain data.
Author: Brayan Yesid Roncancio Suarez

WHY FINE-TUNE?
---------------
The base Helsinki-NLP models are trained on general web text (news, Wikipedia,
subtitles). They are good at general translation but can mistranslate
movie-domain vocabulary:
  - "pelicula de acción" might not map cleanly to "action movie"
  - Chatbot phrases like "here are your recommendations" may sound unnatural

Fine-tuning adapts the base model to movie-specific sentence patterns using
a small set of parallel EN-ES sentence pairs (see SAMPLE_DATA below).
The result is saved to models/translation/<src>-<tgt>/ and automatically
loaded by translator.py instead of the base model.

HOW TO RUN
-----------
    # Fine-tune EN↔ES using the built-in 35 movie sentence pairs:
    python src/translation/fine_tune.py

    # Fine-tune with your own CSV (columns must be named 'en' and 'es'):
    python src/translation/fine_tune.py --csv path/to/data.csv

    # Fine-tune a different language pair (e.g. EN↔FR):
    python src/translation/fine_tune.py --src en --tgt fr

    # Control training length:
    python src/translation/fine_tune.py --epochs 5 --batch 4

OUTPUT
------
Saves the fine-tuned model to:
    models/translation/en-es/   (and models/translation/es-en/ for reverse)

translator.py._load_model() checks this directory first before downloading
from HuggingFace, so the fine-tuned model is used automatically after training.

TRAINING DETAILS
-----------------
  - Base model:    Helsinki-NLP/opus-mt-{src}-{tgt}
  - Trainer:       HuggingFace Seq2SeqTrainer
  - Metric:        BLEU score (sacrebleu)
  - Early stop:    Stops if BLEU does not improve for 2 epochs
  - Default epochs: 3  (enough for the small SAMPLE_DATA)
  - GPU:           Automatically used if available (fp16 enabled)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset
import evaluate

from src.translation.translator import MODEL_REVISIONS

# Where to save the fine-tuned models
SAVE_DIR = Path(__file__).resolve().parents[2] / "models" / "translation"

# ─────────────────────────────────────────────
# Built-in movie-domain parallel corpus (EN-ES)
# Used when no external CSV is provided.
# ─────────────────────────────────────────────
SAMPLE_DATA = [
    {
        "en": "I want to watch an action movie with explosions and car chases.",
        "es": "Quiero ver una película de acción con explosiones y persecuciones de autos.",
    },
    {
        "en": "Recommend me a romantic comedy from the 2000s.",
        "es": "Recomiéndame una comedia romántica de los años 2000.",
    },
    {
        "en": "I am looking for a horror film that is really scary.",
        "es": "Estoy buscando una película de terror que sea realmente aterradora.",
    },
    {
        "en": "Can you suggest a family-friendly adventure movie?",
        "es": "¿Puedes sugerir una película de aventuras apta para toda la familia?",
    },
    {
        "en": "I want something funny to watch tonight.",
        "es": "Quiero ver algo divertido esta noche.",
    },
    {
        "en": "Show me science fiction movies about space exploration.",
        "es": "Muéstrame películas de ciencia ficción sobre la exploración espacial.",
    },
    {
        "en": "I prefer films with a good storyline and deep characters.",
        "es": "Prefiero películas con una buena historia y personajes profundos.",
    },
    {
        "en": "What are the best thriller movies of the last decade?",
        "es": "¿Cuáles son las mejores películas de suspenso de la última década?",
    },
    {
        "en": "I enjoy animated movies with a feel-good ending.",
        "es": "Disfruto las películas animadas con un final feliz.",
    },
    {
        "en": "Please recommend a drama movie with high ratings.",
        "es": "Por favor recomiéndame una película dramática con altas calificaciones.",
    },
    {
        "en": "This movie has great special effects and a compelling plot.",
        "es": "Esta película tiene excelentes efectos especiales y una trama cautivadora.",
    },
    {
        "en": "The cast includes well-known actors and the direction is outstanding.",
        "es": "El elenco incluye actores reconocidos y la dirección es sobresaliente.",
    },
    {
        "en": "I like suspense movies similar to Inception.",
        "es": "Me gustan las películas de suspenso similares a Inception.",
    },
    {
        "en": "Give me a movie recommendation for a relaxing evening.",
        "es": "Dame una recomendación de película para una tarde tranquila.",
    },
    {
        "en": "I want to watch a classic movie from the 1990s.",
        "es": "Quiero ver una película clásica de los años 90.",
    },
    {
        "en": "The movie won several awards for best screenplay.",
        "es": "La película ganó varios premios al mejor guión.",
    },
    {
        "en": "I am in the mood for a dark and intense film.",
        "es": "Tengo ganas de ver una película oscura e intensa.",
    },
    {
        "en": "Are there any good fantasy movies with magic and dragons?",
        "es": "¿Hay buenas películas de fantasía con magia y dragones?",
    },
    {
        "en": "I want a movie with a rating above eight.",
        "es": "Quiero una película con una calificación superior a ocho.",
    },
    {
        "en": "The film received positive reviews from both critics and audiences.",
        "es": "La película recibió críticas positivas tanto de la crítica como del público.",
    },
    {
        "en": "Here are the top five movies that match your preferences.",
        "es": "Aquí están las cinco mejores películas que coinciden con tus preferencias.",
    },
    {
        "en": "This film matches your interest in action and adventure.",
        "es": "Esta película coincide con tu interés en la acción y la aventura.",
    },
    {
        "en": "I recommend this because it fits the genre you requested.",
        "es": "Lo recomiendo porque se ajusta al género que solicitaste.",
    },
    {
        "en": "Sorry, I could not find movies matching your description. Try other words.",
        "es": "Lo siento, no encontré películas que coincidan con tu descripción. Intenta con otras palabras.",
    },
    {"en": "What genre would you like to watch?", "es": "¿Qué género te gustaría ver?"},
    {
        "en": "Do you have a preference for a specific year or era?",
        "es": "¿Tienes preferencia por un año o época específica?",
    },
    {
        "en": "Movies found! Here are my recommendations based on your preferences.",
        "es": "¡Películas encontradas! Aquí están mis recomendaciones basadas en tus preferencias.",
    },
    {
        "en": "This animated film is perfect for a family movie night.",
        "es": "Esta película animada es perfecta para una noche de cine en familia.",
    },
    {
        "en": "The plot twist in this movie will leave you speechless.",
        "es": "El giro de trama en esta película te dejará sin palabras.",
    },
    {
        "en": "This blockbuster combines action, comedy and drama perfectly.",
        "es": "Este éxito de taquilla combina perfectamente acción, comedia y drama.",
    },
    {
        "en": "The cinematography and soundtrack make this film unforgettable.",
        "es": "La fotografía y la banda sonora hacen de esta película algo inolvidable.",
    },
    {
        "en": "I would like a mystery film with unexpected twists.",
        "es": "Me gustaría una película de misterio con giros inesperados.",
    },
    {
        "en": "This is one of the highest rated films on the platform.",
        "es": "Esta es una de las películas mejor calificadas en la plataforma.",
    },
    {
        "en": "The director is known for creating visually stunning movies.",
        "es": "El director es conocido por crear películas visualmente impresionantes.",
    },
    {
        "en": "Hello! What kind of movie are you looking for today?",
        "es": "¡Hola! ¿Qué tipo de película estás buscando hoy?",
    },
]


def load_data(csv_path: str = None) -> list:
    """Loads parallel data from CSV or returns built-in sample data."""
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        if "en" not in df.columns or "es" not in df.columns:
            raise ValueError("CSV must have columns named 'en' and 'es'")
        print(f"Loaded {len(df)} rows from {csv_path}")
        return df[["en", "es"]].dropna().to_dict("records")
    print(f"No CSV provided. Using built-in sample data ({len(SAMPLE_DATA)} pairs).")
    return SAMPLE_DATA


def _tokenize(examples, tokenizer, src_lang, tgt_lang, max_length=128):
    """Tokenizes source and target text for seq2seq training."""
    model_inputs = tokenizer(
        examples[src_lang],
        text_target=examples[tgt_lang],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    # Replace padding token id in labels with -100 so loss ignores them
    labels = model_inputs["labels"]
    model_inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in labels
    ]
    return model_inputs


def _compute_bleu(eval_preds, tokenizer):
    """Computes BLEU score on the evaluation set."""
    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": round(result["score"], 2)}


def fine_tune(
    src_lang: str = "en",
    tgt_lang: str = "es",
    csv_path: str = None,
    epochs: int = 3,
    batch_size: int = 8,
) -> str:
    """
    Fine-tunes a MarianMT model on movie-domain text.

    Args:
        src_lang:   Source language code (e.g. 'en')
        tgt_lang:   Target language code (e.g. 'es')
        csv_path:   Optional path to CSV with 'en' and 'es' columns
        epochs:     Number of training epochs
        batch_size: Batch size per device

    Returns:
        Path where the fine-tuned model was saved.
    """
    base_model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    save_path = SAVE_DIR / f"{src_lang}-{tgt_lang}"
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Load base model ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Fine-tuning: {base_model_id}")
    print(f"  Save path  : {save_path}")
    print(f"{'='*55}\n")

    revision = MODEL_REVISIONS.get(base_model_id)
    if revision is None:
        raise ValueError(f"No reviewed model revision configured for {base_model_id}")
    tokenizer = MarianTokenizer.from_pretrained(base_model_id, revision=revision)
    model = MarianMTModel.from_pretrained(base_model_id, revision=revision)

    # ── Load data ────────────────────────────────────────────────────
    data = load_data(csv_path)

    split = max(1, int(len(data) * 0.8))
    train_records = data[:split]
    eval_records = data[split:] if len(data) > split else data[:2]

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    def tokenize_fn(batch):
        """Tokenize a dataset batch for this language direction."""
        return _tokenize(batch, tokenizer, src_lang, tgt_lang)

    train_ds = train_ds.map(
        tokenize_fn, batched=True, remove_columns=[src_lang, tgt_lang]
    )
    eval_ds = eval_ds.map(
        tokenize_fn, batched=True, remove_columns=[src_lang, tgt_lang]
    )

    # ── Training setup ───────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        logging_dir=save_path / "logs",
        logging_steps=5,
        report_to="none",
        fp16=torch.cuda.is_available(),
        metric_for_best_model="bleu",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: _compute_bleu(p, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ────────────────────────────────────────────────────────
    trainer.train()

    # ── Save final model ─────────────────────────────────────────────
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nFine-tuned model saved to: {save_path}\n")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MarianMT for CineAssist")
    parser.add_argument("--src", default="en", help="Source language (default: en)")
    parser.add_argument("--tgt", default="es", help="Target language (default: es)")
    parser.add_argument("--csv", default=None, help="Path to CSV with en/es columns")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    fine_tune(args.src, args.tgt, args.csv, args.epochs, args.batch)
    # Also fine-tune the reverse direction
    fine_tune(args.tgt, args.src, args.csv, args.epochs, args.batch)


if __name__ == "__main__":
    main()
