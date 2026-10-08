#!/usr/bin/env python3
"""Module that extracts an answer snippet from a reference text using BERT."""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): The reference document containing the answer.

    Returns:
        str: The answer snippet, or None if no valid answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + [
        '[SEP]'
    ]
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)

    type_cls = [0] * len(['[CLS]'])
    type_question = [0] * len(question_tokens)
    type_sep1 = [0] * len(['[SEP]'])
    type_reference = [1] * len(reference_tokens)
    type_sep2 = [1] * len(['[SEP]'])

    input_type_ids = (
        type_cls + type_question + type_sep1 + type_reference + type_sep2
    )

    input_word_ids, input_mask, input_type_ids = map(
        lambda x: tf.expand_dims(tf.constant(x, dtype=tf.int32), 0),
        [input_word_ids, input_mask, input_type_ids]
    )

    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    if short_start > short_end or short_start >= len(tokens):
        return None

    answer_tokens = tokens[short_start:short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    if not answer:
        return None

    return answer
