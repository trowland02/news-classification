from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
from scipy.special import softmax

class SentimentModel:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        # Load the model, tokenizer, and config
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def preprocess(self, text: str) -> str:
        """
        Preprocess text to replace usernames and links with placeholders.
        """
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def transform_score(self, sentiment_scores: np.ndarray) -> float:
        """
        Transform the sentiment score into a 1-100 scale based on the predicted label.
        """
        ranking = np.argsort(sentiment_scores)[::-1]
        top_label = self.config.id2label[ranking[0]]

        # Assign score ranges based on sentiment label
        if top_label == 'negative':
            score = np.interp(sentiment_scores[ranking[0]], [0, 1], [1, 33])
        elif top_label == 'neutral':
            score = np.interp(sentiment_scores[ranking[0]], [0, 1], [34, 66])
        elif top_label == 'positive':
            score = np.interp(sentiment_scores[ranking[0]], [0, 1], [67, 100])
        
        return round(score, 2)

    def score_news_content(self, text: str) -> float:
        """
        Preprocess the text, get the sentiment score, and transform it into a 1-100 scale.
        """
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Transform the score to a 1-100 scale
        final_score = self.transform_score(scores)
        return final_score

