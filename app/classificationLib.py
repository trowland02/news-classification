from transformers import BertTokenizer, BertForSequenceClassification
import torch

class NewsClassifier:
    def __init__(self, model_name: str = "cssupport/bert-news-class"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)

        self.id_to_class = {
            0: 'arts', 1: 'arts & culture', 2: 'black voices', 3: 'business', 4: 'college', 5: 'comedy',
            6: 'crime', 7: 'culture & arts', 8: 'education', 9: 'entertainment', 10: 'environment',
            11: 'fifty', 12: 'food & drink', 13: 'good news', 14: 'green', 15: 'healthy living',
            16: 'home & living', 17: 'impact', 18: 'latino voices', 19: 'media', 20: 'money',
            21: 'parenting', 22: 'parents', 23: 'politics', 24: 'queer voices', 25: 'religion',
            26: 'science', 27: 'sports', 28: 'style', 29: 'style & beauty', 30: 'taste', 31: 'tech',
            32: 'the worldpost', 33: 'travel', 34: 'u.s. news', 35: 'weddings', 36: 'weird news',
            37: 'wellness', 38: 'women', 39: 'world news', 40: 'worldpost'
        }

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length').to(self.device)
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])[0]
        pred_class_idx = torch.argmax(logits, dim=1).item()
        return self.id_to_class[pred_class_idx]
