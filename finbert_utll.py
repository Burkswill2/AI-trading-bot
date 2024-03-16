# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

labels = ["positive", "negative", "neutral"]


def estimate_sentiment(news):
    if news:
        """
        Data Preparation by Tokenization
            # Pass in the object to tokenize
            # "pt" specifies PyTorch tokenizer
            # Padding=True ensures all tokenized inputs are equal in length
            # to(device) enables GPU acceleration if available
        """
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        """
        Model Inference
            # Pass in the generated token_ids
            # set attention_mask = tokens to narrow focus on data and not padding
            # "logits" for raw unnormalized scoring assignments to each class within the model
        """
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]

        """
        Softmax Application
            # Apply the softmax function to the logits
            # Normalization applied to the raw scores between (0,1)
            # dim=1 
        """
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)

        """
        Derive Probability
            # torch.argmax(result) returns index of the normalized highest probability among all classes
        """
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]  # by default return probability of 0.0 and sentiment of neutral


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(
        ['markets responded negatively to the news!', 'traders were displeased!'])
    print(tensor, sentiment)
    print(torch.cuda.is_available())
