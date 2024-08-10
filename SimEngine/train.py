import torch
import json
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from sentence_transformers import SentenceTransformer, models

# Load and prepare data
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def split_train_test(dataset, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, random_state=42):
    sentences1 = [item["sentence1"] for item in dataset]
    sentences2 = [item["sentence2"] for item in dataset]
    scores = [item["similarity_score"] for item in dataset]
    train_sentences1, temp_sentences1, train_sentences2, temp_sentences2, train_scores, temp_scores = train_test_split(
        sentences1, sentences2,scores, test_size=(1 - train_ratio), random_state=random_state)
    dev_size = dev_ratio / (dev_ratio + test_ratio)
    dev_sentences1, test_sentences1, dev_sentences2, test_sentences2, dev_scores, test_scores = train_test_split(
        temp_sentences1, temp_sentences2, temp_scores, test_size=dev_size, random_state=random_state)
    train_data = [{"sentence1": s1, "sentence2": s2, 'similarity_score': score} for s1, s2, score in zip(train_sentences1, train_sentences2, train_scores)]
    dev_data = [{"sentence1": s1, "sentence2": s2, 'similarity_score': score} for s1, s2, score in zip(dev_sentences1, dev_sentences2, dev_scores)]
    test_data = [{"sentence1": s1, "sentence2": s2, 'similarity_score': score} for s1, s2, score in zip(test_sentences1, test_sentences2, test_scores)]
    return {"train": train_data, "dev": dev_data, "test": test_data}

# Model
class BertForSTS(torch.nn.Module):
    def __init__(self, model_name='CAMeL-Lab/bert-base-arabic-camelbert-msa', max_seq_length=128):
        super(BertForSTS, self).__init__()
        self.bert = models.Transformer(model_name, max_seq_length=max_seq_length)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_bert(input_data)['sentence_embedding']
        return output
    
    def save_pretrained(self, save_path):
        self.sts_bert.save(save_path)

def load_tokenizer(model_name='CAMeL-Lab/bert-base-arabic-camelbert-msa'):
    return BertTokenizer.from_pretrained(model_name)

# Loss
class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, loss_fn=torch.nn.MSELoss(), transform_fn=torch.nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fn = loss_fn
        self.transform_fn = transform_fn
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, inputs, labels):
        emb_1 = torch.stack([inp[0] for inp in inputs])
        emb_2 = torch.stack([inp[1] for inp in inputs])
        outputs = self.transform_fn(self.cos_similarity(emb_1, emb_2))
        return self.loss_fn(outputs, labels.squeeze())

# Dataset
class STSBDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text_pair = (data['sentence1'], data['sentence2'])
        labels = torch.tensor(data['similarity_score'])
        inputs = self.tokenizer(text_pair, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        return inputs, labels

def collate_fn(texts):
    input_ids = texts['input_ids']
    attention_masks = texts['attention_mask']
    features = [{'input_ids': input_id, 'attention_mask': attention_mask}
                for input_id, attention_mask in zip(input_ids, attention_masks)]
    return features


# Trainer
class Trainer:
    def __init__(self, model, train_data, val_data, tokenizer, batch_size=8, epochs=8, learning_rate=1e-6, device='mps'):
        self.model = model.to(device)
        self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.total_steps = len(self.train_data) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)
        self.criterion = CosineSimilarityLoss().to(device)

    def train(self):
        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(self.epochs):
            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            for train_data, train_label in tqdm(self.train_data):
                self.model.zero_grad()
                train_data['input_ids'] = train_data['input_ids'].to(self.device)
                train_data['attention_mask'] = train_data['attention_mask'].to(self.device)
                train_data = collate_fn(train_data)
                output = [self.model(feature) for feature in train_data]
                loss = self.criterion(output, train_label.to(self.device))
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            avg_train_loss = total_train_loss / len(self.train_data)
            training_time = time.time() - t0
            t0 = time.time()
            self.model.eval()
            total_eval_loss = 0
            for val_data, val_label in tqdm(self.val_data):
                with torch.no_grad():
                    val_data['input_ids'] = val_data['input_ids'].to(self.device)
                    val_data['attention_mask'] = val_data['attention_mask'].to(self.device)
                    val_data = collate_fn(val_data)
                    output = [self.model(feature) for feature in val_data]
                loss = self.criterion(output, val_label.to(self.device))
                total_eval_loss += loss.item()
            avg_val_loss = total_eval_loss / len(self.val_data)
            validation_time = time.time() - t0
            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        print("Training complete!")
        return training_stats
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        print(f'Model saved to {save_path}')

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model_name = 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
    tokenizer = load_tokenizer(model_name)
    dataset = load_data('finetune.json')
    split_data = split_train_test(dataset)
    train_dataset = STSBDataset(split_data["train"], tokenizer)
    val_dataset = STSBDataset(split_data["dev"], tokenizer)

    model = BertForSTS(model_name)
    trainer = Trainer(model, train_dataset, val_dataset, tokenizer, device=device)
    training_stats = trainer.train()

    print("Training complete!")
    print("Training statistics:")
    for stat in training_stats:
        print(stat)

    # Save the Model
    trainer.save_model('model_finetuned')

if __name__ == "__main__":
    main()
