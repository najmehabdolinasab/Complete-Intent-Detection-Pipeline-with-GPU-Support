from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# ۱. انتخاب مدل پایه (یک مدل چندزبانه سبک و عالی برای شروع)
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ۲. بارگذاری داده‌های آموزشی (فرض بر این است که فایل در data/samples.json است)
# برای گیت‌هاب، می‌توانید از یک دیتاست نمونه استفاده کنید
dataset = load_dataset("json", data_files="data/samples.json")["train"]

# ۳. بارگذاری مدل SetFit
model = SetFitModel.from_pretrained(model_id)

# ۴. تنظیمات آموزش
args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# ۵. تعریف Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

# ۶. شروع آموزش
print("🚀 Starting training...")
trainer.train()

# ۷. ذخیره مدل نهایی
model.save_pretrained("model/persian_intent_model")
print("✅ Model saved successfully in 'model/' directory.")