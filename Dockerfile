# استفاده از ایمیج رسمی پایتورک که ابزارهای CUDA را همراه دارد
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# تنظیم پوشه کاری
WORKDIR /app

# نصب پیش‌نیازهای سیستمی (اگر نیاز بود)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# کپی کردن فایل نیازمندی‌ها و نصب کتابخانه‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن تمام کدهای پروژه
COPY . .

# باز کردن پورت سرویس
EXPOSE 8025

# اجرای سرویس با استفاده از Uvicorn
CMD ["python", "app/main.py"]