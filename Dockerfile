FROM python:3.14-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Run the bot (secrets passed via env vars or docker-compose at runtime)
CMD ["python", "telegram_bot.py"]