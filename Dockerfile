# Line 1: Start from official Python image
FROM python:3.11-slim

# Line 2: Set working directory inside container
WORKDIR /app

# Line 3: Copy requirements first
COPY requirements.txt .

# Line 4: Install all libraries
RUN pip install --no-cache-dir -r requirements.txt

# Line 5: Copy everything else
COPY . .

# Line 6: Tell Docker which port our API uses
EXPOSE 8000

# Line 7: Run the API when container starts
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]