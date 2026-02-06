# Base Image (matching your local Python version)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python libraries (Wheels allow us to skip build-essential)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the APP file (Ensure this matches your actual filename)
COPY app.py .

# Open the port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]