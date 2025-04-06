# Use an official Python runtime as the base image
FROM python:3.11.4-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies

RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
# ENV FLASK_APP=app.py
# ENV FLASK_ENV=production

# Run the application
# CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]

CMD ["python","app.py"]