Utils architect to Setup and Run the Teacher Interface
# Teacher Interface Setup Guide

This document provides a comprehensive, step-by-step guide for setting up the Teacher Interface application from scratch. Follow these instructions to get the project running on your local machine.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Initial Setup](#initial-setup)
4. [Environment Configuration](#environment-configuration)
5. [Installing Dependencies](#installing-dependencies)
6. [Firebase Authentication Setup](#firebase-authentication-setup)
7. [Running the Application](#running-the-application)
8. [Verifying the Setup](#verifying-the-setup)
9. [Troubleshooting](#troubleshooting)
10. [Development Workflow](#development-workflow)

## Prerequisites

Before beginning the setup, ensure you have the following software installed on your system:

### Required Software

1. **Python 3.9 or higher**
   - Verify installation: `python3 --version`
   - Download: https://www.python.org/downloads/

2. **Node.js (v14 or higher)**
   - Verify installation: `node --version`
   - Download: https://nodejs.org/

3. **npm (Node Package Manager)**
   - Usually comes with Node.js
   - Verify installation: `npm --version`

4. **Git**
   - Verify installation: `git --version`
   - Download: https://git-scm.com/downloads

### Required Accounts and Access

1. **OpenAI API Key**
   - Required for question generation and evaluation
   - Sign up at: https://platform.openai.com/
   - Create an API key from: https://platform.openai.com/api-keys

2. **MIT Firebase Proxy Credentials** (if applicable)
   - Server URL: ``
   - Username and password for JWT authentication
   - Contact project administrator if you don't have these credentials

3. **Storybook Assets**
   - Local path to the `interactive-storybook-assets` directory
   - Should contain subdirectories: `qna_json/` and `image/`
   - Each storybook should have corresponding JSON and image files

## Project Structure

Understanding the project layout will help during setup:

```
teacher_interface/
├── backend/                          # Python backend
│   ├── server.py                     # Flask application (main entry point)
│   ├── storage/
│   │   ├── teacher_feedback_records.json # Feedback records (auto-generated)
│   │   └── question_evaluations.json     # Evaluation traces (auto-generated)
│   └── services/                     # Core services
│       ├── contextq_evaluators.py
│       ├── gpt-moral-generation-structured.py
│       ├── gpt-objective-questions-structured-DSpy.py
│       ├── question_generator.py
│       └── teacher_feedback_system.py
├── firebase/                         # Firebase integration
│   ├── firebase_client.py            # JWT authentication client
│   └── manifest.json                 # Firebase configuration
├── frontend/                         # React frontend
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── App.tsx
│       ├── components/
│       ├── services/
│       └── types/
├── prompts/                          # LLM prompts (in parent directory)
│   └── moralQ_prompts/
│       └── story_moral_prompt.txt
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (create this)
└── .env.example                      # Environment template

Note: The prompts directory is located in the parent directory, not inside teacher_interface.
```

## Initial Setup

### Step 1: Clone or Navigate to the Repository

If you haven't already, navigate to the project directory:

```bash
cd /path/to/MoralQ/teacher_interface
```

### Step 2: Create a Python Virtual Environment

Python virtual environments isolate project dependencies and prevent conflicts.

```bash
# From the MoralQ root directory (parent of teacher_interface)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

Verify activation: your terminal prompt should show `(venv)` at the beginning.

**Important:** Always activate the virtual environment before running the application.

### Step 3: Verify Python Installation

Ensure Python can access the required modules:

```bash
python3 --version  # Should show 3.9 or higher
pip --version      # Should show pip version
```

## Environment Configuration

### Step 1: Create the Environment File

Create a `.env` file inside the `teacher_interface/` directory:

```bash
cd teacher_interface
touch .env
```

### Step 2: Configure Environment Variables

Open `.env` in a text editor and add the following variables:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Firebase MIT Proxy Configuration
SERVER_URL=
USERNAME=your_username
PASSWORD=your_password
TOP_LEVEL_NODE=your_top_level_node

# Asset Paths
ASSETS_PATH=/path/to/interactive-storybook-assets

# Optional: Output Paths (if running batch scripts)
OUTPUT_PATH=/path/to/output/moral_segments
OBJECTIVE_QUESTIONS_OUTPUT_PATH=/path/to/output/questions
```

**Important:** Replace all placeholder values with your actual credentials and paths.

### Step 3: Verify Paths

Ensure the following paths exist on your system:

1. **ASSETS_PATH**: Should point to your `interactive-storybook-assets` directory
   ```bash
   ls /path/to/interactive-storybook-assets/qna_json/
   # Should list storybook JSON files
   
   ls /path/to/interactive-storybook-assets/image/
   # Should list storybook image files
   ```

2. **Output paths** (if using batch scripts): Create directories if they don't exist:
   ```bash
   mkdir -p /path/to/output/moral_segments
   mkdir -p /path/to/output/questions
   ```

### Step 4: Protect Sensitive Information

Ensure `.env` is not committed to version control:

```bash
# Check if .gitignore includes .env
cat .gitignore | grep .env

# If not present, add it:
echo "teacher_interface/.env" >> .gitignore
```

## Installing Dependencies

### Step 1: Install Python Dependencies

From the `teacher_interface/` directory with the virtual environment activated:

```bash
cd teacher_interface
pip install -r requirements.txt
```

Expected installation packages include:
- flask
- flask-cors
- openai
- dspy-ai
- python-dotenv
- eventlet

### Step 2: Verify Python Package Installation

Check that key packages are installed:

```bash
pip list | grep -E "flask|openai|dspy"
```

### Step 3: Install Frontend Dependencies

Navigate to the frontend directory and install Node.js packages:

```bash
cd frontend
npm install
```

This will install React, TypeScript, Axios, and other frontend dependencies.

### Step 4: Verify Frontend Installation

Check that `node_modules` was created:

```bash
ls -la node_modules | head -20
```

## Firebase Authentication Setup

**Note:** The Firebase integration is currently not actively used in the application. The `firebase_client.py` authenticates with the MIT Firebase proxy server automatically when the backend server starts. You do not need to start Firebase separately.

### Step 1: Understanding Firebase Integration

The Firebase client is embedded within the backend server and handles:
- JWT authentication with the MIT Firebase proxy server
- Student and storybook data retrieval
- Automatic re-authentication as needed

The authentication happens automatically when you start the backend server (see [Running the Application](#running-the-application) section below).

### Step 2: Optional Firebase Testing (Not Required)

If you want to test Firebase authentication independently:

```bash
# From the teacher_interface directory
cd firebase
python3 firebase_client.py
```

Expected output:
```
Authenticating with server: 
JWT authentication successful
Firebase client authenticated successfully
```

### Step 3: Troubleshoot Authentication Issues

If authentication fails during backend startup:

1. **Verify credentials in `.env`**: Check that `SERVER_URL`, `USERNAME`, and `PASSWORD` are correct
2. **Check network connectivity**: Ensure you can reach the server URL
3. **Contact administrator**: If credentials are still not working, contact the project administrator

## Running the Application

### Step 0: (Optional) Start the Firebase Client (separate step)

You can run the Firebase client as its own command to verify connectivity to the MIT proxy. This is not required for normal operation 

```bash
cd /path/to/MoralQ/teacher_interface/firebase
python3 firebase_client.py
```

Expected behavior:
- Authenticates via JWT, prints a success message, and exits. It does not run a long‑lived server.
- The backend will perform the same authentication when you start it, so this step is optional.

### Step 1: Start the Backend Server

Open a terminal window and navigate to the backend directory:

```bash
cd /path/to/MoralQ/teacher_interface/backend
python3 server.py
```

Expected output:
```

Starting Teacher Interface Server...
Storybook Library Dashboard
Access at: http://localhost:5001
Assets path: /path/to/interactive-storybook-assets
```

**Keep this terminal open.** The server must remain running while using the application.

### Step 2: Start the Frontend Server

Open a **new terminal window** and navigate to the frontend directory:

```bash
cd /path/to/MoralQ/teacher_interface/frontend
npm start
```

Expected output:
```
Starting the development server...
Compiled successfully!

You can now view teacher-interface-frontend in the browser.
  Local:            http://localhost:3000
  On Your Network:  http://YOUR_IP:3000

webpack compiled successfully
```

**Keep this terminal open** as well.

### Step 3: Access the Application

Open your web browser and navigate to:

```
http://localhost:3000
```

You should see the Teacher Interface dashboard.

## Verifying the Setup

### Test 1: Backend Health Check

From a new terminal, test if the backend is responding:

```bash
curl http://localhost:5001/api/storybooks
```

Expected response: JSON array of available storybooks.

### Test 2: Frontend-Backend Connection

1. Open the browser to `http://localhost:3000`
2. Check the browser console (F12) for any connection errors
3. The interface should load without errors

### Test 3: Generate Questions

1. Select a storybook from the dropdown
2. Enter an objective (e.g., "Narrative Skills")
3. Click "Generate Moral"
4. Review the generated moral lesson
5. Click "Approve and Generate Questions"
6. Wait for question generation to complete
7. Verify that questions appear in the results panel

### Test 4: Teacher Feedback Flow

1. After questions are generated, mark some questions as "Good" or "Bad"
2. Enter overall feedback (e.g., "Questions are too difficult for this age group")
3. Click "Submit Feedback"
4. Click "Regenerate Questions"
5. Verify that new questions appear based on your feedback

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Module Not Found Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'dspy'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue 2: Port Already in Use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find and kill the process using port 5001
lsof -ti:5001 | xargs kill -9

# Or use a different port by editing server.py
# app.run(port=5002)
```

#### Issue 3: Environment Variables Not Loading

**Symptoms:**
```
KeyError: 'OPENAI_API_KEY'
```

**Solution:**
1. Verify `.env` file exists in `teacher_interface/` directory
2. Check that variable names match exactly (case-sensitive)
3. Restart the backend server after making changes

#### Issue 4: Storybook Assets Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
1. Verify `ASSETS_PATH` in `.env` points to the correct directory
2. Check that the directory contains `qna_json/` and `image/` subdirectories
3. Verify file permissions: `ls -la /path/to/assets`

#### Issue 5: Frontend Cannot Connect to Backend

**Symptoms:**
```
Network Error: Failed to connect to localhost:5001
```

**Solution:**
1. Verify backend is running: `curl http://localhost:5001/api/storybooks`
2. Check CORS configuration in `server.py`
3. Check firewall settings that might block localhost connections

#### Issue 6: OpenAI API Errors

**Symptoms:**
```
openai.error.AuthenticationError: Invalid API key
```

**Solution:**
1. Verify `OPENAI_API_KEY` in `.env` is correct
2. Check that your OpenAI account has sufficient credits
3. Verify API key permissions in OpenAI dashboard

### Debugging Tips

1. **Check Backend Logs**: The terminal running `server.py` shows detailed logs
2. **Check Frontend Console**: Browser DevTools (F12) shows client-side errors
3. **Verify JSON Files**: Check `backend/storage/question_evaluations.json` and `backend/storage/teacher_feedback_records.json` for data
4. **Test Individual Components**: Run scripts in `services/` directory independently

## Development Workflow

### Making Changes to Backend

1. Edit Python files in `backend/` or `backend/services/`
2. The Flask development server auto-reloads on file changes
3. Check backend terminal for any errors

### Making Changes to Frontend

1. Edit TypeScript/React files in `frontend/src/`
2. Webpack hot-reloads changes automatically
3. Browser will refresh to show changes

### Testing Changes

1. **Clear JSON files** (optional, for clean testing):
   ```bash
   echo '{"evaluations": []}' > backend/storage/question_evaluations.json
   mkdir -p backend/storage
   echo "{}" > backend/storage/teacher_feedback_records.json
   echo "{}" > q_testing.json
   ```

2. **Run a complete test flow**: Generate questions, provide feedback, regenerate

3. **Check logs**: Verify that your changes are working as expected

### Code Cleanup

Before committing changes:

1. Remove debug print statements
2. Remove commented-out code
3. Ensure no credentials are hardcoded
4. Run basic functionality tests

## Production Considerations

For deploying this application to a production environment:

1. **Database Migration**: Replace JSON files with a proper database (PostgreSQL, MongoDB, etc.)
2. **Environment Secrets**: Use a secrets manager (AWS Secrets Manager, HashiCorp Vault)
3. **HTTPS**: Configure SSL certificates
4. **Process Management**: Use a process manager (PM2, supervisor) for long-running processes
5. **Logging**: Implement structured logging with a centralized log aggregator
6. **Monitoring**: Set up application performance monitoring (APM) and error tracking
7. **Load Balancing**: Use a reverse proxy (Nginx) for multiple backend instances
8. **Caching**: Implement Redis for caching frequently accessed data
9. **API Rate Limiting**: Add rate limiting to prevent abuse
10. **Security**: Implement authentication and authorization for the API endpoints

## Additional Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **React Documentation**: https://react.dev/
- **DSPy Documentation**: https://dspy-docs.vercel.app/
- **OpenAI API Documentation**: https://platform.openai.com/docs/

## Getting Help

If you encounter issues not covered in this guide:

1. Check the `ARCHITECTURE.md` file for system design details
2. Review the code comments in relevant files
3. Contact the development team or project administrator
4. Check the GitHub issues (if project is on GitHub)

---

**Last Updated:** October 2025

This setup guide is maintained with the project. If you discover issues or improvements, please update this document accordingly.

