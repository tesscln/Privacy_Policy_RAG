# Privacy Policy Analyzer

A tool that analyzes privacy policies for compliance with GDPR and CNIL regulations. The analyzer provides compliance scores, identifies violations, and offers recommendations for improvement.

## Features

- Compliance analysis against GDPR and CNIL regulations
- Detailed violation detection with severity levels and proofs
- Source-based legal text references
- Compliance score breakdown
- Actionable recommendations

## Setup

### Prerequisites

- Python 3.6 or higher
- Git
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tesscln/Privacy_Policy_RAG.git
cd Privacy_Policy_RAG
```

2. Create and activate a virtual environment:

```bash
# On macOS/Linux:
python -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app:
```bash
streamlit run src/app.py
```

The app will open in your default web browser. If it doesn't open automatically, you can access it at:
- Local URL: http://localhost:8501
- Network URL: http://xxx.xxx.xxx.xxx:8501 (for accessing from other devices on your network)

## Using the Application

1. **Input Your Privacy Policy**:
   - Choose between uploading a .txt file or pasting text directly
   - If uploading a file, ensure it's in .txt format
   - If pasting text, use the text area provided

2. **Analyze**:
   - Click the "Analyze Policy" button
   - Wait for the analysis to complete (this may take a few moments)

3. **Review Results**:
   - Overall Compliance Score: View your policy's compliance percentage
   - Score Components: Check violations, severity, and coverage metrics
   - Detected Violations: Review specific compliance issues with explanations and proofs
   - Relevant Legal Texts: Explore related GDPR articles, recitals, and CNIL texts
   - Recommendations: Get actionable suggestions for improvement

## Troubleshooting

If you encounter any issues:

1. **Model Loading Errors**:
   - Ensure you have a stable internet connection
   - Try restarting the application
   - Check if your Python environment has sufficient memory

2. **File Upload Issues**:
   - Verify the file is in .txt format
   - Try copying and pasting the text directly instead
   - Check the file encoding (UTF-8 recommended)

3. **Performance Issues**:
   - For large privacy policies, give the analysis more time to complete
   - Try breaking down very large policies into smaller sections

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
