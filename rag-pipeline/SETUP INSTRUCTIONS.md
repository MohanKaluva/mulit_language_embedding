"""
COMPLETE SETUP GUIDE:

1. CREATE PROJECT STRUCTURE:
   mkdir -p rag-pipeline/app/api rag-pipeline/app/core rag-pipeline/app/utils
   mkdir -p rag-pipeline/config rag-pipeline/data/{raw,processed,index}
   mkdir -p rag-pipeline/tests

2. CREATE ALL FILES:
   Copy each section above into its respective file following the 
   "FILE: path/to/file.py" headers

3. INSTALL DEPENDENCIES:
   cd rag-pipeline
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

4. CONFIGURE:
   cp .env.example .env
   # Edit .env as needed

5. RUN SERVER:
   python run_server.py
   # OR
   uvicorn app.main:app --reload

6. TEST API:
   - Open http://localhost:8000/docs
   - Upload a test file
   - Try a query

7. RUN TESTS:
   pytest tests/test_api.py -v

That's it! Your RAG pipeline API is ready! 🚀
"""