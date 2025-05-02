import requests
import json
import sys
import tree_sitter_python
from tree_sitter import Parser, Query, Language
from collections import Counter
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# DI-Bench/.cache
PY_LANGUAGE = Language(tree_sitter_python.language())

class PyDependencyResolver:
    def __init__(self, repo_dir, ollama_url="http://localhost:11434"):
        """
        Initialize the Python Dependency Resolver.
        
        Args:
            repo_dir: Directory containing Python repositories
            ollama_url: URL for Ollama API
        """
        self.repo_dir = repo_dir
        self.ollama_url = ollama_url
        
        # Initialize tree-sitter parser for Python
        self.query = PY_LANGUAGE.query("[(import_statement) (import_from_statement)] @import")
        self.parser = Parser(PY_LANGUAGE)
        
        # Initialize RAG components
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.vectorstore = None
        
        # Cache for PyPI metadata
        self.pypi_cache = {}
        
    def find_repositories(self):
        """Find all Python repositories in the specified directory."""
        repos = set()
        base = Path(self.repo_dir)
        python_markers = ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"]
        skip_dirs = set([".venv", "venv", "__pycache__", "node_modules"])
        for git_dir in base.rglob(".git"):
            repo = git_dir.parent
            if skip_dirs.intersection(repo.parts):
                continue
            has_py = any(repo.rglob("*.py"))
            has_marker = any((repo / m).exists() for m in python_markers)
            if has_py or has_marker:
                repos.add(repo)
        return repos
    
    def extract_imports(self, repo_path):
        """
        Extract import statements from Python files in the repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary of import statements and their frequencies
        """
        imports = []
        
        for file_path in repo_path.rglob("*.py"):
            if not file_path.is_file():
                continue

            try:
                # Read raw bytes so we can slice based on Tree-sitter byte offsets
                data = file_path.read_bytes()

                # Parse from bytes
                tree = self.parser.parse(data)

                # For human readability you can decode once if you need the text,
                # but for slicing imports it's easiest to keep them in bytes.
                for capture_name, nodes in self.query.captures(tree.root_node).items():
                    for node in nodes:
                        imp = data[node.start_byte : node.end_byte].decode("utf-8")
                        imports.append(imp)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Count frequency of each import string
        return dict(Counter(imports))
    
    def get_packages(self, imports_data):
        """Extract the package names from the import statements."""
        packages = set()
        for import_statement, frequency in imports_data.items():
            package = None
            if import_statement.startswith('import '):
                # Handle 'import numpy as np'
                package = import_statement.split(' ')[1].split('.')[0].split(',')[0]
            elif import_statement.startswith('from '):
                # Handle 'from numpy import array'
                package = import_statement.split(' ')[1].split('.')[0]
            if package != None and package != "" and package not in sys.stdlib_module_names:
                packages.add(package)
        return list(packages)
    
    def fetch_pypi_metadata(self, package_name):
        """
        Fetch metadata for a package from PyPI.
        
        Args:
            package_name: Name of the Python package
            
        Returns:
            Dictionary with package metadata
        """
        # Return cached data if available
        if package_name in self.pypi_cache:
            return self.pypi_cache[package_name]
        
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                releases = data["releases"].keys()
                metadata = {
                    "name": data["info"]["name"],
                    "summary": data["info"]["summary"],
                    "version": data["info"]["version"],
                    "requires_dist": data["info"].get("requires_dist", []),
                    # "releases": releases,
                }
                # Cache the result
                self.pypi_cache[package_name] = metadata
                return metadata
            else:
                print(f"Failed to fetch PyPI metadata for {package_name}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching PyPI data for {package_name}: {e}")
            return None
    
    def build_knowledge_base(self, packages):
        """
        Build a RAG knowledge base from PyPI metadata.
        
        Args:
            imports_data: Dictionary of import statements and frequencies
            
        Returns:
            FAISS vector store with package metadata
        """
        documents = []
        
        for package_name in packages:            
            metadata = self.fetch_pypi_metadata(package_name)
            
            if metadata:
                # Create document for each package
                doc_text = (
                    f"Package: {metadata['name']}\n"
                    f"Summary: {metadata['summary']}...\n"
                    f"Newest Version: {metadata['version']}\n"
                    f"Requirements: {metadata['requires_dist']}\n"
                    # f"Past Releases: {metadata['releases']}\n"
                )
                
                documents.append(Document(
                    page_content=doc_text,
                    metadata={
                        "package_name": metadata["name"],
                    }
                ))
        
        if not documents:
            print("No valid documents were created. Cannot build knowledge base.")
            return None
        
        # Split documents into chunks
        doc_chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(doc_chunks, self.embeddings)
        self.vectorstore = vectorstore
        
        return vectorstore
    
    def query_knowledge_base(self, k=5):
        """
        Query the RAG knowledge base.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Knowledge base not built. Call build_knowledge_base first.")
        
        query = "Find all unique packages and their versions."

        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def generate_with_ollama(self, prompt, model="llama3.1", temperature=0.7):
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: The prompt to send to the model
            model: Model name in Ollama
            temperature: Generation temperature
            
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                print(f"Error from Ollama API: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None
    
    def generate_dependencies_for_repo(self, repo_path):
        """
        Generate dependencies for a repository using RAG and Ollama.
        
        Args:
            repo_path: Path to the repository
            query: Optional custom query for the RAG system
            
        Returns:
            Analysis results
        """
        print(f"\nAnalyzing repository: {repo_path}")
        
        # Extract imports
        imports = self.extract_imports(repo_path)
        if not imports:
            print("No imports found in repository.")
            return None
        
        print(f"Found {len(imports)} unique import statements")

        # Extract Package names
        packages = self.get_packages(imports)
        print(f"The extracted packages: {packages}")
        
        # Build knowledge base
        kb = self.build_knowledge_base(packages)
        if not kb:
            print("Failed to build knowledge base.")
            return None
        
        # Query the knowledge base
        relevant_docs = self.query_knowledge_base(len(imports))
        
        # Create context for the LLM
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        # print(context)
        # Create the prompt for Ollama
        ollama_prompt = f"""You are a Python code expert. 
Based on the following information about the packages and dependencies used in a Python repository,
return a list of all Python packages with their specific versions, and the python version required to run. 

All necessary packages that need versions:
{', '.join([f"{package}" for package in packages])}

Extra package information from PyPI:
{context}

Respond **only** with the JSON object matching this schema, do not include any additional text:
{{"python_packages": [{{"package": "<String>", "version": "<String>"}}], "python_version": "<String>"}}
"""
        
        # Generate analysis with Ollama
        ollama_ans = self.generate_with_ollama(ollama_prompt)
        
        results = {
            "repository": repo_path,
            "packages_count": len(packages),
            "retrieved_context": context,
            "packages": ollama_ans
        }
        
        return results


def main():
    # Get repository directory from user
    repo_dir = input("Enter the directory containing Python repositories: ")
    
    # Create analyzer
    generator = PyDependencyResolver(repo_dir)
    
    # Find repositories
    repos = generator.find_repositories()
    print(f"Found {len(repos)} Python repositories")
    
    # Initialize results list
    all_results = []
    
    # Analyze each repository
    for repo in repos:
        results = generator.generate_dependencies_for_repo(repo)
        if results:
            all_results.append(results)
            
            # Print analysis
            print("\n" + "="*50)
            print(f"Repository: {results['repository']}")
            print(f"Package count: {results['packages_count']}")
            print("\nPackages:")
            print(results['packages'])
            print("\n" + "="*50)
    
    # Save results to JSON file
    with open("repository_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to repository_analysis_results.json")


if __name__ == "__main__":
    main()