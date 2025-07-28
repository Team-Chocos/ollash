import os
import json
import subprocess
import pandas as pd
import numpy as np
from datasets import load_dataset
import time
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import difflib
import re
from collections import Counter
from huggingface_hub import login
from pathlib import Path
import docker
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
login(token=os.environ["HUGGINGFACE_TOKEN"])

class EnhancedNL2SHBenchmark:
    def __init__(self):
        """Initialize the enhanced NL2SH benchmark system with paper's evaluation methods"""
        self.results = []
        self.benchmark_dir = Path.home() / '.ollash' / 'benchmarks'
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TF-IDF vectorizer as alternative to sentence transformers
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize ICL examples
        self.icl_examples = None
        
        # Define recommended models for 8GB RAM setup
        self.recommended_models = [
            'llama3.2:3b',
            'llama3.2:1b',
            'codellama:7b',
            'deepseek-coder:6.7b',
            'qwen2.5-coder:7b',
            'phi3:mini',
            'mistral:7b',
            'granite-code:3b',
        ]
        
        # Bash utilities for constrained decoding
        self.bash_utilities = [
            'find', 'grep', 'awk', 'sed', 'ls', 'cat', 'sort', 'uniq', 
            'head', 'tail', 'cut', 'tr', 'wc', 'chmod', 'chown', 'cp', 
            'mv', 'rm', 'mkdir', 'rmdir', 'tar', 'gzip', 'curl', 'wget',
            'ps', 'kill', 'top', 'df', 'du', 'mount', 'umount', 'ln',
            'touch', 'which', 'whereis', 'locate', 'file', 'stat'
        ]
        
    def get_recommended_models(self) -> List[str]:
        """Get the list of recommended models for 8GB RAM setup"""
        return self.recommended_models.copy()
        
    def load_test_dataset(self) -> List[Dict]:
        """Load the NL2SH test dataset"""
        try:
            ds = load_dataset("westenfelder/NL2SH-ALFA", "test")
            
            if "train" in ds:
                data = []
                for item in tqdm(ds["train"], desc="Loading dataset items"):
                    data.append({
                        "nl": item["nl"], 
                        "sh": item["bash"],
                        "sh_alt": item["bash2"],
                        "difficulty": item["difficulty"]
                    })
                
                logger.info(f"Successfully loaded {len(data)} samples")
                return data
            else:
                logger.error("No train split found in dataset")
                return []
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    

    # ================= PAPER'S FUNCTIONAL EQUIVALENCE HEURISTICS =================
    
    def bleu_feh(self, predicted: str, actual: str, threshold: float = 0.75) -> bool:
        """BLEU-based functional equivalence heuristic from the paper"""
        if not predicted or not actual:
            return False
        
        try:
            reference = [actual.split()]
            candidate = predicted.split()
            bleu_score = sentence_bleu(reference, candidate)
            return bleu_score >= threshold
        except Exception:
            return False

    def execute_command_safely(self, command: str, timeout: int = 30) -> str:
        """Execute command in Docker container (like InterCode-ALFA)"""
        try:
            client = docker.from_env()
            container = client.containers.run(
                'ubuntu:20.04',
                command=['bash', '-c', command],
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
                timeout=timeout
            )
            return container.decode('utf-8') if isinstance(container, bytes) else str(container)
        except Exception as e:
            logger.warning(f"Command execution failed: {e}")
            return ""

    def compare_outputs_with_tfidf(self, output1: str, output2: str) -> float:
        """Compare command outputs using TF-IDF (replacing sentence transformers)"""
        try:
            if not output1 and not output2:
                return 1.0
            if not output1 or not output2:
                return 0.0
            
            # Use TF-IDF vectorizer
            documents = [output1, output2]
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                # Calculate cosine similarity
                similarity = np.dot(tfidf_matrix[0].toarray(), tfidf_matrix[1].toarray().T)[0, 0]
                return float(similarity)
            except:
                # Fallback to simple text similarity if TF-IDF fails
                return self._simple_text_similarity(output1, output2)
        except Exception:
            return 0.0

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity as fallback"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Use difflib for similarity
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def evaluate_outputs_with_llm(self, task: str, pred_cmd: str, actual_cmd: str, 
                                 pred_output: str, actual_output: str) -> float:
        """Use LLM to evaluate functional equivalence (paper's best method - 95% accuracy)"""
        prompt = f"""You will be given a task, two Bash commands, and their outputs. 
Determine if the second command accomplishes the task. Return only 'true' or 'false'.

Task: {task}
Ground Truth Command: {actual_cmd}
Model Command: {pred_cmd}
Ground Truth Output: {actual_output}
Model Command Output: {pred_output}

Answer (true/false):"""
        
        try:
            response = self.query_ollama_model('llama3.1:8b', prompt)
            return 1.0 if response.strip().lower() == 'true' else 0.0
        except Exception:
            return 0.0

    def execution_based_feh(self, predicted: str, actual: str, nl_query: str) -> Dict[str, float]:
        """Execution + LLM evaluation (paper's best method - 95% accuracy)"""
        try:
            # Execute both commands in controlled environment
            pred_output = self.execute_command_safely(predicted)
            actual_output = self.execute_command_safely(actual)
            
            # Method 1: Execution + TF-IDF (replacing mxbai-embed)
            tfidf_score = self.compare_outputs_with_tfidf(pred_output, actual_output)
            
            # Method 2: Execution + LLM evaluation (95% accuracy from paper)  
            llm_score = self.evaluate_outputs_with_llm(nl_query, predicted, actual, 
                                                      pred_output, actual_output)
            
            return {
                'execution_tfidf_score': tfidf_score,
                'execution_llm_score': llm_score,
                'functional_equivalent': llm_score >= 0.8,
                'pred_output': pred_output,
                'actual_output': actual_output
            }
        except Exception as e:
            logger.warning(f"Execution-based evaluation failed: {e}")
            return {
                'execution_tfidf_score': 0.0, 
                'execution_llm_score': 0.0, 
                'functional_equivalent': False,
                'pred_output': '',
                'actual_output': ''
            }

    # ================= PAPER'S TRANSLATION ENHANCEMENT METHODS =================
    
    def constrained_decoding_query(self, model_name: str, prompt: str) -> str:
        """Apply constrained decoding - constrain first token to Bash utilities"""
        response = self.query_ollama_model(model_name, prompt)
        
        # Post-process to prefer bash utilities if response doesn't start with one
        if response and not any(response.strip().split()[0] in self.bash_utilities for _ in [None]):
            words = response.split()
            for i, word in enumerate(words):
                if word in self.bash_utilities:
                    response = ' '.join(words[i:])
                    break
        
        return response

    def create_icl_examples_with_tfidf(self, training_data: List[Dict], k: int = 25) -> List[Dict]:
        """Create ICL examples using TF-IDF and k-means clustering (replacing sentence transformers)"""
        if len(training_data) < k:
            return training_data
        
        try:
            # Create TF-IDF embeddings for commands
            commands = [item['sh'] for item in training_data]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(commands)
            
            # Convert sparse matrix to dense for k-means
            embeddings = tfidf_matrix.toarray()
            
            # Cluster embeddings
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Select representative examples (closest to centroids)
            examples = []
            for i in range(k):
                cluster_indices = np.where(clusters == i)[0]
                if len(cluster_indices) > 0:
                    # Find closest to centroid
                    centroid = kmeans.cluster_centers_[i]
                    distances = [np.linalg.norm(embeddings[idx] - centroid) for idx in cluster_indices]
                    closest_idx = cluster_indices[np.argmin(distances)]
                    examples.append(training_data[closest_idx])
            
            return examples
        except Exception as e:
            logger.warning(f"ICL example creation failed: {e}")
            # Fallback: select diverse examples based on command variety
            return self._fallback_icl_selection(training_data, k)

    def _fallback_icl_selection(self, training_data: List[Dict], k: int) -> List[Dict]:
        """Fallback ICL selection based on command diversity"""
        # Group by first command word (utility)
        command_groups = {}
        for item in training_data:
            first_word = item['sh'].split()[0] if item['sh'].split() else 'unknown'
            if first_word not in command_groups:
                command_groups[first_word] = []
            command_groups[first_word].append(item)
        
        # Select diverse examples
        examples = []
        utilities = list(command_groups.keys())
        
        # Round-robin selection from different utilities
        while len(examples) < k and utilities:
            for utility in utilities[:]:
                if command_groups[utility] and len(examples) < k:
                    examples.append(command_groups[utility].pop(0))
                if not command_groups[utility]:
                    utilities.remove(utility)
        
        return examples[:k]

    def icl_enhanced_query(self, model_name: str, prompt: str, examples: List[Dict]) -> str:
        """Query with in-context learning examples"""
        icl_prompt = "Your task is to translate natural language to Bash commands.\n\n"
        
        # Add examples
        for example in examples[:25]:  # Limit to 25 as per paper
            icl_prompt += f"Input: {example['nl']}\nOutput: {example['sh']}\n\n"
        
        icl_prompt += f"Input: {prompt}\nOutput: "
        
        return self.query_ollama_model(model_name, icl_prompt)

    def load_icl_examples(self):
        """Load and prepare ICL examples"""
        training_data = self.load_training_dataset()
        if training_data:
            self.icl_examples = self.create_icl_examples_with_tfidf(training_data, k=25)
            logger.info(f"Loaded {len(self.icl_examples)} ICL examples")
        else:
            logger.warning("No training data available for ICL examples")

    # ================= CORE BENCHMARK METHODS =================
    
    def unload_ollama_model(self, model_name: str) -> bool:
        """Unload an Ollama model from memory"""
        try:
            logger.info(f"Unloading model from memory: {model_name}")
            result = subprocess.run(['ollama', 'stop', model_name], 
                                capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Successfully unloaded {model_name}")
                return True
            else:
                logger.warning(f"Failed to unload {model_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error unloading {model_name}: {e}")
            return False

    def unload_all_models(self) -> bool:
        """Unload all models from memory"""
        try:
            logger.info("Unloading all models from memory")
            result = subprocess.run(['ollama', 'stop'], 
                                capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("Successfully unloaded all models")
                return True
            else:
                logger.warning(f"Failed to unload all models: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error unloading all models: {e}")
            return False
    
    def get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                logger.error(f"Failed to get Ollama models: {result.stderr}")
                return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []
    
    def pull_ollama_model(self, model_name: str) -> bool:
        """Pull an Ollama model if not already available"""
        try:
            logger.info(f"Pulling model: {model_name}")
            with tqdm(total=1, desc=f"Pulling {model_name}") as pbar:
                result = subprocess.run(['ollama', 'pull', model_name], 
                                      capture_output=True, text=True, timeout=600)
                pbar.update(1)
                
            if result.returncode == 0:
                logger.info(f"Successfully pulled {model_name}")
                return True
            else:
                logger.error(f"Failed to pull {model_name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout pulling {model_name}")
            return False
        except Exception as e:
            logger.error(f"Error pulling {model_name}: {e}")
            return False
    
    def pull_missing_models(self, required_models: List[str]) -> List[str]:
        """Pull any missing models from the required list"""
        available_models = self.get_available_ollama_models()
        missing_models = [model for model in required_models if model not in available_models]
        
        if not missing_models:
            print("All required models are already available!")
            return required_models
        
        print(f"\nNeed to download {len(missing_models)} missing models:")
        for i, model in enumerate(missing_models, 1):
            print(f"  {i}. {model}")
        
        confirm = input(f"\nProceed with downloading {len(missing_models)} models? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Skipping model downloads. Will only test available models.")
            return [model for model in required_models if model in available_models]
        
        print(f"\nDownloading {len(missing_models)} models...")
        successfully_pulled = []
        
        with tqdm(missing_models, desc="Downloading models", unit="model") as pbar:
            for model in pbar:
                pbar.set_description(f"Downloading {model}")
                if self.pull_ollama_model(model):
                    successfully_pulled.append(model)
                else:
                    logger.error(f"Failed to download {model}")
                pbar.set_postfix_str(f"Success: {len(successfully_pulled)}/{len(missing_models)}")
        
        final_available = self.get_available_ollama_models()
        return [model for model in required_models if model in final_available]
    
    def query_ollama_model(self, model_name: str, prompt: str, max_retries: int = 3) -> str:
        """Query an Ollama model with a prompt"""
        system_prompt = """You are a shell command generator. Your ONLY task is to convert natural language to shell commands.

STRICT RULES:
- Output ONLY the shell command, nothing else
- No explanations, no descriptions, no prefixes like "Output:", "Sure", "Here's the command"
- No conversational responses like "I'm ready!"
- Just the raw shell command that would work in a terminal
- If unsure, provide the most common/standard command

EXAMPLES:
Input: list files in current directory
ls

Input: copy file hello.php to hello-COPY.php  
cp hello.php hello-COPY.php

Input: find all .txt files
find . -name "*.txt"

Now convert this natural language to shell command:"""
        
        full_prompt = f"{system_prompt}\nInput: {prompt}\n"
        
        for attempt in range(max_retries):
            try:
                result = subprocess.run([
                    'ollama', 'run', model_name, full_prompt
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    
                    # Enhanced cleaning
                    unwanted_prefixes = [
                        "output:", "here's the command:", "sure,", "i'm ready!", 
                        "the shell command is:", "here's the shell command:",
                        "you can use:", "try this:", "use this command:",
                        "the command would be:", "here you go:", "certainly!",
                        "of course!", "here it is:", "this should work:",
                    ]
                    
                    lines = response.split('\n')
                    cleaned_response = ""
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        line_lower = line.lower()
                        is_unwanted = any(prefix in line_lower for prefix in unwanted_prefixes)
                        
                        if not is_unwanted and line:
                            for prefix in unwanted_prefixes:
                                if line_lower.startswith(prefix):
                                    line = line[len(prefix):].strip()
                                    break
                            
                            # Fixed: Remove markdown code blocks properly
                            line = line.replace('``````', '').strip()
                            
                            if line and not line.startswith('#'):
                                cleaned_response = line
                                break
                    
                    return cleaned_response if cleaned_response else response
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout on attempt {attempt + 1} for {model_name}")
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1} for {model_name}: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(1)
        
        logger.error(f"All attempts failed for {model_name}")
        return ""

    # ================= ENHANCED BENCHMARK METHODS =================
    
    def benchmark_model_enhanced(self, model_name: str, test_data: List[Dict], 
                                sample_size: int = None, methods: List[str] = None) -> Dict:
        """Enhanced benchmark using paper's methods"""
        logger.info(f"Enhanced benchmarking model: {model_name}")
        
        if sample_size:
            test_data = test_data[:sample_size]
        
        if methods is None:
            methods = ['baseline', 'parsing', 'icl']  # Default methods
        
        results = {
            'model': model_name,
            'predictions': [],
            'total_samples': len(test_data),
            'methods_tested': methods
        }
        
        # Initialize metrics for each method
        for method in methods:
            results[f'{method}_execution_tfidf_scores'] = []
            results[f'{method}_execution_llm_scores'] = []
            results[f'{method}_functional_equivalence_rates'] = []
        
        with tqdm(test_data, desc=f"Enhanced testing {model_name}") as pbar:
            for item in pbar:
                nl_query = item['nl']
                expected_sh = item['sh']
                expected_alt = item.get('sh_alt', '')
                
                # Test different methods
                prediction_result = {
                    'nl': nl_query,
                    'expected': expected_sh,
                    'expected_alt': expected_alt,
                    'predictions': {}
                }
                
                for method in methods:
                    # Get prediction based on method
                    if method == 'baseline':
                        prediction = self.query_ollama_model(model_name, nl_query)
                    elif method == 'constrained_decoding':
                        prediction = self.constrained_decoding_query(model_name, nl_query)
                    elif method == 'parsing':
                        prediction = self.query_ollama_model(model_name, nl_query)
                        # Apply markdown parsing
                        prediction = self.extract_command_from_markdown(prediction)
                    elif method == 'icl':
                        if self.icl_examples:
                            prediction = self.icl_enhanced_query(model_name, nl_query, self.icl_examples)
                        else:
                            prediction = self.query_ollama_model(model_name, nl_query)
                    else:
                        prediction = self.query_ollama_model(model_name, nl_query)
                    
                    prediction_result['predictions'][method] = prediction
                    
                    # Evaluate using paper's execution-based FEH
                    if prediction:
                        eval_result = self.execution_based_feh(prediction, expected_sh, nl_query)
                        
                        results[f'{method}_execution_tfidf_scores'].append(eval_result['execution_tfidf_score'])
                        results[f'{method}_execution_llm_scores'].append(eval_result['execution_llm_score'])
                        results[f'{method}_functional_equivalence_rates'].append(1.0 if eval_result['functional_equivalent'] else 0.0)
                        
                        # Store evaluation details
                        prediction_result[f'{method}_eval'] = eval_result
                    else:
                        results[f'{method}_execution_tfidf_scores'].append(0.0)
                        results[f'{method}_execution_llm_scores'].append(0.0)
                        results[f'{method}_functional_equivalence_rates'].append(0.0)
                        
                        prediction_result[f'{method}_eval'] = {
                            'execution_tfidf_score': 0.0,
                            'execution_llm_score': 0.0,
                            'functional_equivalent': False
                        }
                
                results['predictions'].append(prediction_result)
                
                # Update progress with current best method score
                if methods and results[f'{methods[0]}_execution_llm_scores']:
                    current_avg = np.mean(results[f'{methods[0]}_execution_llm_scores'])
                    pbar.set_postfix_str(f"Avg LLM Score: {current_avg:.3f}")
        
        # Calculate aggregate metrics for each method
        for method in methods:
            results[f'avg_{method}_tfidf_score'] = np.mean(results[f'{method}_execution_tfidf_scores'])
            results[f'avg_{method}_llm_score'] = np.mean(results[f'{method}_execution_llm_scores'])
            results[f'avg_{method}_functional_equivalence'] = np.mean(results[f'{method}_functional_equivalence_rates'])
        
        return results

    def extract_command_from_markdown(self, text: str) -> str:
        """Extract command from markdown code blocks"""
        if not text:
            return ""
        
        # Look for code blocks
        code_block_patterns = [
            r'``````',  # Fixed: Added closing quote
            r'`([^`]+)`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Fixed: matches is a list, need to get first match
                command = matches[0].strip()
                if command:
                    # Remove any remaining markdown artifacts
                    command = command.replace('bash\n', '').replace('sh\n', '').replace('shell\n', '')
                    return command.strip()
        
        # If no code blocks found, return cleaned text
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.lower().startswith(('here', 'the', 'you')):
                return line
        
        return text.strip()
    
    def run_enhanced_benchmark(self, models: List[str], sample_size: int = None, 
                              methods: List[str] = None) -> List[Dict]:
        """Run enhanced benchmark with paper's evaluation methods"""
        # Load test dataset
        logger.info("Loading test dataset...")
        test_data = self.load_test_dataset()
        if not test_data:
            logger.error("Failed to load test dataset")
            return []
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Load ICL examples
        if methods is None or 'icl' in methods:
            logger.info("Loading ICL examples...")
            self.load_icl_examples()
        
        # Clear any existing models from memory
        self.unload_all_models()
        
        all_results = []
        
        with tqdm(models, desc="Benchmarking models", unit="model", ncols=100) as model_pbar:
            for model in model_pbar:
                model_pbar.set_description(f"Processing {model}")
                
                try:
                    # Pull model if not available
                    available_models = self.get_available_ollama_models()
                    if model not in available_models:
                        model_pbar.set_postfix_str("Pulling model...")
                        if not self.pull_ollama_model(model):
                            logger.error(f"Failed to pull model {model}, skipping...")
                            continue
                    
                    # Run enhanced benchmark
                    model_pbar.set_postfix_str("Running benchmark...")
                    model_results = self.benchmark_model_enhanced(model, test_data, sample_size, methods)
                    all_results.append(model_results)
                    
                    # Save intermediate results
                    intermediate_filename = f"enhanced_results_{int(time.time())}.json"
                    self.save_results(all_results, intermediate_filename)
                    
                    # Update progress with results
                    if methods:
                        best_method_score = max([model_results[f'avg_{method}_llm_score'] for method in methods])
                        model_pbar.set_postfix_str(f"Best Score: {best_method_score:.3f}")
                    
                finally:
                    # Unload model after testing
                    model_pbar.set_postfix_str("Unloading model...")
                    self.unload_ollama_model(model)
                    time.sleep(2)
        
        # Final cleanup
        self.unload_all_models()
        
        return all_results
    
    def create_enhanced_leaderboard(self, results: List[Dict], methods: List[str] = None) -> pd.DataFrame:
        """Create enhanced leaderboard DataFrame from results"""
        if methods is None:
            methods = ['baseline', 'parsing', 'icl']
        
        leaderboard_data = []
        
        for result in results:
            row_data = {'Model': result['model']}
            
            for method in methods:
                if f'avg_{method}_llm_score' in result:
                    row_data[f'{method.title()} LLM Score'] = result[f'avg_{method}_llm_score']
                    row_data[f'{method.title()} TF-IDF Score'] = result[f'avg_{method}_tfidf_score']
                    row_data[f'{method.title()} Functional Equiv'] = result[f'avg_{method}_functional_equivalence']
            
            row_data['Total Samples'] = result['total_samples']
            leaderboard_data.append(row_data)
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by best LLM score across methods
        llm_score_columns = [col for col in df.columns if 'LLM Score' in col]
        if llm_score_columns:
            df['Best LLM Score'] = df[llm_score_columns].max(axis=1)
            df = df.sort_values('Best LLM Score', ascending=False)
        
        df = df.reset_index(drop=True)
        df.index += 1
        
        return df
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        try:
            filepath = self.benchmark_dir / filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def save_leaderboard(self, leaderboard: pd.DataFrame, filename: str):
        """Save leaderboard to CSV file"""
        try:
            filepath = self.benchmark_dir / filename
            leaderboard.to_csv(filepath, index_label='Rank')
            logger.info(f"Leaderboard saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving leaderboard: {e}")

# ================= UI AND UTILITY FUNCTIONS =================

def print_banner():
    """Print a nice banner for the enhanced benchmark"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    ENHANCED NL2SH BENCHMARK SYSTEM                       ║
║              Integrated with Paper's Evaluation Methods                  ║
║            Natural Language to Shell Command Evaluation                  ║
╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_section_header(title: str):
    """Print a formatted section header"""
    width = 80
    print(f"\n{'═' * width}")
    print(f"║ {title.center(width-4)} ║")
    print(f"{'═' * width}")

def print_method_options():
    """Print available enhancement methods"""
    print(f"\n┌─────────────────────────────────────────────────────────────────┐")
    print(f"│ Enhancement Methods Available (from the paper)                 │")
    print(f"├─────────────────────────────────────────────────────────────────┤")
    print(f"│ 1. baseline           - Standard model output                  │")
    print(f"│ 2. parsing            - Extract commands from markdown         │")
    print(f"│ 3. constrained_decoding - Constrain first token to bash utils │")
    print(f"│ 4. icl                - In-context learning with examples      │")
    print(f"└─────────────────────────────────────────────────────────────────┘")

def main():
    print_banner()
    
    # Initialize enhanced benchmark system
    benchmark = EnhancedNL2SHBenchmark()
    
    print_section_header("ENHANCED EVALUATION METHODS")
    print("This system integrates the paper's evaluation methods:")
    print("-  Execution + LLM evaluation (95% accuracy)")
    print("-  Execution + TF-IDF comparison (replaces sentence transformers)")
    print("-  Multiple translation enhancement methods")
    print("-  Comprehensive functional equivalence testing")
    
    print_section_header("CHECKING AVAILABLE MODELS")
    print("Scanning for locally available Ollama models...")
    available_models = benchmark.get_available_ollama_models()
    recommended_models = benchmark.get_recommended_models()
    
    print(f"Found {len(available_models)} locally available models")
    
    # Model selection (similar to original)
    print(f"\n┌─────────────────────────────────────────────────┐")
    print(f"│ Model Selection Options                         │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ 1. Test recommended models (8GB RAM optimized) │")
    print(f"│ 2. Test all available models                    │")
    print(f"│ 3. Select specific models                       │")
    print(f"└─────────────────────────────────────────────────┘")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        models_to_test = benchmark.pull_missing_models(recommended_models)
    elif choice == "2":
        models_to_test = available_models if available_models else benchmark.pull_missing_models(recommended_models)
    elif choice == "3":
        if available_models:
            print("\nAvailable models:")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            try:
                indices = [int(x.strip()) - 1 for x in input("Enter model numbers (comma-separated): ").split(',')]
                models_to_test = [available_models[i] for i in indices if 0 <= i < len(available_models)]
            except (ValueError, IndexError):
                print("Invalid input. Using recommended models.")
                models_to_test = benchmark.pull_missing_models(recommended_models)
        else:
            models_to_test = benchmark.pull_missing_models(recommended_models)
    else:
        models_to_test = benchmark.pull_missing_models(recommended_models)
    
    if not models_to_test:
        print("No models selected for testing. Exiting.")
        return
    
    print_section_header("ENHANCEMENT METHODS SELECTION")
    print_method_options()
    
    methods_input = input("\nSelect methods to test (comma-separated, e.g., '1,2,4' or 'all'): ").strip()
    
    method_map = {
        '1': 'baseline',
        '2': 'parsing', 
        '3': 'constrained_decoding',
        '4': 'icl'
    }
    
    if methods_input.lower() == 'all':
        methods_to_test = list(method_map.values())
    else:
        try:
            selected_indices = [x.strip() for x in methods_input.split(',')]
            methods_to_test = [method_map[idx] for idx in selected_indices if idx in method_map]
        except:
            print("Invalid input. Using default methods: baseline, parsing, icl")
            methods_to_test = ['baseline', 'parsing', 'icl']
    
    if not methods_to_test:
        methods_to_test = ['baseline', 'parsing', 'icl']
    
    print(f"Selected methods: {', '.join(methods_to_test)}")
    
    print_section_header("DATASET CONFIGURATION")
    print("Loading test dataset...")
    test_data = benchmark.load_test_dataset()
    if test_data:
        total_samples = len(test_data)
        print(f"Dataset loaded successfully: {total_samples} samples")

        sample_input = input(f"\nEnter sample size (or press Enter for full dataset): ").strip()
        if sample_input:
            try:
                sample_size = min(int(sample_input), total_samples)
                print(f"Using sample size: {sample_size}")
            except ValueError:
                print("Invalid input. Using full dataset.")
                sample_size = None
        else:
            sample_size = None
            print(f"Using full dataset: {total_samples} samples")
    else:
        print("Failed to load dataset. Exiting.")
        return
    
    print_section_header("ENHANCED BENCHMARK EXECUTION")
    print(f"Starting enhanced benchmark:")
    print(f"-  Models: {len(models_to_test)}")
    print(f"-  Methods: {len(methods_to_test)} ({', '.join(methods_to_test)})")
    print(f"-  Dataset size: {sample_size or total_samples}")
    print(f"-  Using paper's 95% accuracy evaluation method")
    print(f"-  Using TF-IDF instead of sentence transformers")
    
    # Run enhanced benchmark
    results = benchmark.run_enhanced_benchmark(models_to_test, sample_size, methods_to_test)
    
    if results:
        print_section_header("ENHANCED BENCHMARK RESULTS")
        
        # Create and display enhanced leaderboard
        leaderboard = benchmark.create_enhanced_leaderboard(results, methods_to_test)
        print("\nENHANCED NL2SH BENCHMARK LEADERBOARD")
        print("(Using paper's execution + LLM evaluation method with TF-IDF)")
        print("\n" + str(leaderboard))
        
        # Save results
        benchmark.save_results(results, 'enhanced_final_results.json')
        benchmark.save_leaderboard(leaderboard, 'enhanced_final_leaderboard.csv')
        
        benchmark_dir = Path.home() / '.ollash' / 'benchmarks'
        print(f"\nFiles saved:")
        print(f"   Enhanced results: {benchmark_dir / 'enhanced_final_results.json'}")
        print(f"   Enhanced leaderboard: {benchmark_dir / 'enhanced_final_leaderboard.csv'}")
        
        # Show method comparison for top model
        print_section_header("METHOD COMPARISON")
        if results:
            top_model = max(results, key=lambda x: max([x.get(f'avg_{method}_llm_score', 0) for method in methods_to_test]))
            print(f"Top model: {top_model['model']}")
            print("\nMethod Performance Comparison:")
            for method in methods_to_test:
                llm_score = top_model.get(f'avg_{method}_llm_score', 0)
                func_equiv = top_model.get(f'avg_{method}_functional_equivalence', 0)
                print(f"  {method.ljust(20)}: LLM Score = {llm_score:.3f}, Functional Equiv = {func_equiv:.3f}")
        
        print(f"\nEnhanced benchmark completed successfully!")
        print(f"Results include paper's execution-based evaluation with TF-IDF similarity!")
        
    else:
        print("No results to display")

if __name__ == "__main__":
    main()
