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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
login(token=os.environ["HUGGINGFACE_TOKEN"])

class NL2SHBenchmark:
    def __init__(self):
        """Initialize the NL2SH benchmark system with lightweight similarity metrics"""
        self.results = []
        # Create benchmarks directory
        self.benchmark_dir = Path.home() / '.ollash' / 'benchmarks'
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Define recommended models for 8GB RAM setup
        self.recommended_models = [
            'llama3.2:3b',      # ~2GB - Latest Llama, good performance
            'llama3.2:1b',      # ~1GB - Smallest Llama, fast inference
            'codellama:7b',     # ~4GB - Best coding model in this size
            'deepseek-coder:6.7b', # ~4GB - Excellent for coding tasks
            'qwen2.5-coder:7b', # ~4GB - Strong coding performance
            'phi3:mini',        # ~2GB - Microsoft's efficient model
            'mistral:7b',       # ~4GB - Good general performance
            'granite-code:3b',  # ~2GB - IBM's code model
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
                # Add tqdm for dataset loading
                for item in tqdm(ds["train"], desc="Loading dataset items"):
                    # Include both bash commands for more comprehensive evaluation
                    data.append({
                        "nl": item["nl"], 
                        "sh": item["bash"],          # Primary command
                        "sh_alt": item["bash2"],     # Alternative command
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
        
    def unload_ollama_model(self, model_name: str) -> bool:
        """Unload an Ollama model from memory"""
        try:
            logger.info(f"Unloading model from memory: {model_name}")
            # Stop the specific model
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
            # Stop all running models
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
            # Use tqdm for model pulling progress (though subprocess doesn't show internal progress)
            with tqdm(total=1, desc=f"Pulling {model_name}") as pbar:
                result = subprocess.run(['ollama', 'pull', model_name], 
                                      capture_output=True, text=True, timeout=600)  # Increased timeout for larger models
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
        
        # Return all available models (existing + successfully pulled)
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
                    
                    # Enhanced cleaning - remove common unwanted prefixes/responses
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
                            
                        # Skip lines that are just unwanted responses
                        line_lower = line.lower()
                        is_unwanted = any(prefix in line_lower for prefix in unwanted_prefixes)
                        
                        if not is_unwanted and line:
                            # Remove any remaining prefixes from the line
                            for prefix in unwanted_prefixes:
                                if line_lower.startswith(prefix):
                                    line = line[len(prefix):].strip()
                                    break
                            
                            # Remove markdown code blocks
                            line = line.replace('``````', '').strip()
                            
                            if line and not line.startswith('#'):  # Avoid comments
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
                time.sleep(1)  # Wait before retry
        
        logger.error(f"All attempts failed for {model_name}")
        return ""
    
    def normalize_command(self, command: str) -> str:
        """Normalize shell command for better comparison"""
        # Remove extra whitespace
        command = ' '.join(command.split())
        # Convert to lowercase for case-insensitive comparison
        command = command.lower()
        return command
    
    def sequence_similarity(self, predicted: str, actual: str) -> float:
        """Compute sequence similarity using difflib"""
        if not predicted or not actual:
            return 0.0
        
        predicted_norm = self.normalize_command(predicted)
        actual_norm = self.normalize_command(actual)
        
        # Use difflib's SequenceMatcher for similarity
        similarity = difflib.SequenceMatcher(None, predicted_norm, actual_norm).ratio()
        return float(similarity)
    
    def jaccard_similarity(self, predicted: str, actual: str) -> float:
        """Compute Jaccard similarity based on word tokens"""
        if not predicted or not actual:
            return 0.0
        
        predicted_norm = self.normalize_command(predicted)
        actual_norm = self.normalize_command(actual)
        
        # Split into words/tokens
        pred_tokens = set(predicted_norm.split())
        actual_tokens = set(actual_norm.split())
        
        # Jaccard similarity = intersection / union
        intersection = len(pred_tokens.intersection(actual_tokens))
        union = len(pred_tokens.union(actual_tokens))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def command_structure_similarity(self, predicted: str, actual: str) -> float:
        """Compute similarity based on command structure (commands, flags, patterns)"""
        if not predicted or not actual:
            return 0.0
        
        def extract_features(command):
            features = {
                'commands': [],
                'flags': [],
                'operators': [],
                'patterns': []
            }
            
            # Extract commands (first word and words after pipes/operators)
            words = command.split()
            if words:
                features['commands'].append(words[0])
            
            # Extract flags (words starting with -)
            for word in words:
                if word.startswith('-'):
                    features['flags'].append(word)
            
            # Extract operators
            operators = ['|', '&&', '||', ';', '>', '>>', '<', '&']
            for op in operators:
                if op in command:
                    features['operators'].append(op)
            
            # Extract file patterns (words with wildcards)
            for word in words:
                if '*' in word or '?' in word or '[' in word:
                    features['patterns'].append(word)
            
            return features
        
        pred_features = extract_features(self.normalize_command(predicted))
        actual_features = extract_features(self.normalize_command(actual))
        
        # Calculate weighted similarity
        similarities = []
        weights = {'commands': 0.4, 'flags': 0.3, 'operators': 0.2, 'patterns': 0.1}
        
        for feature_type, weight in weights.items():
            pred_set = set(pred_features[feature_type])
            actual_set = set(actual_features[feature_type])
            
            if not pred_set and not actual_set:
                sim = 1.0
            elif not pred_set or not actual_set:
                sim = 0.0
            else:
                intersection = len(pred_set.intersection(actual_set))
                union = len(pred_set.union(actual_set))
                sim = intersection / union if union > 0 else 0.0
            
            similarities.append(sim * weight)
        
        return sum(similarities)
    
    def compute_composite_similarity(self, predicted: str, actual: str, actual_alt: str = None) -> Dict[str, float]:
        """Compute multiple similarity metrics and create a composite score"""
        if not predicted:
            return {
                'sequence_similarity': 0.0,
                'jaccard_similarity': 0.0,
                'structure_similarity': 0.0,
                'composite_score': 0.0
            }
        
        # Calculate similarities for primary command
        similarities = self._calculate_similarities(predicted, actual)
        
        # If alternative command exists, calculate similarities and take the better score
        if actual_alt and actual_alt.strip():
            alt_similarities = self._calculate_similarities(predicted, actual_alt)
            
            # Take the maximum score for each metric
            similarities = {
                key: max(similarities[key], alt_similarities[key])
                for key in similarities.keys()
            }
        
        return similarities

    def _calculate_similarities(self, predicted: str, actual: str) -> Dict[str, float]:
        """Helper method to calculate individual similarities"""
        if not actual:
            return {
                'sequence_similarity': 0.0,
                'jaccard_similarity': 0.0,
                'structure_similarity': 0.0,
                'composite_score': 0.0
            }
        
        seq_sim = self.sequence_similarity(predicted, actual)
        jaccard_sim = self.jaccard_similarity(predicted, actual)
        struct_sim = self.command_structure_similarity(predicted, actual)
        composite = (seq_sim * 0.3 + jaccard_sim * 0.3 + struct_sim * 0.4)
        
        return {
            'sequence_similarity': seq_sim,
            'jaccard_similarity': jaccard_sim,
            'structure_similarity': struct_sim,
            'composite_score': composite
        }
    
    def exact_match_score(self, predicted: str, actual: str) -> float:
        """Compute exact match score"""
        if not predicted or not actual:
            return 0.0
        
        # Normalize both commands for comparison
        pred_norm = self.normalize_command(predicted)
        actual_norm = self.normalize_command(actual)
        
        return 1.0 if pred_norm == actual_norm else 0.0
    
    def functional_equivalence_score(self, predicted: str, actual: str) -> float:
        """Simple heuristic for functional equivalence"""
        if not predicted or not actual:
            return 0.0
        
        # Check if commands achieve similar functionality
        # This is a simplified heuristic - you could expand this
        
        pred_norm = self.normalize_command(predicted)
        actual_norm = self.normalize_command(actual)
        
        # Extract base commands
        pred_cmd = pred_norm.split()[0] if pred_norm.split() else ""
        actual_cmd = actual_norm.split()[0] if actual_norm.split() else ""
        
        # Same base command is a good start
        if pred_cmd == actual_cmd:
            # If same command, check flag similarity
            pred_flags = set(word for word in pred_norm.split() if word.startswith('-'))
            actual_flags = set(word for word in actual_norm.split() if word.startswith('-'))
            
            if pred_flags and actual_flags:
                flag_sim = len(pred_flags.intersection(actual_flags)) / len(pred_flags.union(actual_flags))
                return 0.7 + 0.3 * flag_sim  # Base score + flag bonus
            else:
                return 0.7  # Same command, different or no flags
        
        return 0.0
    
    def benchmark_model(self, model_name: str, test_data: List[Dict], 
                       sample_size: int = None) -> Dict:
        """Benchmark a single model on the test dataset"""
        logger.info(f"Benchmarking model: {model_name}")
        
        if sample_size:
            test_data = test_data[:sample_size]
        
        results = {
            'model': model_name,
            'predictions': [],
            'sequence_similarities': [],
            'jaccard_similarities': [],
            'structure_similarities': [],
            'composite_scores': [],
            'exact_matches': [],
            'functional_equivalence': [],
            'total_samples': len(test_data),
            'successful_predictions': 0
        }
        
        # Enhanced tqdm with more details
        with tqdm(test_data, desc=f"Testing {model_name}", 
                 unit="queries", ncols=100, leave=True) as pbar:
            for i, item in enumerate(pbar):
                nl_query = item['nl']
                expected_sh = item['sh']
                expected_alt = item.get('sh_alt', '')
                
                # Update progress bar with current query info
                pbar.set_postfix_str(f"Query: {nl_query[:30]}...")
                
                # Get prediction from model
                predicted_sh = self.query_ollama_model(model_name, nl_query)
                
                if predicted_sh:
                    results['successful_predictions'] += 1
                    
                    # Compute all similarity metrics (include alternative command)
                    similarities = self.compute_composite_similarity(predicted_sh, expected_sh, expected_alt)
                    exact_match = self.exact_match_score(predicted_sh, expected_sh)
                    func_equiv = self.functional_equivalence_score(predicted_sh, expected_sh)
                    
                    prediction_result = {
                        'nl': nl_query,
                        'expected': expected_sh,
                        'expected_alt': expected_alt,
                        'predicted': predicted_sh,
                        **similarities,
                        'exact_match': exact_match,
                        'functional_equivalence': func_equiv
                    }
                    
                    results['predictions'].append(prediction_result)
                    results['sequence_similarities'].append(similarities['sequence_similarity'])
                    results['jaccard_similarities'].append(similarities['jaccard_similarity'])
                    results['structure_similarities'].append(similarities['structure_similarity'])
                    results['composite_scores'].append(similarities['composite_score'])
                    results['exact_matches'].append(exact_match)
                    results['functional_equivalence'].append(func_equiv)
                else:
                    empty_result = {
                        'nl': nl_query,
                        'expected': expected_sh,
                        'expected_alt': expected_alt,
                        'predicted': '',
                        'sequence_similarity': 0.0,
                        'jaccard_similarity': 0.0,
                        'structure_similarity': 0.0,
                        'composite_score': 0.0,
                        'exact_match': 0.0,
                        'functional_equivalence': 0.0
                    }
                    results['predictions'].append(empty_result)
                    results['sequence_similarities'].append(0.0)
                    results['jaccard_similarities'].append(0.0)
                    results['structure_similarities'].append(0.0)
                    results['composite_scores'].append(0.0)
                    results['exact_matches'].append(0.0)
                    results['functional_equivalence'].append(0.0)
                
                # Update progress bar with current metrics
                if results['composite_scores']:
                    current_avg = np.mean(results['composite_scores'])
                    pbar.set_postfix_str(f"Avg Score: {current_avg:.3f}")
        
        # Calculate aggregate metrics
        results['avg_sequence_similarity'] = np.mean(results['sequence_similarities'])
        results['avg_jaccard_similarity'] = np.mean(results['jaccard_similarities'])
        results['avg_structure_similarity'] = np.mean(results['structure_similarities'])
        results['avg_composite_score'] = np.mean(results['composite_scores'])
        results['exact_match_rate'] = np.mean(results['exact_matches'])
        results['functional_equivalence_rate'] = np.mean(results['functional_equivalence'])
        results['success_rate'] = results['successful_predictions'] / len(test_data)
        
        logger.info(f"Completed {model_name}: "
                   f"Composite Score: {results['avg_composite_score']:.3f}, "
                   f"Exact Match: {results['exact_match_rate']:.3f}, "
                   f"Success Rate: {results['success_rate']:.3f}")
        
        return results
    
    def run_benchmark(self, models: List[str], sample_size: int = None) -> List[Dict]:
        """Run benchmark on multiple models"""
        # Load test dataset
        logger.info("Loading test dataset...")
        test_data = self.load_test_dataset()
        if not test_data:
            logger.error("Failed to load test dataset")
            return []
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Clear any existing models from memory before starting
        self.unload_all_models()
        
        all_results = []
        
        # Add tqdm for model iteration
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
                    
                    # Run benchmark
                    model_pbar.set_postfix_str("Running benchmark...")
                    model_results = self.benchmark_model(model, test_data, sample_size)
                    all_results.append(model_results)
                    
                    # Save intermediate results with timestamp
                    intermediate_filename = f"intermediate_results_{int(time.time())}.json"
                    self.save_results(all_results, intermediate_filename)
                    
                    # Update model progress bar with current results
                    model_pbar.set_postfix_str(f"Score: {model_results['avg_composite_score']:.3f}")
                    
                finally:
                    # IMPORTANT: Unload the model after testing to free memory
                    model_pbar.set_postfix_str("Unloading model...")
                    self.unload_ollama_model(model)
                    # Small delay to ensure cleanup
                    time.sleep(2)
        
        # Final cleanup - ensure all models are unloaded
        self.unload_all_models()
        
        return all_results
    
    def create_leaderboard(self, results: List[Dict]) -> pd.DataFrame:
        """Create a leaderboard DataFrame from results"""
        leaderboard_data = []
        
        for result in results:
            leaderboard_data.append({
                'Model': result['model'],
                'Composite Score': result['avg_composite_score'],
                'Sequence Similarity': result['avg_sequence_similarity'],
                'Jaccard Similarity': result['avg_jaccard_similarity'],
                'Structure Similarity': result['avg_structure_similarity'],
                'Exact Match Rate': result['exact_match_rate'],
                'Functional Equiv Rate': result['functional_equivalence_rate'],
                'Success Rate': result['success_rate'],
                'Total Samples': result['total_samples']
            })
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by composite score (primary) and exact match rate (secondary)
        df = df.sort_values(['Composite Score', 'Exact Match Rate'], ascending=False)
        df = df.reset_index(drop=True)
        df.index += 1  # Start ranking from 1
        
        return df
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file in the benchmarks directory"""
        try:
            filepath = self.benchmark_dir / filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def save_leaderboard(self, leaderboard: pd.DataFrame, filename: str):
        """Save leaderboard to CSV file in the benchmarks directory"""
        try:
            filepath = self.benchmark_dir / filename
            leaderboard.to_csv(filepath, index_label='Rank')
            logger.info(f"Leaderboard saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving leaderboard: {e}")

    def format_leaderboard_display(self, leaderboard: pd.DataFrame) -> str:
        """Format leaderboard for better terminal display"""
        # Create a formatted string representation
        header = "┌─────┬─────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐"
        separator = "├─────┼─────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤"
        footer = "└─────┴─────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘"
        
        # Column headers
        col_header = "│ Rank│ Model                   │ Composite Score  │ Sequence Sim     │ Jaccard Sim      │ Structure Sim    │ Exact Match Rate │ Functional Equiv │ Success Rate     │ Total Samples    │"
        
        formatted_output = f"{header}\n{col_header}\n{separator}\n"
        
        for i, row in leaderboard.iterrows():
            rank = str(i).center(4)
            model = row['Model'][:23].ljust(23)
            comp_score = f"{row['Composite Score']:.3f}".center(16)
            seq_sim = f"{row['Sequence Similarity']:.3f}".center(16)
            jaccard_sim = f"{row['Jaccard Similarity']:.3f}".center(16)
            struct_sim = f"{row['Structure Similarity']:.3f}".center(16)
            exact_match = f"{row['Exact Match Rate']:.3f}".center(16)
            func_equiv = f"{row['Functional Equiv Rate']:.3f}".center(16)
            success_rate = f"{row['Success Rate']:.3f}".center(16)
            total_samples = str(int(row['Total Samples'])).center(16)
            
            row_line = f"│ {rank}│ {model} │ {comp_score} │ {seq_sim} │ {jaccard_sim} │ {struct_sim} │ {exact_match} │ {func_equiv} │ {success_rate} │ {total_samples} │"
            formatted_output += f"{row_line}\n"
        
        formatted_output += footer
        return formatted_output

def print_banner():
    """Print a nice banner for the benchmark"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                          NL2SH BENCHMARK SYSTEM                          ║
║                  Natural Language to Shell Command Evaluation            ║
╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_section_header(title: str):
    """Print a formatted section header"""
    width = 80
    print(f"\n{'═' * width}")
    print(f"║ {title.center(width-4)} ║")
    print(f"{'═' * width}")

def print_model_list(models: List[str], title: str):
    """Print a formatted list of models"""
    print(f"\n┌─────┬─────────────────────────────────────────────┐")
    print(f"│  #  │ {title.ljust(43)} │")
    print(f"├─────┼─────────────────────────────────────────────┤")
    for i, model in enumerate(models, 1):
        print(f"│ {str(i).rjust(2)}  │ {model.ljust(43)} │")
    print(f"└─────┴─────────────────────────────────────────────┘")

def print_model_list_with_details(models: List[str], title: str, available_models: List[str]):
    """Print a formatted list of models with availability status"""
    print(f"\n┌─────┬─────────────────────────────────────────────┬──────────────┐")
    print(f"│  #  │ {title.ljust(43)} │ Status       │")
    print(f"├─────┼─────────────────────────────────────────────┼──────────────┤")
    for i, model in enumerate(models, 1):
        status = "Available" if model in available_models else "Need Download"
        status_color = status.ljust(12)
        print(f"│ {str(i).rjust(2)}  │ {model.ljust(43)} │ {status_color} │")
    print(f"└─────┴─────────────────────────────────────────────┴──────────────┘")

def print_example_predictions(predictions: List[Dict], model_name: str):
    """Print formatted example predictions"""
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ Example Predictions from Top Model: {model_name.ljust(60)} │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    for i, pred in enumerate(predictions[:3], 1):
        print(f"\n┌── Query {i} " + "─" * 70)
        print(f"│ Input:      {pred['nl']}")
        print(f"│ Expected:   {pred['expected']}")
        if pred.get('expected_alt'):
            print(f"│ Expected Alt: {pred['expected_alt']}")
        print(f"│ Predicted:  {pred['predicted']}")
        print(f"│ Score:      {pred['composite_score']:.3f}")
        print(f"└" + "─" * 77)

# Example usage
def main():
    print_banner()
    
    # Initialize benchmark system
    benchmark = NL2SHBenchmark()
    
    print_section_header("CHECKING AVAILABLE MODELS")
    print("Scanning for locally available Ollama models...")
    available_models = benchmark.get_available_ollama_models()
    
    # Get recommended models
    recommended_models = benchmark.get_recommended_models()
    
    print(f"Found {len(available_models)} locally available models")
    
    # Enhanced model selection options
    print(f"\n┌─────────────────────────────────────────────────┐")
    print(f"│ Model Selection Options                         │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ 1. Test recommended models (8GB RAM optimized) │")
    print(f"│ 2. Test all available models                    │")
    print(f"│ 3. Select specific models                       │")
    print(f"└─────────────────────────────────────────────────┘")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Recommended models option
        print_section_header("RECOMMENDED MODELS FOR 8GB RAM")
        print("These models are optimized for 8GB RAM systems and provide excellent performance:")
        print_model_list_with_details(recommended_models, "Recommended Models", available_models)
        
        print(f"\nRecommended setup details:")
        print(f"• Total models: {len(recommended_models)}")
        print(f"• Memory optimized for 8GB RAM")
        print(f"• Mix of general and code-specialized models")
        print(f"• Includes latest Llama 3.2, CodeLlama, DeepSeek-Coder, etc.")
        
        # Check which models need to be downloaded
        models_to_test = benchmark.pull_missing_models(recommended_models)
        if not models_to_test:
            print("No models available for testing. Exiting.")
            return
        
        print(f"\nFinal model list for testing: {len(models_to_test)} models")
        for i, model in enumerate(models_to_test, 1):
            print(f"  {i}. {model}")
            
    elif choice == "2":
        # All available models
        if available_models:
            print_model_list(available_models, "Available Models")
            models_to_test = available_models
            print(f"Selected all {len(models_to_test)} available models for testing")
        else:
            print("No locally available models found. Using recommended list.")
            models_to_test = benchmark.pull_missing_models(recommended_models)
            
    elif choice == "3":
        # Custom selection
        if available_models:
            print_model_list(available_models, "Available Models")
            print("\nEnter model numbers to test (comma-separated, e.g., 1,3,5):")
            try:
                indices = [int(x.strip()) - 1 for x in input().split(',')]
                models_to_test = [available_models[i] for i in indices if 0 <= i < len(available_models)]
                print(f"Selected {len(models_to_test)} models for testing")
            except (ValueError, IndexError):
                print("Invalid input. Using recommended models.")
                models_to_test = benchmark.pull_missing_models(recommended_models)
        else:
            print("No locally available models found. Using recommended list.")
            models_to_test = benchmark.pull_missing_models(recommended_models)
    else:
        print("Invalid choice. Using recommended models.")
        models_to_test = benchmark.pull_missing_models(recommended_models)
    
    if not models_to_test:
        print("No models selected for testing. Exiting.")
        return
    
    print_section_header("DATASET CONFIGURATION")
    print("Loading test dataset to determine size...")
    test_data = benchmark.load_test_dataset()
    if test_data:
        total_samples = len(test_data)
        print(f"Dataset loaded successfully")
        print(f"Total samples available: {total_samples}")

        sample_input = input(f"\nEnter sample size for testing (or press Enter for full dataset of {total_samples}): ").strip()
        if sample_input:
            try:
                sample_size = int(sample_input)
                sample_size = min(sample_size, total_samples)  # Don't exceed available samples
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
    
    print_section_header("BENCHMARK EXECUTION")
    print(f"Starting benchmark with {len(models_to_test)} models")
    if sample_size:
        print(f"Sample size: {sample_size}")
    else:
        print(f"Full dataset: {total_samples} samples")
    
    results = benchmark.run_benchmark(models_to_test, sample_size)
    
    if results:
        print_section_header("BENCHMARK RESULTS")
        
        # Create and display leaderboard
        leaderboard = benchmark.create_leaderboard(results)
        print("\nNL2SH BENCHMARK LEADERBOARD")
        print(benchmark.format_leaderboard_display(leaderboard))
        
        # Save final results (overwrites each time)
        benchmark.save_results(results, 'final_results.json')
        benchmark.save_leaderboard(leaderboard, 'final_leaderboard.csv')
        
        benchmark_dir = Path.home() / '.ollash' / 'benchmarks'
        print(f"\nFiles saved:")
        print(f"   Final results: {benchmark_dir / 'final_results.json'}")
        print(f"   Final leaderboard: {benchmark_dir / 'final_leaderboard.csv'}")
        print(f"   Intermediate results: {benchmark_dir}")
        
        # Print example predictions from top model
        print_section_header("EXAMPLE PREDICTIONS")
        top_model_results = sorted(results, key=lambda x: x['avg_composite_score'], reverse=True)[0]
        print_example_predictions(top_model_results['predictions'], top_model_results['model'])
        
        print(f"\nBenchmark completed successfully!")
        print(f"Top performing model: {top_model_results['model']} (Score: {top_model_results['avg_composite_score']:.3f})")
        
    else:
        print("No results to display")

if __name__ == "__main__":
    main()
