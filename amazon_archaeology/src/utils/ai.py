"""
Cost-efficient OpenAI API integration.
Implements strategies to reduce API costs through:
1. Caching
2. Tiered model usage (cheaper models for initial screening)
3. Efficient prompt construction and batching
"""

import os
from typing import Any, Dict, List, Optional, Union
import time
import openai
from openai import OpenAI

from ..config import OPENAI_API_KEY, OPENAI_MODEL_INITIAL, OPENAI_MODEL_VALIDATION
from .cache import cache_openai_response, save_openai_response

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants for cost control
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def get_completion(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    use_cache: bool = True,
    max_tokens: Optional[int] = None,
    screening: bool = True,
    max_cache_age_days: Optional[int] = None
) -> str:
    """
    Get completion from OpenAI with built-in cost optimization.
    
    Args:
        prompt: The text prompt
        model: Model override (will use tiered approach if None)
        temperature: Model temperature (lower = more deterministic)
        use_cache: Whether to use caching
        max_tokens: Maximum tokens to generate
        screening: If True, use initial model for screening, validation model for complex cases
        max_cache_age_days: Max age for cached responses
        
    Returns:
        Text completion from the API
    """
    # Step 1: Select model based on tier
    if model is None:
        model = OPENAI_MODEL_INITIAL if screening else OPENAI_MODEL_VALIDATION
    
    # Step 2: Check cache first
    if use_cache:
        cached_response = cache_openai_response(model, prompt, max_cache_age_days)
        if cached_response:
            return cached_response.get("text", "")
    
    # Step 3: Make API call with retries
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens or 1024,
            )
            
            # Extract the text
            response_text = completion.choices[0].text.strip()
            
            # Cache successful response
            if use_cache:
                save_openai_response(model, prompt, {"text": response_text})
                
            return response_text
        
        except (openai.RateLimitError, openai.APIError, openai.Timeout) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                raise e
    
    # If we get here, all retries failed
    raise Exception("Failed to get completion after multiple retries")

def classify_features(
    features: List[Dict],
    use_cache: bool = True
) -> List[Dict]:
    """
    Classify archaeological features using tiered approach.
    
    Args:
        features: List of feature dictionaries with metadata
        use_cache: Whether to use caching
        
    Returns:
        Features with confidence scores
    """
    # First pass: Use cheaper model for initial screening
    results = []
    for feature in features:
        # Prepare prompt
        prompt = _prepare_classification_prompt(feature)
        
        # Get initial classification with cheaper model
        confidence = float(get_completion(
            prompt=prompt,
            model=OPENAI_MODEL_INITIAL,
            use_cache=use_cache,
            screening=True
        ))
        
        # Add confidence to feature
        feature["initial_confidence"] = confidence
        results.append(feature)
    
    # Second pass: Use more expensive model only for ambiguous cases
    for feature in results:
        if 0.3 <= feature["initial_confidence"] <= 0.7:
            # This is an ambiguous case, use more expensive model
            prompt = _prepare_classification_prompt(feature, detailed=True)
            
            # Get detailed classification
            confidence = float(get_completion(
                prompt=prompt,
                model=OPENAI_MODEL_VALIDATION,  # More expensive model
                use_cache=use_cache,
                screening=False
            ))
            
            feature["final_confidence"] = confidence
        else:
            # Clear case, keep initial confidence
            feature["final_confidence"] = feature["initial_confidence"]
    
    return results

def _prepare_classification_prompt(
    feature: Dict,
    detailed: bool = False
) -> str:
    """
    Prepare a prompt for feature classification.
    
    Args:
        feature: Feature dictionary with metadata
        detailed: Whether to include more details for validation
        
    Returns:
        Formatted prompt string
    """
    # Basic prompt for initial screening
    if not detailed:
        return f"""
        Analyze the following geographical feature and determine if it is likely to be an archaeological site.
        Return a single number between 0 and 1 representing the confidence (0 = natural, 1 = archaeological).
        
        Feature type: {feature.get('type', 'Unknown')}
        Size (meters): {feature.get('size', 'Unknown')}
        Shape: {feature.get('shape', 'Unknown')}
        Elevation profile: {feature.get('elevation', 'Unknown')}
        Vegetation pattern: {feature.get('vegetation', 'Unknown')}
        """
    
    # Detailed prompt for validation
    return f"""
    Conduct a detailed analysis of this geographical feature to determine if it's an archaeological site.
    Return a single number between 0 and 1 representing the confidence (0 = natural, 1 = archaeological).
    
    Feature type: {feature.get('type', 'Unknown')}
    Size (meters): {feature.get('size', 'Unknown')}
    Shape: {feature.get('shape', 'Unknown')}
    Elevation profile: {feature.get('elevation', 'Unknown')}
    Vegetation pattern: {feature.get('vegetation', 'Unknown')}
    
    Water proximity (m): {feature.get('water_proximity', 'Unknown')}
    Terrain context: {feature.get('terrain', 'Unknown')}
    Regional archaeological context: {feature.get('regional_context', 'Unknown')}
    Historical mentions: {feature.get('historical_mentions', 'Unknown')}
    """

def analyze_historical_text(
    text: str,
    region: Dict,
    use_cache: bool = True
) -> List[Dict]:
    """
    Extract potential site references from historical texts.
    
    Args:
        text: Historical text to analyze
        region: Region dictionary with bounds
        use_cache: Whether to use caching
        
    Returns:
        List of potential site references with coordinates and confidence
    """
    prompt = f"""
    Extract potential archaeological site references from this historical text.
    Focus on the Amazon region within these bounds: {region['bounds']}
    Return a list of potential sites in JSON format with:
    - Description: Text description
    - Coordinates: Estimated coordinates (if possible)
    - Confidence: 0-1 score on likelihood this is a real site
    - Sources: Specific text indicating this site's existence
    
    Text: {text[:5000]}  # Limit text length to control token usage
    """
    
    # Use expensive model directly for this complex task
    response = get_completion(
        prompt=prompt,
        model=OPENAI_MODEL_VALIDATION,
        use_cache=use_cache,
        screening=False,
        max_tokens=2048
    )
    
    # Simple parsing - in production, use more robust JSON parsing
    try:
        # Convert response to Python object
        # This is a simplified approach - real implementation would need
        # more robust parsing of the OpenAI response
        import json
        return json.loads(response)
    except:
        # Fallback if response isn't valid JSON
        return [] 