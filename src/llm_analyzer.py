from transformers import pipeline, logging as transformers_logging, AutoModelForSequenceClassification, AutoTokenizer
from retriever import retrieve_relevant_text
import json
from pathlib import Path
import logging
import torch
import warnings
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress transformer warnings
transformers_logging.set_verbosity_error()

# Suppress ChromaDB warnings about existing embeddings
logging.getLogger('chromadb').setLevel(logging.ERROR)

# Compliance scoring weights
WEIGHTS = {
    'violations': 0.4,    # W_1: Weight for number of violations
    'severity': 0.5,      # W_2: Weight for severity of violations
    'coverage': 0.1       # W_3: Weight for coverage of legal provisions
}

# Define severity levels and their impact
SEVERITY_LEVELS = {
    'critical': 1.0,      # Missing consent, data security breaches
    'high': 0.8,          # Missing DPO, inadequate data retention
    'medium': 0.5,        # Incomplete transparency, missing contact info
    'low': 0.2            # Minor formatting issues, incomplete documentation
}

# Cache for model predictions
@lru_cache(maxsize=1024)
def get_cached_prediction(model_type: str, text: str, aspect: str) -> tuple:
    """Cache model predictions to ensure consistency."""
    classifier = _get_classifier()
    
    try:
        # Create a more specific prompt for legal analysis
        prompt = f"Analyze the following privacy policy text regarding {aspect}: {text}"
        
        # Use zero-shot classification with legal-specific labels
        result = classifier(
            prompt,
            [f"compliant with {aspect}", f"non-compliant with {aspect}"],
            multi_label=False
        )
        
        # Get the prediction and confidence
        is_compliant = result["labels"][0].startswith("compliant")
        confidence = result["scores"][0]
        
        # If non-compliant, get explanation and violation proof
        explanation = ""
        violation_proof = ""
        if not is_compliant:
            # Get explanation with specific issue
            explanation_prompt = f"""Analyze this privacy policy regarding {aspect} and explain the specific issue in one sentence.
            Focus on what exactly is problematic or missing in the policy's treatment of {aspect}.
            Text to analyze: {text}"""
            
            explanation_result = classifier(
                explanation_prompt,
                [
                    f"The policy's {aspect} section is incomplete or missing",
                    f"The policy's {aspect} description is unclear or insufficient",
                    f"The policy's {aspect} approach violates regulations"
                ],
                multi_label=False
            )
            explanation = explanation_result["labels"][0]
            
            # Find relevant sections in the text
            section_prompt = f"""Find the section in this privacy policy that discusses {aspect}.
            If there is no section about {aspect}, respond with 'No section found about {aspect}'.
            If there is a section, quote the EXACT relevant text.
            
            Text to analyze: {text}"""
            
            # First check if there's any relevant section
            has_section = classifier(
                section_prompt,
                [
                    f"Found section about {aspect}",
                    f"No section about {aspect}"
                ],
                multi_label=False
            )
            
            if has_section["labels"][0].startswith("Found"):
                # Split into sentences and find the most relevant ones
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                relevant_sentences = []
                
                for sentence in sentences:
                    relevance = classifier(
                        f"Does this sentence discuss {aspect}? '{sentence}'",
                        [
                            f"discusses {aspect}",
                            f"not about {aspect}"
                        ],
                        multi_label=False
                    )
                    if relevance["labels"][0].startswith("discusses"):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    # Join the relevant sentences
                    quoted_text = '. '.join(relevant_sentences[:2]) + '.'  # Limit to 2 most relevant sentences
                    violation_proof = f"Policy states: \"{quoted_text}\" This is problematic because it"
                    
                    # Get specific reason why this text is problematic
                    reason_prompt = f"""Explain specifically why this text is problematic regarding {aspect}:
                    '{quoted_text}'"""
                    
                    reason_result = classifier(
                        reason_prompt,
                        [
                            "lacks required details",
                            "is too vague or unclear",
                            "contradicts regulations"
                        ],
                        multi_label=False
                    )
                    
                    violation_proof += f" {reason_result['labels'][0]}"
                else:
                    violation_proof = f"The policy mentions {aspect} but provides no specific details"
            else:
                violation_proof = f"The policy does not contain any section addressing {aspect}"
        
        # Adjust confidence based on model type
        if model_type == "legal":
            confidence = min(1.0, confidence * 1.2)  # Boost confidence for legal model
        
        return (is_compliant, confidence, explanation, violation_proof)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return (False, 0.0, "Error in analysis", "No violation proof available due to error")

def _get_classifier():
    """Get or initialize the classifier with improved legal models."""
    if not hasattr(_get_classifier, "instance"):
        device = -1  # Default to CPU
        if torch.cuda.is_available():
            device = 0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                _ = torch.zeros(1).to('mps')
                device = 'mps'
            except:
                logger.warning("MPS device detected but not working, falling back to CPU")
                device = -1
        
        logger.info("Loading legal model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Try loading the specialized legal model first
                model_name = "nlpaueb/legal-bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                _get_classifier.instance = pipeline(
                    "zero-shot-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                logger.info("Successfully loaded legal-bert model")
            except Exception as e:
                logger.error(f"Error loading legal-bert model: {str(e)}")
                try:
                    # Fallback to DeBERTa-v3 which has good performance on legal tasks
                    model_name = "microsoft/deberta-v3-base"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    
                    _get_classifier.instance = pipeline(
                        "zero-shot-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                    logger.info("Successfully loaded DeBERTa-v3 model")
                except Exception as e:
                    logger.error(f"Error loading DeBERTa model: {str(e)}")
                    # Final fallback to a simpler model
                    _get_classifier.instance = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=device
                    )
                    logger.info("Using BART-large-MNLI as fallback model")
    
    return _get_classifier.instance

def calculate_compliance_score(aspect_results, relevant_legal_texts):
    """Calculate compliance score based on multiple factors."""
    try:
        # First, identify which aspects are mentioned in the policy (whether compliant or not)
        covered_aspects = set()
        for aspect, result in aspect_results.items():
            # Consider an aspect covered if we found any text about it (compliant or not)
            if not result.get("violation_proof", "").startswith("The policy does not contain any section"):
                covered_aspects.add(aspect)
        
        # Define related terms for each aspect to improve matching
        aspect_terms = {
            "data collection consent": ["consent", "permission", "agree", "authorize", "opt-in", "collect"],
            "data subject rights": ["right", "access", "rectification", "erasure", "portability", "object"],
            "security measures": ["security", "protection", "safeguard", "encrypt", "confidential"],
            "data retention policy": ["retention", "store", "keep", "delete", "period", "duration"],
            "international transfers": ["transfer", "international", "third country", "cross-border"],
            "breach notification": ["breach", "incident", "notification", "compromise", "violation"],
            "DPO appointment": ["dpo", "data protection officer", "privacy officer"],
            "processing records": ["processing", "record", "documentation", "register"]
        }
        
        # Calculate coverage based on matching terms in legal texts
        gdpr_articles = 0
        gdpr_recitals = 0
        cnil_texts = 0
        
        # Track which aspects are covered by which sources
        aspect_coverage = {aspect: {'gdpr_articles': 0, 'gdpr_recitals': 0, 'cnil_texts': 0} 
                         for aspect in covered_aspects}
        
        # For each legal text, check coverage of aspects
        for text in relevant_legal_texts:
            source = text["source"]
            content = text["content"].lower()
            
            # Check each covered aspect
            for aspect in covered_aspects:
                # Check if any related terms are in the content
                relevant_terms = aspect_terms.get(aspect, [aspect])
                matches_aspect = any(term.lower() in content for term in relevant_terms)
                
                if matches_aspect:
                    if "GDPR_text" in source:
                        aspect_coverage[aspect]['gdpr_articles'] += 1
                        gdpr_articles += 1
                    elif "GDPR_recitals" in source:
                        aspect_coverage[aspect]['gdpr_recitals'] += 1
                        gdpr_recitals += 1
                    elif "CNIL_text" in source:
                        aspect_coverage[aspect]['cnil_texts'] += 1
                        cnil_texts += 1
        
        # Calculate source coverage for each aspect
        aspect_source_coverage = {}
        for aspect in covered_aspects:
            coverage = aspect_coverage[aspect]
            sources_with_matches = (
                (1 if coverage['gdpr_articles'] > 0 else 0) +
                (1 if coverage['gdpr_recitals'] > 0 else 0) +
                (1 if coverage['cnil_texts'] > 0 else 0)
            ) / 3.0
            aspect_source_coverage[aspect] = sources_with_matches
        
        # Overall source coverage is average of aspect coverages
        source_coverage = (sum(aspect_source_coverage.values()) / len(aspect_source_coverage)) if aspect_source_coverage else 0.0
        
        # Calculate quantity coverage based on average matches per aspect
        if covered_aspects:
            avg_matches_per_aspect = (gdpr_articles + gdpr_recitals + cnil_texts) / len(covered_aspects)
            quantity_coverage = min(1.0, avg_matches_per_aspect / 3)  # Expect at least 3 relevant texts per aspect
        else:
            quantity_coverage = 0.0
        
        # Combine source diversity and quantity metrics
        coverage_score = (source_coverage * 0.6) + (quantity_coverage * 0.4)
        
        # Calculate violations and severity
        violations = 0
        severity_score = 0
        
        for aspect, result in aspect_results.items():
            if not result.get("compliant", False):
                violations += 1
                severity = result.get("severity", 0.2)
                severity_score += severity
        
        # Normalize severity score
        severity_score = min(1.0, severity_score / len(aspect_results))
        
        # Calculate final score using the formula
        # Score = 100 - (W1 * violations * 10) - (W2 * severity * 100) + (W3 * coverage * 100)
        final_score = 100 - (
            WEIGHTS['violations'] * violations * 10 +  # Scale violations to 0-100 range
            WEIGHTS['severity'] * severity_score * 100 +
            WEIGHTS['coverage'] * coverage_score * 100
        )
        
        # Ensure score is between 0 and 100
        final_score = max(0, min(100, final_score))
        
        return {
            'score': final_score,
            'details': {
                'coverage_score': coverage_score,
                'coverage_breakdown': {
                    'gdpr_articles': gdpr_articles,
                    'gdpr_recitals': gdpr_recitals,
                    'cnil_texts': cnil_texts,
                    'source_coverage': source_coverage,
                    'quantity_coverage': quantity_coverage,
                    'covered_aspects': list(covered_aspects),
                    'aspect_coverage': aspect_coverage,
                    'aspect_source_coverage': aspect_source_coverage
                },
                'violations': violations,
                'severity_score': severity_score,
                'weights_applied': WEIGHTS
            }
        }
    except Exception as e:
        logger.error(f"Error calculating compliance score: {str(e)}")
        return {
            'score': 0,
            'details': {
                'error': str(e)
            }
        }

def determine_severity(aspect):
    """Determine severity level for different compliance aspects."""
    severity_mapping = {
        'data collection consent': 'critical',
        'security measures': 'critical',
        'data subject rights': 'high',
        'breach notification': 'high',
        'DPO appointment': 'high',
        'data retention policy': 'medium',
        'international transfers': 'medium',
        'processing records': 'low'
    }
    return SEVERITY_LEVELS.get(severity_mapping.get(aspect, 'low'), 0.2)

def analyze_privacy_policy(policy_text):
    """Analyzes a privacy policy and checks compliance against legal texts."""
    try:
        # Get relevant legal texts from both GDPR and CNIL
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            relevant_legal_texts = retrieve_relevant_text(policy_text)
        
        # Define specific compliance aspects to check
        compliance_aspects = [
            "data collection consent",
            "data subject rights",
            "security measures",
            "data retention policy",
            "international transfers",
            "breach notification",
            "DPO appointment",
            "processing records"
        ]
        
        # Analyze each compliance aspect
        aspect_results = {}
        for aspect in compliance_aspects:
            try:
                # Use cached prediction
                is_compliant, confidence, explanation, violation_proof = get_cached_prediction("legal", policy_text, aspect)
                
                aspect_results[aspect] = {
                    "compliant": is_compliant,
                    "confidence": confidence,
                    "severity": determine_severity(aspect),
                    "explanation": explanation,
                    "violation_proof": violation_proof
                }
            except Exception as e:
                logger.error(f"Error analyzing aspect '{aspect}': {str(e)}")
                aspect_results[aspect] = {
                    "compliant": False,
                    "confidence": 0.0,
                    "severity": determine_severity(aspect),
                    "error": str(e)
                }
        
        # Calculate compliance score
        compliance_score = calculate_compliance_score(aspect_results, relevant_legal_texts)
        
        # Format the results
        result = {
            "overall_compliance": {
                "score": compliance_score['score'],
                "status": "Compliant" if compliance_score['score'] > 70 else "Partially Compliant" if compliance_score['score'] > 50 else "Non-compliant",
                "details": compliance_score['details']
            },
            "aspect_analysis": aspect_results,
            "relevant_legal_texts": relevant_legal_texts,
            "recommendations": generate_recommendations(aspect_results)
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in privacy policy analysis: {str(e)}")
        raise

def generate_recommendations(aspect_results):
    """Generate specific recommendations based on compliance analysis."""
    try:
        recommendations = []
        
        # Only include non-compliant aspects
        non_compliant_aspects = [
            (aspect, result) 
            for aspect, result in aspect_results.items() 
            if not result.get("compliant", False)
        ]
        
        # Sort by severity and take top 3 most critical issues
        sorted_aspects = sorted(
            non_compliant_aspects,
            key=lambda x: x[1].get("severity", 0),
            reverse=True
        )[:3]
        
        aspect_descriptions = {
            "data collection consent": "explicit user consent for data collection",
            "data subject rights": "user rights to access, modify, and delete data",
            "security measures": "appropriate data protection measures",
            "data retention policy": "clear data retention periods",
            "international transfers": "compliant international data transfers",
            "breach notification": "data breach notification procedures",
            "DPO appointment": "Data Protection Officer appointment",
            "processing records": "records of processing activities"
        }
        
        for aspect, result in sorted_aspects:
            severity = result.get("severity", 0.5)
            priority = "Critical" if severity > 0.8 else "High" if severity > 0.5 else "Medium"
            
            recommendations.append({
                "priority": priority,
                "description": f"Improve {aspect_descriptions[aspect]}",
                "severity": severity
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return []
