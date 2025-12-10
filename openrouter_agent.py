# -*- coding: utf-8 -*-
"""
DOCX Form Filler Agent with OpenRouter + QWEN 8B
"""

import json
import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from docx import Document
from rapidfuzz import fuzz
from pydantic import BaseModel, Field

from openai import OpenAI
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME,
    GENERATION_CONFIG,
    CONFIDENCE_THRESHOLD,
    FUZZY_MATCH_THRESHOLD,
    SITE_URL,
    SITE_NAME
)

# ============================================================================
# Data Models
# ============================================================================

class FieldMapping(BaseModel):
    placeholder: str = Field(description="Placeholder text from document")
    matched_field: str = Field(description="JSON field name to map")
    confidence: float = Field(description="Confidence score 0-1")

class FieldMappingResponse(BaseModel):
    mappings: List[FieldMapping] = Field(description="List of field mappings")
    total_mapped: int = Field(description="Total count of mapped fields")

# ============================================================================
# Input Processing
# ============================================================================

class InputNormalizer:
    """Normalize and load input data from various formats"""
    
    @staticmethod
    def normalize_text(text: Any) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return unicodedata.normalize('NFC', text.strip())
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            InputNormalizer.normalize_text(k): InputNormalizer.normalize_text(v)
            for k, v in data.items()
        }
    
    @staticmethod
    def auto_load(file_path: str) -> Dict[str, str]:
        ext = Path(file_path).suffix.lower()
        if ext == '.json':
            return InputNormalizer.load_json(file_path)
        raise ValueError(f"Unsupported format: {ext}")

# ============================================================================
# Placeholder Detection
# ============================================================================

class PlaceholderDetector:
    """Detect placeholders in DOCX documents"""
    
    PATTERNS = {
        'underscores': re.compile(r'_{5,}'),
        'dots': re.compile(r'\.{4,}'),
        'brackets': re.compile(r'\{[^}]+\}'),
    }
    
    @staticmethod
    def extract_from_paragraphs(paragraphs) -> List[Tuple[str, str, Any]]:
        placeholders = []
        for para in paragraphs:
            text = para.text
            for pattern in PlaceholderDetector.PATTERNS.values():
                for match in pattern.finditer(text):
                    # Extract context (100 chars before/after)
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()
                    placeholders.append((match.group(), context, para))
        return placeholders
    
    @staticmethod
    def extract_all(doc: Document) -> List[Tuple[str, str, Any]]:
        placeholders = PlaceholderDetector.extract_from_paragraphs(doc.paragraphs)
        
        # Extract from tables
        for table in doc.tables or []:
            for row in table.rows:
                for cell in row.cells:
                    placeholders.extend(
                        PlaceholderDetector.extract_from_paragraphs(cell.paragraphs)
                    )
        return placeholders

# ============================================================================
# OpenRouter LLM Mapper
# ============================================================================

class OpenRouterMapper:
    """Map placeholders to JSON fields using OpenRouter + QWEN"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.model = MODEL_NAME
        self.extra_headers = {}
        
        if SITE_URL:
            self.extra_headers["HTTP-Referer"] = SITE_URL
        if SITE_NAME:
            self.extra_headers["X-Title"] = SITE_NAME
    
    def map_fields(
        self,
        json_keys: List[str],
        placeholders: List[Tuple[str, str, Any]]
    ) -> Dict[str, str]:
        """Map JSON fields to document placeholders using QWEN"""
        
        # Prepare sample data (limit to avoid token overflow)
        sample_placeholders = [
            {"placeholder": ph, "context": ctx}
            for ph, ctx, _ in placeholders[:50]
        ]
        
        system_prompt = f"""You are an expert at mapping JSON fields to document placeholders.

Given:
- JSON FIELDS: {json.dumps(json_keys, ensure_ascii=False)}
- DOCUMENT PLACEHOLDERS: {json.dumps(sample_placeholders, ensure_ascii=False)}

Task: Match each placeholder to the most appropriate JSON field based on context.

Return ONLY valid JSON in this format:
{{
    "mappings": [
        {{"placeholder": "____", "matched_field": "field_name", "confidence": 0.95}}
    ],
    "total_mapped": 5
}}

Rules:
1. Only include mappings with confidence > 0.7
2. Match based on semantic meaning in context
3. Handle Romanian text and diacritics
4. Return valid JSON only, no explanations"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Create the mapping now."}
                ],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                max_tokens=GENERATION_CONFIG["max_tokens"],
                extra_headers=self.extra_headers
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Clean markdown code blocks if present
            content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE)
            
            data = json.loads(content)
            mapping_obj = FieldMappingResponse(**data)
            
            # Build valid mapping dictionary
            valid_mapping = {}
            for fm in mapping_obj.mappings:
                if fm.matched_field in json_keys and fm.confidence > CONFIDENCE_THRESHOLD:
                    valid_mapping[fm.placeholder] = fm.matched_field
            
            print(f"  ✓ OpenRouter mapped {len(valid_mapping)} fields")
            return valid_mapping
            
        except Exception as e:
            print(f"  ⚠ OpenRouter failed: {e}. Using fuzzy fallback.")
            return {}

# ============================================================================
# Fuzzy Matcher (Fallback)
# ============================================================================

class FuzzyMatcher:
    """Fuzzy matching fallback for unmapped fields"""
    
    ROMANIAN_DIACRITICS = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
    }
    
    @staticmethod
    def normalize_romanian(text: str) -> str:
        text = text.lower()
        for old, new in FuzzyMatcher.ROMANIAN_DIACRITICS.items():
            text = text.replace(old, new.lower())
        return text
    
    @staticmethod
    def match(
        placeholder_context: str,
        json_keys: List[str],
        threshold: int = FUZZY_MATCH_THRESHOLD
    ) -> Optional[str]:
        norm_context = FuzzyMatcher.normalize_romanian(placeholder_context)
        best_match, best_score = None, 0
        
        for key in json_keys:
            norm_key = FuzzyMatcher.normalize_romanian(key)
            score = max(
                fuzz.partial_ratio(norm_context, norm_key),
                fuzz.token_sort_ratio(norm_context, norm_key)
            )
            if score > best_score and score >= threshold:
                best_score, best_match = score, key
        
        return best_match

# ============================================================================
# Document Renderer
# ============================================================================

class DocxRenderer:
    """Replace placeholders in DOCX while preserving formatting"""
    
    @staticmethod
    def replace_in_paragraph(para, placeholder: str, value: str) -> bool:
        full_text = para.text
        if placeholder not in full_text:
            return False
        
        # Handle multi-line values
        if '\n' in value:
            value = value.replace('\n', ' ')
        
        placeholder_start = full_text.find(placeholder)
        
        # Find affected runs
        affected_runs = []
        current_pos = 0
        
        for i, run in enumerate(para.runs):
            run_start = current_pos
            run_end = current_pos + len(run.text)
            
            if run_start < placeholder_start + len(placeholder) and run_end > placeholder_start:
                affected_runs.append((i, run_start, run_end, run))
            current_pos = run_end
        
        if not affected_runs:
            return False
        
        # Replace in first run (preserves formatting best)
        first_run_idx, first_start, _, first_run = affected_runs[0]
        before = full_text[:placeholder_start]
        after = full_text[placeholder_start + len(placeholder):]
        first_run.text = before + value + after
        
        return True
    
    @staticmethod
    def fill_all_occurrences(doc: Document, placeholder: str, value: str) -> int:
        count = 0
        for para in doc.paragraphs:
            if DocxRenderer.replace_in_paragraph(para, placeholder, value):
                count += 1
        return count

# ============================================================================
# Main Agent
# ============================================================================

class DocxFormFillerAgent:
    """Main agent orchestrating the form filling process"""
    
    def __init__(self, use_llm: bool = True):
        self.normalizer = InputNormalizer()
        self.detector = PlaceholderDetector()
        self.llm_mapper = OpenRouterMapper() if use_llm else None
        self.fuzzy_matcher = FuzzyMatcher()
        self.renderer = DocxRenderer()
    
    def process(
        self,
        docx_path: str,
        data_path: str,
        output_path: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Process document and fill placeholders"""
        
        stats = {
            "total_placeholders": 0,
            "filled_placeholders": 0,
            "llm_mappings": 0,
            "fuzzy_mappings": 0,
            "unmapped": 0
        }
        
        if verbose:
            print("=" * 60)
            print("DOCX FORM FILLER - OpenRouter + QWEN")
            print("=" * 60)
        
        # 1. Load data
        data = self.normalizer.auto_load(data_path)
        if verbose:
            print(f"[1/5] ✓ Loaded {len(data)} fields")
        
        # 2. Load document
        doc = Document(docx_path)
        if verbose:
            print("[2/5] ✓ Document loaded")
        
        # 3. Detect placeholders
        placeholders = self.detector.extract_all(doc)
        stats["total_placeholders"] = len(placeholders)
        if verbose:
            print(f"[3/5] ✓ Found {len(placeholders)} placeholders")
        
        # Group placeholders
        placeholder_groups = {}
        for ph, ctx, obj in placeholders:
            placeholder_groups.setdefault(ph, []).append((ctx, obj))
        
        # 4. Map with OpenRouter
        llm_mapping = {}
        if self.llm_mapper:
            llm_mapping = self.llm_mapper.map_fields(list(data.keys()), placeholders)
            stats["llm_mappings"] = len(llm_mapping)
        
        # 5. Fill document
        if verbose:
            print("[4/5] Filling placeholders...")
        
        for placeholder_text, occurrences in placeholder_groups.items():
            context, _ = occurrences[0]
            matched_key = None
            
            # Try LLM mapping first
            for mapped_ph, field in llm_mapping.items():
                if mapped_ph in placeholder_text or placeholder_text in mapped_ph:
                    matched_key = field
                    break
            
            # Fuzzy fallback
            if not matched_key:
                matched_key = self.fuzzy_matcher.match(context, list(data.keys()))
                if matched_key:
                    stats["fuzzy_mappings"] += 1
            
            if matched_key and matched_key in data:
                value = data[matched_key]
                count = self.renderer.fill_all_occurrences(doc, placeholder_text, value)
                stats["filled_placeholders"] += count
                if verbose:
                    print(f"  ✓ [{count}x] {placeholder_text[:30]}... → {matched_key[:30]}")
            else:
                stats["unmapped"] += 1
        
        # 6. Save
        doc.save(output_path)
        if verbose:
            print(f"[5/5] ✓ Saved: {output_path}")
        
        if verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            for k, v in stats.items():
                print(f"{k}: {v}")
            print("=" * 60)
        
        return stats
