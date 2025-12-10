# -*- coding: utf-8 -*-
"""
DOCX Form Filler - FIXED: Match labels, not patterns
"""

import json
import re
import unicodedata
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from docx import Document
from rapidfuzz import fuzz
from pydantic import BaseModel, Field
from openai import OpenAI

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME,
    GENERATION_CONFIG, CONFIDENCE_THRESHOLD, DEBUG_MODE
)

# ============================================================================
# Models
# ============================================================================

class FieldMapping(BaseModel):
    label: str = Field(description="Label text before placeholder")
    json_key: str = Field(description="Exact JSON key")
    confidence: float = Field(ge=0.0, le=1.0)

class MappingResponse(BaseModel):
    mappings: List[FieldMapping]

# ============================================================================
# Input Normalizer
# ============================================================================

class InputNormalizer:
    @staticmethod
    def normalize_text(text) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return unicodedata.normalize('NFC', text.strip())
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalized = {}
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            normalized[InputNormalizer.normalize_text(k)] = InputNormalizer.normalize_text(v)
        
        return normalized

# ============================================================================
# Placeholder Detector - Extract LABEL + PLACEHOLDER
# ============================================================================

class PlaceholderDetector:
    PATTERNS = {
        'underscores': re.compile(r'_{5,}'),
        'dots': re.compile(r'\.{4,}'),
    }
    
    @staticmethod
    def extract_label_and_placeholder(text: str, match) -> Tuple[str, str]:
        """Extract label before placeholder"""
        placeholder = match.group()
        
        # Get text before placeholder (up to 150 chars)
        before_text = text[:match.start()]
        label = before_text[-150:].strip() if len(before_text) > 0 else ""
        
        # Clean label: keep last sentence/phrase
        if ':' in label:
            label = label.split(':')[-1].strip()
        elif ',' in label:
            label = label.split(',')[-1].strip()
        
        return label, placeholder
    
    @staticmethod
    def extract_all(doc: Document) -> List[Tuple[str, str, any]]:
        """Returns: [(label, placeholder, paragraph_obj)]"""
        results = []
        
        for para in doc.paragraphs:
            text = para.text
            for pattern in PlaceholderDetector.PATTERNS.values():
                for match in pattern.finditer(text):
                    label, placeholder = PlaceholderDetector.extract_label_and_placeholder(text, match)
                    results.append((label, placeholder, para))
        
        # Tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        text = para.text
                        for pattern in PlaceholderDetector.PATTERNS.values():
                            for match in pattern.finditer(text):
                                label, placeholder = PlaceholderDetector.extract_label_and_placeholder(text, match)
                                results.append((label, placeholder, para))
        
        return results

# ============================================================================
# LLM Mapper - Match LABELS to JSON KEYS
# ============================================================================

class OpenRouterMapper:
    def __init__(self):
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.model = MODEL_NAME
    
    def map_labels_to_keys(self, json_keys: List[str], labels: List[str]) -> Dict[str, str]:
        """Map document labels to JSON keys"""
        
        system_prompt = """You are mapping Romanian document labels to JSON field names.

TASK: Match each label (text before blank) to the correct JSON key.

RULES:
1. Match by semantic meaning
2. Handle Romanian: "numele" = name, "adresa" = address, "data" = date
3. Be specific: "Data completării" → "Data completarii" (exact match)
4. Return confidence >0.99 only
5. One label = one JSON key
6. Ensure you are not destroying the document/the format.
7. Fit the size of the text in the places where it doesn't directly fit, don't change the dots or date lines but fit the text just above or in between.

RESPONSE FORMAT (JSON only):
{
  "mappings": [
    {"label": "CIF", "json_key": "CIF", "confidence": 0.95}
  ]
}"""

        user_prompt = f"""JSON KEYS:
{json.dumps(json_keys, ensure_ascii=False)}

DOCUMENT LABELS:
{json.dumps(labels[:40], ensure_ascii=False)}

Map labels to keys."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=4096
            )
            
            content = response.choices[0].message.content.strip()
            content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE | re.DOTALL)
            
            data = json.loads(content)
            mapping_obj = MappingResponse(**data)
            
            result = {}
            for m in mapping_obj.mappings:
                if m.json_key in json_keys and m.confidence >= CONFIDENCE_THRESHOLD:
                    result[m.label] = m.json_key
                    if DEBUG_MODE:
                        print(f"[MAP] '{m.label}' → '{m.json_key}' ({m.confidence:.2f})")
            
            print(f"✓ Mapped {len(result)}/{len(labels[:40])} labels")
            return result
            
        except Exception as e:
            print(f"⚠ LLM failed: {e}")
            return {}

# ============================================================================
# Fuzzy Matcher for Fallback
# ============================================================================

class FuzzyMatcher:
    DIACRITICS = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
    }
    
    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        for old, new in FuzzyMatcher.DIACRITICS.items():
            text = text.replace(old, new.lower())
        return text
    
    @staticmethod
    def match(label: str, json_keys: List[str]) -> Optional[str]:
        if not label or len(label) < 3:
            return None
        
        norm_label = FuzzyMatcher.normalize(label)
        best_match, best_score = None, 0
        
        for key in json_keys:
            norm_key = FuzzyMatcher.normalize(key)
            score = max(
                fuzz.ratio(norm_label, norm_key),
                fuzz.partial_ratio(norm_label, norm_key),
                fuzz.token_sort_ratio(norm_label, norm_key)
            )
            
            if score > best_score and score >= 85:
                best_score, best_match = score, key
        
        return best_match

# ============================================================================
# Document Renderer
# ============================================================================

class DocxRenderer:
    @staticmethod
    def replace_in_paragraph(para, placeholder: str, value: str) -> bool:
        full_text = para.text
        if placeholder not in full_text:
            return False
        
        if '\n' in value:
            value = value.replace('\n', ' ')
        
        # Find placeholder and replace
        new_text = full_text.replace(placeholder, value, 1)
        
        # Replace in first run to preserve formatting
        if para.runs:
            para.runs[0].text = new_text
            # Clear other runs
            for run in para.runs[1:]:
                run.text = ""
            return True
        
        return False
    
    @staticmethod
    def fill_all(doc: Document, placeholder: str, value: str) -> int:
        count = 0
        
        for para in doc.paragraphs:
            if DocxRenderer.replace_in_paragraph(para, placeholder, value):
                count += 1
        
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        if DocxRenderer.replace_in_paragraph(para, placeholder, value):
                            count += 1
        
        return count

# ============================================================================
# Main Agent
# ============================================================================

class DocxFormFillerAgent:
    def __init__(self, use_llm: bool = True):
        self.normalizer = InputNormalizer()
        self.detector = PlaceholderDetector()
        self.llm_mapper = OpenRouterMapper() if use_llm else None
        self.fuzzy_matcher = FuzzyMatcher()
        self.renderer = DocxRenderer()
    
    def process(self, docx_path: str, data_path: str, output_path: str, verbose: bool = True):
        stats = {
            "total_placeholders": 0,
            "filled": 0,
            "llm_mapped": 0,
            "fuzzy_mapped": 0,
            "unmapped": 0
        }
        
        if verbose:
            print("=" * 70)
            print("DOCX FORM FILLER - Label Matching")
            print("=" * 70)
        
        # 1. Load JSON data
        data = self.normalizer.load_json(data_path)
        if verbose:
            print(f"[1/5] ✓ Loaded {len(data)} JSON fields")
        
        # 2. Load document
        doc = Document(docx_path)
        if verbose:
            print(f"[2/5] ✓ Loaded document")
        
        # 3. Extract labels + placeholders
        extractions = self.detector.extract_all(doc)
        stats["total_placeholders"] = len(extractions)
        
        # Group by label
        label_groups = {}
        for label, placeholder, para in extractions:
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append((placeholder, para))
        
        if verbose:
            print(f"[3/5] ✓ Found {len(extractions)} placeholders ({len(label_groups)} unique labels)")
            if DEBUG_MODE:
                print(f"[DEBUG] Sample labels: {list(label_groups.keys())[:5]}")
        
        # 4. Map labels to JSON keys
        label_to_key = {}
        
        if self.llm_mapper:
            if verbose:
                print("[4/5] Mapping with LLM...")
            label_to_key = self.llm_mapper.map_labels_to_keys(
                list(data.keys()),
                list(label_groups.keys())
            )
            stats["llm_mapped"] = len(label_to_key)
        
        # 5. Fill document
        if verbose:
            print("[5/5] Filling...")
        
        for label, placeholders in label_groups.items():
            json_key = None
            source = None
            
            # Try LLM mapping
            if label in label_to_key:
                json_key = label_to_key[label]
                source = "LLM"
            
            # Fuzzy fallback
            if not json_key:
                json_key = self.fuzzy_matcher.match(label, list(data.keys()))
                if json_key:
                    source = "FUZZY"
                    stats["fuzzy_mapped"] += 1
            
            # Fill if matched
            if json_key and json_key in data:
                value = data[json_key]
                
                for placeholder, para in placeholders:
                    if self.renderer.replace_in_paragraph(para, placeholder, value):
                        stats["filled"] += 1
                
                if verbose:
                    print(f"  ✓ [{source}] '{label[:30]}...' → '{json_key[:30]}' = '{value[:30]}'")
            else:
                stats["unmapped"] += 1
                if DEBUG_MODE and label.strip():
                    print(f"  ✗ UNMAPPED: '{label[:50]}'")
        
        # Save
        doc.save(output_path)
        if verbose:
            print(f"✓ Saved: {output_path}")
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            for k, v in stats.items():
                print(f"{k:.<30} {v}")
            fill_rate = (stats["filled"] / stats["total_placeholders"] * 100) if stats["total_placeholders"] > 0 else 0
            print(f"{'Fill rate':.<30} {fill_rate:.1f}%")
            print("=" * 70)
        
        return stats