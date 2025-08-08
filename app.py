import streamlit as st
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import csv
import ast
import shutil
import os
import zipfile
import random
import tempfile
import base64

# === Helper Functions ===
def generate_and_randomize_sequence(length):
    """Generates a sequence of integers from 1 to length and randomizes it."""
    sequence = list(range(1, length + 1))
    random.shuffle(sequence)
    return sequence

def clear_output_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def zip_files_and_get_download_link(files_to_zip, zip_name):
    """Create a zip file and return a download link"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
            for file_path in files_to_zip:
                zipf.write(file_path, os.path.basename(file_path))
        
        # Read the zip file as bytes
        with open(tmp_zip.name, 'rb') as f:
            zip_bytes = f.read()
        
        # Create a download link
        b64 = base64.b64encode(zip_bytes).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{zip_name}">Download {zip_name}</a>'
        return href

def make_jsons_zip(output_dir, json_zip_name="jsons.zip"):
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    zip_path = os.path.join(output_dir, json_zip_name)
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in json_files:
            zipf.write(os.path.join(output_dir, f), arcname=f)
    return zip_path

def extract_dummy_ids(doc_content: str):
    """Extract all unique dummy IDs from the question blocks."""
    dummy_ids = []
    for block in doc_content.strip().split('---'):
        match = re.search(r'id:\s*(new\w+)', block)
        if match:
            dummy_ids.append(match.group(1))
    return dummy_ids

def write_id_mapping_csv(id_mapping, output_dir="enhanced_questions", filename="id_mapping.csv"):
    output_path = Path(output_dir) / filename
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["dummy_id", "official_id"])
        for dummy_id, official_id in id_mapping.items():
            writer.writerow([dummy_id, official_id])
    print(f"ID mapping CSV written to {output_path}")

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Enums and Dataclass ===
class QuestionType(Enum):
    MCQ = "mcq"
    MRQ = "mrq"
    STRING = "string"
    OQ = "oq"
    GAP_TEXT = "gapText"
    MATCHING = "matching"
    INPUT_BOX = "input_box"
    FRQ = "frq"
    FRQ_AI = "frq_ai"

class Language(Enum):
    ARABIC = "ar"
    ENGLISH = "en"

@dataclass
class ProcessingStats:
    total_questions: int = 0
    successful: int = 0
    failed: int = 0
    generated_files: List[str] = None
    def __post_init__(self):
        if self.generated_files is None:
            self.generated_files = []

# === Question Processor ===
class QuestionProcessor:
    # Expanded Arabic LaTeX mapping with better organization
    ARABIC_LATEX_MAP = {
        # Functions and operators
        "F(x)": "\\dotlessqaft (\\seen)",
        "Q'": "\\dotlessnoont \\prime",
        # Single characters - uppercase
        'N': '\\tah', 'Z': '\\sadt', 'Q': '\\dotlessnoont', 'X': '\\seent',
        'Y': '\\sadt', 'A': '\\alt{\\alef}', 'B': '\\beh', 'C': '\\jeemi',
        'D': '\\dal', 'E': '\\hehi', 'F': '\\waw', 'M': '\\meem',
        'K': '\\kaf', 'L': '\\lam', 'O': '\\waw', 'R': '\\haht',
        # Single characters - lowercase
        'x': '\\seen', 'y': '\\sad', 'z': '\\ain', 'n': '\\noon',
        's': '\\feh', 'r': '\\aint',
    }

    def __init__(self, output_dir: str = "generated_questions", id_mapping: dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stats = ProcessingStats()
        self.id_mapping = id_mapping or {}

    def get_current_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_math_field(self, equation: str, language: Language, force_en: bool = False) -> str:
        if not equation:
            return ""
        try:
            is_arabic = language == Language.ARABIC and not force_en
            latex_value = self.convert_to_arabic_latex(equation) if is_arabic else equation
            locale_attrs = ' locale="ar" lang="ar"' if is_arabic else ''
            # Preserve spaces in the equation, especially between numbers and units
            # Don't strip spaces from the latex_value
            processed_value = (latex_value
                            .replace('&', '&amp;')
                            .replace('<', '&lt;')
                            .replace('>', '&gt;')
                            .replace('"', '&quot;')
                            .replace("'", '&#39;'))
            return (f'<span class="LexicalTheme__math--inline" data-node-type="math" '
                  f'data-node-variation="inline">'
                  f'<math-field default-mode="inline-math" read-only="true" '
                  f'value="{processed_value}"{locale_attrs}></math-field></span>')
        except Exception as e:
            logger.error(f"Error creating math field for equation '{equation}': {e}")
            return f'<span class="error">Math field error: {equation}</span>'

    def convert_to_arabic_latex(self, equation: str) -> str:
        if not equation or not isinstance(equation, str):
            logger.warning(f"Invalid equation input: {equation}")
            return str(equation) if equation else ""
        try:
            # Protect LaTeX commands first
            latex_commands = re.findall(r'(\\[a-zA-Z]+(?:\{[^}]*\})?)', equation)
            protected_equation = equation
            for i, command in enumerate(latex_commands):
                placeholder = f'__LATEX_CMD_{i}__'
                protected_equation = protected_equation.replace(command, placeholder, 1)
            
            # Get sorted keys for replacement
            sorted_keys = sorted(self.ARABIC_LATEX_MAP.keys(), key=len, reverse=True)
            
            # Split by spaces to preserve spacing
            parts = re.split(r'(\s+)', protected_equation)  # This preserves the spaces
            processed_parts = []
            for part in parts:
                if re.match(r'\s+', part):  # If it's whitespace, keep it as-is
                    processed_parts.append(part)
                elif part in sorted_keys:
                    processed_parts.append(self.ARABIC_LATEX_MAP[part])
                else:
                    # Apply replacements within the part
                    processed_part = part
                    for key in sorted_keys:
                        if key in processed_part:
                            processed_part = processed_part.replace(key, self.ARABIC_LATEX_MAP[key])
                    processed_parts.append(processed_part)
            
            # Join without adding extra spaces
            result = ''.join(processed_parts)
            
            # Restore LaTeX commands
            for i, command in enumerate(latex_commands):
                placeholder = f'__LATEX_CMD_{i}__'
                result = result.replace(placeholder, command, 1)
            
            return result
        except Exception as e:
            logger.error(f"Error in Arabic LaTeX conversion: {e}")
            return equation

    def process_text_for_html(self, text: str, language: Language, text_align: str = None, text_indent: str = None) -> str:
        if not text:
            return ""
        try:
            direction = ' dir="rtl"' if language == Language.ARABIC else ''
            blank_html = '<span data-node-type="blank-line" data-node-variation="space">&nbsp;</span>'
            processed_text = text.replace("_____", blank_html)
            
            # Build style attributes
            style_parts = []
            if text_align:
                style_parts.append(f"text-align: {text_align}")
            if text_indent:
                style_parts.append(f"text-indent: {text_indent}")
            style_attr = f' style="{"; ".join(style_parts)}"' if style_parts else ''
            
            # Updated regex to be more precise about math delimiters
            math_pattern = r'(``[^`]*``|`[^`]*`)'
            parts = re.split(math_pattern, processed_text)
            final_html_content = ""
            for i, part in enumerate(parts):
                if not part:
                    continue
                if part.startswith('``') and part.endswith('``'):
                    math_content = part[2:-2]  # Don't strip spaces here
                    if math_content:
                        final_html_content += self.create_math_field(math_content, language, force_en=True)
                elif part.startswith('`') and part.endswith('`'):
                    math_content = part[1:-1]  # Don't strip spaces here
                    if math_content:
                        final_html_content += self.create_math_field(math_content, language)
                else:
                    if '<span data-node-type="blank-line"' in part:
                        final_html_content += part
                    else:
                        # Handle spacing around text more carefully
                        escaped_text = (part.replace('&', '&amp;')
                                            .replace('<', '&lt;')
                                            .replace('>', '&gt;'))
                        # If this is just whitespace and we're between math expressions, preserve it
                        if escaped_text.strip() == "" and i > 0 and i < len(parts) - 1:
                            # Check if adjacent parts are math expressions
                            prev_is_math = i > 0 and (parts[i-1].startswith('`'))
                            next_is_math = i < len(parts) - 1 and (parts[i+1].startswith('`'))
                            if prev_is_math or next_is_math:
                                final_html_content += f'<span style="white-space: pre-wrap;">{escaped_text}</span>'
                            else:
                                final_html_content += f'<span style="white-space: pre-wrap;">{escaped_text}</span>'
                        else:
                            final_html_content += f'<span style="white-space: pre-wrap;">{escaped_text}</span>'
            
            return f'<p class="LexicalTheme__paragraph"{direction}{style_attr}>{final_html_content}</p>'
        except Exception as e:
            logger.error(f"Error processing text for HTML: {e}")
            return f'<p class="error">Text processing error: {text[:50]}...</p>'

    def validate_metadata(self, metadata: Dict[str, str]) -> Tuple[bool, List[str]]:
        required_fields = ['id', 'language']
        errors = []
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate language
        if 'language' in metadata:
            try:
                Language(metadata['language'])
            except ValueError:
                errors.append(f"Invalid language: {metadata['language']}")
        
        # Validate ID is numeric
        if 'id' in metadata:
            try:
                int(metadata['id'])
            except ValueError:
                errors.append(f"ID must be numeric: {metadata['id']}")
        
        return len(errors) == 0, errors

    def create_base_json_structure(self, metadata: Dict[str, str]) -> Dict[str, Any]:
        language = Language(metadata['language'])
        question_id = int(metadata['id'])
        mapped_id = question_id
        if language == Language.ARABIC and 'mapped_id' in metadata:
            mapped_id = question_id
            question_id = int(metadata['mapped_id'])
        
        dialect_map = {
            Language.ARABIC: ["modern_standard"],
            Language.ENGLISH: ["american", "british"]
        }
        
        return {
            "parts": [],
            "statement": None,
            "instance_number": 1,
            "metadata": {
                "id": question_id,
                "mapped_id": mapped_id,
                "category": metadata.get('category', 'lesson'),
                "language": language.value,
                "country": metadata.get('country', 'eg'),
                "dialect": dialect_map[language],
                "source_id": {
                    "value": 209168454958,
                    "page_number": None
                },
                "description": metadata.get('description', ''),
                "example_id": None,
                "has_example": False,
                "instances_count": 1,
                "publication_date": self.get_current_timestamp(),
                "parts_count": 0
            }
        }

    def process_mcq_mrq_part(self, content: Dict[str, str], language: Language) -> Dict[str, Any]:
        choices = []
        if 'choices' not in content:
            raise ValueError("MCQ/MRQ questions must have choices")
        
        choices_lines = [line.strip() for line in content['choices'].strip().split('\n') if line.strip()]
        rand_list = generate_and_randomize_sequence(len(choices_lines))
        
        for j, choice_line in enumerate(choices_lines):
            is_key = choice_line.startswith('*')
            is_distractor = choice_line.startswith('-')
            if is_key or is_distractor:
                text = choice_line[1:].strip()
            else:
                text = choice_line.strip()
            
            choices.append({
                "type": "key" if is_key else "distractor",
                "html_content": self.process_text_for_html(text, language),
                "values": [],
                "unit": None,
                "index": j,
                "fixed_order": rand_list[j],
                "last_order": False
            })
        
        return {"choices": choices}

    def process_string_part(self, content: Dict[str, str], metadata: Dict[str, str],
                          part_metadata: Dict[str, str], language: Language) -> Dict[str, Any]:
        if 'answer' not in content:
            raise ValueError("String questions must have an answer")
        
        result = {
            "choices": None,
            "answer": [self.process_text_for_html(content['answer'].strip(), language)]
        }
        
        ai_id = part_metadata.get('ai_template_id') or metadata.get('ai_template_id')
        if ai_id:
            result["ai"] = {"ai_template_id": ai_id}
        
        return result

    def process_oq_part(self, content: Dict[str, str], language: Language, direction: str = "vertical") -> Dict[str, Any]:
        if 'choices' not in content:
            raise ValueError("OQ questions must have choices")
        
        choices = []
        oq_lines = [line.strip() for line in content['choices'].strip().split('\n') if line.strip()]
        rand_list = generate_and_randomize_sequence(len(oq_lines))
        
        for j, line in enumerate(oq_lines):
            choices.append({
                "type": "distractor",
                "html_content": self.process_text_for_html(line, language),
                "correct_order": j + 1,
                "values": [],
                "unit": None,
                "index": j,
                "fixed_order": rand_list[j],
                "last_order": False
            })
        
        return {
            "direction": direction,
            "choices": choices
        }

    def process_gap_text_part(self, content: Dict[str, str], language: Language) -> Dict[str, Any]:
        if 'gaps' not in content or 'stem' not in content:
            raise ValueError("Gap text questions must have gaps and stem")
        
        gap_text_keys = []
        gap_lines = [
            line.strip() for line in content['gaps'].strip().split('\n')
            if line.strip() and line.strip() != "---"
        ]
        
        for j, line in enumerate(gap_lines):
            gap_text_keys.append({
                "value": line,
                "correct_order": j + 1
            })
        
        stem_html_gap = '<span data-node-type="blank-line" data-node-variation="gap">&nbsp;</span>'
        processed_stem = content['stem'].strip().replace("[BLANK]", stem_html_gap)
        
        return {
            "choices": [],
            "gap_text_keys": gap_text_keys,
            "stem": self.process_text_for_html(processed_stem, language)
        }

    def process_matching_part(self, content: Dict[str, str], language: Language) -> Dict[str, Any]:
        if 'matching_pairs' not in content:
            raise ValueError("Matching questions must have matching_pairs")
        
        choices = []
        pair_lines = [line.strip() for line in content['matching_pairs'].strip().split('\n') if line.strip()]
        
        for j, line in enumerate(pair_lines):
            if '|' not in line:
                raise ValueError(f"Matching pair must contain '|' separator: {line}")
            
            group1_text, group2_text = line.split('|', 1)
            
            choices.append({
                "type": "distractor",
                "html_content": self.process_text_for_html(group1_text.strip(), language),
                "group": 1,
                "correct_order": j + 1,
                "values": [],
                "unit": None,
                "index": j,
                "fixed_order": j + 1,
                "last_order": False
            })
            
            choices.append({
                "type": "distractor",
                "html_content": self.process_text_for_html(group2_text.strip(), language),
                "group": 2,
                "correct_order": j + 1,
                "values": [],
                "unit": None,
                "index": j + len(pair_lines),
                "fixed_order": j + 1 + len(pair_lines),
                "last_order": False
            })
        
        return {"choices": choices}

    def process_input_box_part(self, content: Dict[str, str]) -> Dict[str, Any]:
        if 'answer' not in content:
            raise ValueError("Input box questions must have an answer")
        
        answer_parts = content['answer'].strip().split('|')
        value = answer_parts[0].strip()
        unit = answer_parts[1].strip() if len(answer_parts) > 1 else None
        
        return {
            "choices": [],
            "answer": {
                "value": value,
                "unit": unit,
                "constrains": {"type": "integer"}
            }
        }

    def process_frq_part(self, content: Dict[str, str], metadata: Dict[str, str],
                        part_metadata: Dict[str, str], language: Language,
                        question_type: QuestionType) -> Dict[str, Any]:
        if 'answer' not in content:
            raise ValueError("FRQ questions must have an answer")
        
        result = {
            "choices": [],
            "answer": self.process_text_for_html(content['answer'].strip(), language)
        }
        
        if question_type == QuestionType.FRQ_AI:
            ai_id = part_metadata.get('ai_template_id') or metadata.get('ai_template_id')
            if not ai_id:
                raise ValueError("'ai_template_id' is required for frq_ai type")
            result["ai"] = {"ai_template_id": ai_id}
        
        return result

    def process_question_part(self, part_block: str, part_counter: int,
                            metadata: Dict[str, str], language: Language) -> Optional[Dict[str, Any]]:
        try:
            part_metadata, content, current_tag = {}, {}, None
            
            for line in part_block.strip().split('\n'):
                tag_match = re.match(r'^\[([A-Z_]+)\]$', line)
                meta_match = re.match(r'^([a-z_]+):\s*(.*)', line)
                
                if tag_match:
                    current_tag = tag_match.group(1).lower()
                    content[current_tag] = ""
                elif meta_match and current_tag is None:
                    part_metadata[meta_match.group(1)] = meta_match.group(2)
                elif current_tag and line.strip():
                    content[current_tag] += line + "\n"
            
            for key in content:
                content[key] = content[key].strip()
            
            part_type_str = part_metadata.get('type') or metadata.get('type')
            if not part_type_str:
                raise ValueError("Question type not specified")
            
            try:
                question_type = QuestionType(part_type_str)
            except ValueError:
                raise ValueError(f"Unsupported question type: {part_type_str}")
            
            part = {
                "n": part_counter,
                "type": question_type.value,
                "subtype": None,
                "standalone": False
            }
            
            if 'stem' in content:
                part["stem"] = self.process_text_for_html(content['stem'], language)
            
            if question_type in [QuestionType.MCQ, QuestionType.MRQ]:
                part.update(self.process_mcq_mrq_part(content, language))
            elif question_type == QuestionType.STRING:
                part.update(self.process_string_part(content, metadata, part_metadata, language))
            elif question_type == QuestionType.OQ:
                # direction is an optional metadata key, default to horizontal if not present
                oq_direction = part_metadata.get('direction') or metadata.get('direction') or "horizontal"
                part.update(self.process_oq_part(content, language, direction=oq_direction))
            elif question_type == QuestionType.GAP_TEXT:
                gap_result = self.process_gap_text_part(content, language)
                part["stem"] = gap_result["stem"]
                part["choices"] = gap_result["choices"]
                part["gap_text_keys"] = gap_result["gap_text_keys"]
            elif question_type == QuestionType.MATCHING:
                part.update(self.process_matching_part(content, language))
            elif question_type == QuestionType.INPUT_BOX:
                part.update(self.process_input_box_part(content))
            elif question_type in [QuestionType.FRQ, QuestionType.FRQ_AI]:
                part.update(self.process_frq_part(content, metadata, part_metadata, language, question_type))
            
            return part
        except Exception as e:
            logger.error(f"Error processing part {part_counter}: {e}")
            raise

    def process_single_question(self, block: str, question_number: int) -> Optional[str]:
        try:
            logger.info(f"Processing Question #{question_number}...")
            
            lines = [line.rstrip() for line in block.strip().split('\n')]
            metadata = {}
            metadata_end_index = 0
            
            # Parse metadata
            for j, line in enumerate(lines):
                if line.startswith('['):
                    metadata_end_index = j
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            
            # Replace dummy ID with official ID if mapping is provided
            dummy_id = metadata.get('id')
            if dummy_id and dummy_id in self.id_mapping:
                metadata['id'] = self.id_mapping[dummy_id]
            
            # Validate metadata
            is_valid, errors = self.validate_metadata(metadata)
            if not is_valid:
                raise ValueError(f"Metadata validation failed: {', '.join(errors)}")
            
            language = Language(metadata['language'])
            final_json = self.create_base_json_structure(metadata)
            
            content_str = '\n'.join(lines[metadata_end_index:])
            
            if '[STATEMENT]' in content_str and '[PART]' in content_str:
                statement_content, parts_str = content_str.split('[PART]', 1)
                statement_text = statement_content.replace('[STATEMENT]', '').strip()
                if statement_text:
                    final_json["statement"] = self.process_text_for_html(statement_text, language)
                
                part_blocks_raw = ('[PART]' + parts_str).split('[PART]')
                part_blocks = [p.strip() for p in part_blocks_raw if p.strip()]
            else:
                part_blocks = [content_str] if content_str.strip() else []
            
            for part_counter, part_block in enumerate(part_blocks, 1):
                if not part_block.strip():
                    continue
                part = self.process_question_part(part_block, part_counter, metadata, language)
                if part:
                    final_json['parts'].append(part)
            
            final_json['metadata']['parts_count'] = len(final_json['parts'])
            
            if not final_json['parts']:
                raise ValueError("No valid parts found in question")
            
            file_name = f"{metadata['id']}.json"
            file_path = self.output_dir / file_name
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Successfully created {file_name}")
            return file_name
        except Exception as e:
            logger.error(f"❌ Error processing Question #{question_number}: {e}")
            return None

    def process_google_doc(self, doc_content: str) -> ProcessingStats:
        logger.info("--- Starting Bulk Processing ---")
        self.stats = ProcessingStats()
        
        question_blocks = [block.strip() for block in doc_content.strip().split('---') if block.strip()]
        self.stats.total_questions = len(question_blocks)
        
        logger.info(f"Found {self.stats.total_questions} question blocks to process")
        
        for i, block in enumerate(question_blocks, 1):
            result = self.process_single_question(block, i)
            if result:
                self.stats.successful += 1
                self.stats.generated_files.append(result)
            else:
                self.stats.failed += 1
        
        logger.info("--- Bulk Processing Complete ---")
        logger.info(f"Total Questions: {self.stats.total_questions}")
        logger.info(f"Successful: {self.stats.successful}")
        logger.info(f"Failed: {self.stats.failed}")
        logger.info(f"Success Rate: {(self.stats.successful/self.stats.total_questions*100):.1f}%")
        
        return self.stats

def create_summary_csv(json_files, output_dir, lesson_id):
    rows = []
    for json_file in json_files:
        file_path = Path(output_dir) / json_file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            qid = data["metadata"]["id"]
            qtype = data["parts"][0]["type"] if data["parts"] else ""
            lang = data["metadata"]["language"] if "language" in data["metadata"] else ""
            rows.append({
                "Question Id": qid,
                "Lang": lang,
                "Lesson Id": lesson_id,
                "Attributes": qtype,
                "Category": "lesson",
                "Dialect": ""
            })
    
    summary_csv_path = Path(output_dir) / "question_set_summary.csv"
    with open(summary_csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Question Id", "Lang", "Lesson Id", "Attributes", "Category", "Dialect"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"Question set summary CSV written to {summary_csv_path}")

# Streamlit App
def main():
    st.title("Question Processor")
    st.markdown("Convert text-based questions to JSON format")
    
    # Create session state variables
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # ID Mapping Section
    st.sidebar.subheader("ID Mapping")
    enable_id_mapping = st.sidebar.checkbox("Enable ID Mapping", help="Map dummy IDs to official IDs")
    id_mapping_text = ""
    if enable_id_mapping:
        id_mapping_text = st.sidebar.text_area(
            "Enter ID mapping as JSON (e.g., {\"new123\": \"298195892474\"})",
            height=100,
            help="Provide a JSON object mapping dummy IDs to official 12-digit IDs"
        )
        try:
            if id_mapping_text:
                id_mapping = json.loads(id_mapping_text)
                st.sidebar.success(f"Loaded {len(id_mapping)} ID mappings")
            else:
                id_mapping = {}
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON format for ID mapping")
            id_mapping = {}
    else:
        id_mapping = {}
    
    # Main content area
    st.header("Input Questions")
    
    # Text area for question input
    question_input = st.text_area(
        "Paste your questions here (use '---' to separate questions)",
        height=300,
        help="Format your questions according to the specification. Use '---' to separate individual questions."
    )
    
    # Example format
    with st.expander("See Question Format Example"):
        st.code("""
id: 12345
language: en
type: mcq

[STATEMENT]
What is the capital of France?

[PART]
type: mcq

[CHOICES]
* Paris
- London
- Berlin
- Madrid
        """, language="text")
    
    # Process button
    if st.button("Process Questions"):
        if not question_input.strip():
            st.error("Please enter some questions to process")
            return
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize the processor with ID mapping
            processor = QuestionProcessor(output_dir=temp_dir, id_mapping=id_mapping)
            
            # Process the questions
            with st.spinner("Processing questions..."):
                stats = processor.process_google_doc(question_input)
            
            # Display processing statistics
            st.header("Processing Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Questions", stats.total_questions)
            col2.metric("Successful", stats.successful)
            col3.metric("Failed", stats.failed)
            
            if stats.failed > 0:
                st.warning(f"{stats.failed} question(s) failed to process. Check the logs for details.")
            
            # Create summary CSV
            json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            if json_files:
                lesson_id = st.text_input("Enter Lesson ID for summary:", "lesson_001")
                create_summary_csv(json_files, temp_dir, lesson_id)
                
                # Store processed files and output directory in session state
                st.session_state.processed_files = [os.path.join(temp_dir, f) for f in json_files + ["question_set_summary.csv"]]
                st.session_state.output_dir = temp_dir
                
                st.success(f"Successfully processed {stats.successful} questions!")
                
                # Show a sample of the processed JSON
                st.subheader("Sample Processed Question")
                sample_file = os.path.join(temp_dir, json_files[0])
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_json = json.load(f)
                st.json(sample_json)
    
    # Download section
    if st.session_state.processed_files:
        st.header("Download Results")
        
        # Option to download individual JSON files
        st.subheader("Individual JSON Files")
        for file_path in st.session_state.processed_files:
            if file_path.endswith('.json'):
                file_name = os.path.basename(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Create download button for each JSON file
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_content,
                    file_name=file_name,
                    mime="application/json"
                )
        
        # Option to download all files as a zip
        st.subheader("Download All Files")
        lesson_id = st.text_input("Enter Lesson ID for zip file:", "lesson_001")
        if st.button("Create ZIP Archive"):
            zip_link = zip_files_and_get_download_link(st.session_state.processed_files, f"{lesson_id}.zip")
            st.markdown(zip_link, unsafe_allow_html=True)
            
            # Also create a JSONs-only zip
            json_files_only = [f for f in st.session_state.processed_files if f.endswith('.json')]
            json_zip_link = zip_files_and_get_download_link(json_files_only, "jsons.zip")
            st.markdown(json_zip_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
