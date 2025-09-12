import re
import os
from typing import List, Tuple, Optional

class RFileHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_functions_and_classes(self) -> List[Tuple[str, Optional[str], int, int, Optional[str], List[str]]]:
        """
        Get the functions and S4 classes in a given R file.
        Returns a list of tuples, each containing:
        1. the function or class name,
        2. parent name (None for R, as R doesn't have nested functions in the same way),
        3. start line number,
        4. end line number,
        5. doc string (roxygen comments),
        6. params (function parameters).
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        functions_and_classes = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments (except roxygen)
            if not line or (line.startswith('#') and not line.startswith('#\'') and not line.startswith('#@')):
                i += 1
                continue
            
            # Check for function definitions
            func_match = self._match_function(lines, i)
            if func_match:
                name, start_line, end_line, doc_string, params = func_match
                functions_and_classes.append((name, None, start_line + 1, end_line + 1, doc_string, params))
                i = end_line + 1
                continue
            
            # Check for S4 class definitions
            class_match = self._match_s4_class(lines, i)
            if class_match:
                name, start_line, end_line, doc_string = class_match
                functions_and_classes.append((name, None, start_line + 1, end_line + 1, doc_string, []))
                i = end_line + 1
                continue
            
            # Check for S3 class methods (functions with class-specific naming)
            s3_match = self._match_s3_method(lines, i)
            if s3_match:
                name, start_line, end_line, doc_string, params = s3_match
                functions_and_classes.append((name, None, start_line + 1, end_line + 1, doc_string, params))
                i = end_line + 1
                continue
            
            i += 1
        
        return functions_and_classes
    
    def _match_function(self, lines: List[str], start_idx: int) -> Optional[Tuple[str, int, int, Optional[str], List[str]]]:
        """Match function definitions in R code."""
        # Collect roxygen documentation before function
        doc_string = self._extract_roxygen_doc(lines, start_idx)
        doc_start_idx = start_idx
        
        # Skip roxygen comments to find function definition
        while start_idx < len(lines) and (lines[start_idx].strip().startswith('#\'') or 
                                         lines[start_idx].strip().startswith('#@') or 
                                         not lines[start_idx].strip()):
            start_idx += 1
        
        if start_idx >= len(lines):
            return None
        
        # Pattern for function definition: name <- function(params) or name = function(params)
        func_pattern = r'^(\s*)([a-zA-Z_][a-zA-Z0-9_.\$]*)\s*(<-|=)\s*function\s*\('
        
        line = lines[start_idx]
        match = re.match(func_pattern, line)
        
        if not match:
            return None
        
        func_name = match.group(2)
        indent_level = len(match.group(1))
        
        # Extract parameters
        params = self._extract_function_params(lines, start_idx)
        
        # Find the end of the function by tracking braces
        end_idx = self._find_function_end(lines, start_idx, indent_level)
        
        return func_name, doc_start_idx, end_idx, doc_string, params
    
    def _match_s4_class(self, lines: List[str], start_idx: int) -> Optional[Tuple[str, int, int, Optional[str]]]:
        """Match S4 class definitions."""
        doc_string = self._extract_roxygen_doc(lines, start_idx)
        doc_start_idx = start_idx
        
        # Skip documentation to find class definition
        while start_idx < len(lines) and (lines[start_idx].strip().startswith('#\'') or 
                                         lines[start_idx].strip().startswith('#@') or 
                                         not lines[start_idx].strip()):
            start_idx += 1
        
        if start_idx >= len(lines):
            return None
        
        # Pattern for S4 class: setClass("ClassName", ...)
        class_pattern = r'setClass\s*\(\s*["\']([^"\']+)["\']'
        
        line = lines[start_idx]
        match = re.search(class_pattern, line)
        
        if not match:
            return None
        
        class_name = match.group(1)
        
        # Find the end by tracking parentheses
        end_idx = self._find_parentheses_end(lines, start_idx)
        
        return class_name, doc_start_idx, end_idx, doc_string
    
    def _match_s3_method(self, lines: List[str], start_idx: int) -> Optional[Tuple[str, int, int, Optional[str], List[str]]]:
        """Match S3 method definitions (method.class pattern)."""
        doc_string = self._extract_roxygen_doc(lines, start_idx)
        doc_start_idx = start_idx
        
        # Skip documentation
        while start_idx < len(lines) and (lines[start_idx].strip().startswith('#\'') or 
                                         lines[start_idx].strip().startswith('#@') or 
                                         not lines[start_idx].strip()):
            start_idx += 1
        
        if start_idx >= len(lines):
            return None
        
        # Pattern for S3 method: method.class <- function(params)
        s3_pattern = r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*(<-|=)\s*function\s*\('
        
        line = lines[start_idx]
        match = re.match(s3_pattern, line)
        
        if not match:
            return None
        
        method_name = match.group(2)
        indent_level = len(match.group(1))
        
        # Extract parameters
        params = self._extract_function_params(lines, start_idx)
        
        # Find the end of the function
        end_idx = self._find_function_end(lines, start_idx, indent_level)
        
        return method_name, doc_start_idx, end_idx, doc_string, params
    
    def _extract_roxygen_doc(self, lines: List[str], start_idx: int) -> Optional[str]:
        """Extract roxygen2 documentation comments."""
        doc_lines = []
        i = start_idx
        
        # Go backwards to find the start of roxygen comments
        while i > 0 and (lines[i-1].strip().startswith('#\'') or lines[i-1].strip().startswith('#@') or not lines[i-1].strip()):
            if lines[i-1].strip().startswith('#\'') or lines[i-1].strip().startswith('#@'):
                i -= 1
            elif not lines[i-1].strip():
                i -= 1
            else:
                break
        
        # Collect roxygen comments
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#\'') or line.startswith('#@'):
                # Remove the roxygen prefix
                clean_line = re.sub(r'^#[\'@]\s?', '', line)
                doc_lines.append(clean_line)
                i += 1
            elif not line:  # Empty line
                i += 1
            else:
                break
        
        return '\n'.join(doc_lines) if doc_lines else None
    
    def _extract_function_params(self, lines: List[str], start_idx: int) -> List[str]:
        """Extract function parameters from function definition."""
        params = []
        
        # Find the function line and extract parameters
        func_line_complete = ""
        i = start_idx
        paren_count = 0
        found_opening = False
        
        while i < len(lines):
            line = lines[i]
            func_line_complete += line
            
            # Count parentheses to find the complete parameter list
            for char in line:
                if char == '(':
                    paren_count += 1
                    found_opening = True
                elif char == ')':
                    paren_count -= 1
            
            if found_opening and paren_count == 0:
                break
            i += 1
        
        # Extract parameters using regex
        param_match = re.search(r'function\s*\((.*?)\)', func_line_complete, re.DOTALL)
        if param_match:
            param_str = param_match.group(1).strip()
            if param_str:
                # Split by comma, but be careful with nested parentheses and quotes
                params = self._smart_split_params(param_str)
                # Clean up parameter names (remove default values, whitespace)
                params = [re.split(r'\s*=\s*', param.strip())[0].strip() for param in params]
                params = [param for param in params if param and param != '...']
        
        return params
    
    def _smart_split_params(self, param_str: str) -> List[str]:
        """Split parameters by comma, handling nested structures."""
        params = []
        current_param = ""
        paren_count = 0
        quote_char = None
        
        for char in param_str:
            if quote_char:
                current_param += char
                if char == quote_char and (len(current_param) == 1 or current_param[-2] != '\\'):
                    quote_char = None
            elif char in ['"', "'"]:
                quote_char = char
                current_param += char
            elif char == '(':
                paren_count += 1
                current_param += char
            elif char == ')':
                paren_count -= 1
                current_param += char
            elif char == ',' and paren_count == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
        
        if current_param.strip():
            params.append(current_param.strip())
        
        return params
    
    def _find_function_end(self, lines: List[str], start_idx: int, indent_level: int) -> int:
        """Find the end of a function by tracking braces and indentation."""
        brace_count = 0
        in_function = False
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            
            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_function = True
                elif char == '}':
                    brace_count -= 1
            
            # If we've closed all braces, we're at the end
            if in_function and brace_count == 0:
                return i
            
            # If no braces are used, look for next function or end of file
            if not in_function and i > start_idx:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    # Check if this looks like a new function or assignment at same/higher level
                    if re.match(r'^(\s*)[a-zA-Z_][a-zA-Z0-9_.\$]*\s*(<-|=)', line):
                        current_indent = len(re.match(r'^(\s*)', line).group(1))
                        if current_indent <= indent_level:
                            return i - 1
            
            i += 1
        
        return len(lines) - 1
    
    def _find_parentheses_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a parenthetical expression."""
        paren_count = 0
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            for char in line:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        return i
            i += 1
        
        return len(lines) - 1
    
    def get_imports(self) -> List[str]:
        """
        Get library imports and source statements in R code.
        Returns a list of library names and sourced files.
        """
        imports = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Match library() calls
            lib_match = re.search(r'library\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)', line)
            if lib_match:
                imports.append(f"library({lib_match.group(1)})")
            
            # Match require() calls
            req_match = re.search(r'require\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)', line)
            if req_match:
                imports.append(f"require({req_match.group(1)})")
            
            # Match source() calls
            src_match = re.search(r'source\s*\(\s*["\']([^"\']+)["\']\s*\)', line)
            if src_match:
                imports.append(f"source({src_match.group(1)})")
            
            # Match :: namespace calls (just collect unique packages)
            ns_matches = re.findall(r'([a-zA-Z_][a-zA-Z0-9_.]*)::', line)
            for ns in ns_matches:
                ns_import = f"{ns}::"
                if ns_import not in imports:
                    imports.append(ns_import)
        
        return imports


# Example usage:
if __name__ == "__main__":
    # Example R file analysis
    handler = RFileHandler("example.R")
    
    # Get functions and classes
    functions_and_classes = handler.get_functions_and_classes()
    print("Functions and Classes:")
    for item in functions_and_classes:
        name, parent, start, end, doc, params = item
        print(f"  {name}: lines {start}-{end}, params: {params}")
        if doc:
            print(f"    Doc: {doc[:50]}...")
    
    # Get imports
    imports = handler.get_imports()
    print(f"\nImports: {imports}")