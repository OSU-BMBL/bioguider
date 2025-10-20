import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class RSymbol:
    name: str
    parent: Optional[str]
    start_line: int
    end_line: int
    docstring: Optional[str]
    params: List[str]


class RFileHandler:
    # only up to "function("
    FUNC_DEF_HEAD_RE = re.compile(
        r'(?P<name>[A-Za-z.][\w.]*)\s*<-\s*function\s*\(',
        re.MULTILINE,
    )
    
    S3_METHOD_HEAD_RE = re.compile(
        r'(?P<generic>[A-Za-z.][\w.]*)\.(?P<class>[A-Za-z.][\w.]*)\s*<-\s*function\s*\(',
        re.MULTILINE,
    )
    
    # R6 method head: "name = function("
    R6_METHOD_HEAD_RE = re.compile(
        r'(?P<mname>[A-Za-z.][\w.]*)\s*=\s*function\s*\(',
        re.MULTILINE,
    )
    
    # S4 method head inside setMethod(... function(
    S4_METHOD_HEAD_RE = re.compile(
        r'setMethod\s*\(\s*["\'](?P<generic>[^"\']+)["\']\s*,.*?function\s*\(',
        re.MULTILINE | re.DOTALL,
    )

    FUNC_DEF_RE = re.compile(
        # name <- function( ... ) {   with multi-line args allowed
        r'(?P<name>[A-Za-z.][\w.]*)\s*<-\s*function\s*\((?P<args>[^)]*)\)\s*\{',
        re.MULTILINE,
    )
    S3_METHOD_RE = re.compile(
        r'(?P<generic>[A-Za-z.][\w.]*)\.(?P<class>[A-Za-z.][\w.]*)\s*<-\s*function\s*\((?P<args>[^)]*)\)\s*\{',
        re.MULTILINE,
    )
    R6_CLASS_RE = re.compile(
        r'(?P<varname>[A-Za-z.][\w.]*)\s*<-\s*R6Class\s*\(\s*["\'](?P<classname>[^"\']+)["\']',
        re.MULTILINE | re.DOTALL,
    )
    R6_METHOD_RE = re.compile(
        r'(?P<mname>[A-Za-z.][\w.]*)\s*=\s*function\s*\((?P<args>[^)]*)\)\s*\{',
        re.MULTILINE,
    )
    S4_CLASS_RE = re.compile(
        r'setClass\s*\(\s*["\'](?P<classname>[^"\']+)["\']',
        re.MULTILINE,
    )
    S4_METHOD_RE = re.compile(
        r'setMethod\s*\(\s*["\'](?P<generic>[^"\']+)["\']\s*,.*?function\s*\((?P<args>[^)]*)\)\s*\{',
        re.MULTILINE | re.DOTALL,
    )
    S4_SIG_CLASS_RE = re.compile(
        r'signature\s*=\s*(?:list\s*\(|\()\s*(?:[^)]*class\s*=\s*["\'](?P<classname>[^"\']+)["\']|["\'](?P<classname2>[^"\']+)["\'])',
        re.MULTILINE,
    )
    LIB_REQUIRE_RE = re.compile(
        r'\b(?:library|require)\s*\(\s*([A-Za-z.][\w.]*)\s*\)',
        re.MULTILINE,
    )
    NS_USE_RE = re.compile(
        r'(?P<pkg>[A-Za-z.][\w.]*):::{0,2}(?P<sym>[A-Za-z.][\w.]*)',
        re.MULTILINE,
    )

    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()
        self.lines = self.text.splitlines()
        self._brace_map = self._build_brace_map_safely()  # FIX: ignore comments/strings

    # ---------------- Public API ----------------

    def get_functions_and_classes(self) -> List[Tuple[str, Optional[str], int, int, Optional[str], List[str]]]:
        items: List[RSymbol] = []
        items.extend(self._parse_functions())
        items.extend(self._parse_s3_methods())
        items.extend(self._parse_r6())
        items.extend(self._parse_s4())
        items.sort(key=lambda s: (s.start_line, s.end_line))
        return [(i.name, i.parent, i.start_line, i.end_line, i.docstring, i.params) for i in items]

    def get_imports(self) -> List[str]:
        pkgs = set(self.LIB_REQUIRE_RE.findall(self.text))
        for m in self.NS_USE_RE.finditer(self.text):
            pkgs.add(m.group('pkg'))
        return sorted(pkgs)

    # ---------------- Parsers ----------------

    def _parse_functions(self) -> List[RSymbol]:
        syms: List[RSymbol] = []
        for m in self.FUNC_DEF_HEAD_RE.finditer(self.text):
            name = m.group('name')
            open_paren = m.end() - 1  # points at '('
            close_paren = self._matching_paren_pos_global(open_paren)
            if close_paren is None:
                continue
            args_text = self.text[open_paren + 1: close_paren]
            args = self._parse_params(args_text)
    
            block_open = self._find_next_code_brace_after(close_paren + 1)
            if block_open is None:
                continue
            block_close = self._matching_brace_pos(block_open)
    
            start_line = self._pos_to_line(block_open)
            end_line = self._pos_to_line(block_close)
            doc = self._roxygen_before(m.start())
    
            syms.append(RSymbol(name=name, parent=None,
                                start_line=start_line, end_line=end_line,
                                docstring=doc, params=args))
    
            # nested
            syms.extend(self._parse_nested_functions(block_open, block_close, parent=name))
        return syms

    def _parse_nested_functions(self, abs_start: int, abs_end: int, parent: str) -> List[RSymbol]:
        sub = self.text[abs_start:abs_end+1]
        syms: List[RSymbol] = []
        for m in self.FUNC_DEF_HEAD_RE.finditer(sub):
            open_rel = m.end() - 1
            close_rel = self._matching_paren_pos_in_text(sub, open_rel)
            if close_rel is None:
                continue
            args_text = sub[open_rel + 1: close_rel]
            args = self._parse_params(args_text)
    
            # brace after ')' within the slice
            func_open_rel = self._find_next_char_in_text(sub, '{', close_rel + 1)
            if func_open_rel is None:
                continue
            func_close_rel = self._matching_brace_pos_in_text(sub, func_open_rel)
            if func_close_rel is None:
                continue
    
            block_open = abs_start + func_open_rel
            block_close = abs_start + func_close_rel
            name = m.group('name')
            doc = self._roxygen_before(block_open)
            syms.append(RSymbol(
                name=name, parent=parent,
                start_line=self._pos_to_line(block_open),
                end_line=self._pos_to_line(block_close),
                docstring=doc, params=args
            ))
        return syms


    def _parse_s3_methods(self) -> List[RSymbol]:
        syms: List[RSymbol] = []
        for m in self.S3_METHOD_HEAD_RE.finditer(self.text):
            generic = m.group('generic')
            clazz = m.group('class')
            name = f"{generic}.{clazz}"
    
            open_paren = m.end() - 1
            close_paren = self._matching_paren_pos_global(open_paren)
            if close_paren is None:
                continue
            args_text = self.text[open_paren + 1: close_paren]
            args = self._parse_params(args_text)
    
            block_open = self._find_next_code_brace_after(close_paren + 1)
            if block_open is None:
                continue
            block_close = self._matching_brace_pos(block_open)
    
            syms.append(RSymbol(
                name=name, parent=generic,
                start_line=self._pos_to_line(block_open),
                end_line=self._pos_to_line(block_close),
                docstring=self._roxygen_before(m.start()),
                params=args
            ))
        return syms


    def _parse_r6(self) -> List[RSymbol]:
        syms: List[RSymbol] = []
        for m in self.R6_CLASS_RE.finditer(self.text):
            classname = m.group('classname')
            # Find the first '{' after R6Class( — it's the class call's body brace
            first_brace = self._find_next_code_brace_after(m.end())
            if first_brace is None:
                continue
            class_end = self._matching_brace_pos(first_brace)
            syms.append(RSymbol(
                name=classname, parent=None,
                start_line=self._pos_to_line(first_brace),
                end_line=self._pos_to_line(class_end),
                docstring=self._roxygen_before(m.start()),
                params=[]
            ))
            # Methods within public/private/active lists
            class_text = self.text[m.start():class_end+1]
            base = m.start()
            for sect in ('public', 'private', 'active'):
                for meth in self._parse_r6_section_methods(class_text, base, sect, classname):
                    syms.append(meth)
        return syms

    def _parse_r6_section_methods(self, class_text: str, base: int, section: str, parent_class: str) -> List[RSymbol]:
        syms: List[RSymbol] = []
        for sec in re.finditer(rf'{section}\s*=\s*list\s*\(', class_text):
            lst_open = sec.end() - 1
            lst_close = self._matching_paren_pos_in_text(class_text, lst_open)
            if lst_close is None:
                continue
            list_text = class_text[lst_open:lst_close+1]
            for m in self.R6_METHOD_HEAD_RE.finditer(list_text):
                open_rel = m.end() - 1
                close_rel = self._matching_paren_pos_in_text(list_text, open_rel)
                if close_rel is None:
                    continue
                args_text = list_text[open_rel + 1: close_rel]
                args = self._parse_params(args_text)
    
                func_open_rel = self._find_next_char_in_text(list_text, '{', close_rel + 1)
                if func_open_rel is None:
                    continue
                func_close_rel = self._matching_brace_pos_in_text(list_text, func_open_rel)
                if func_close_rel is None:
                    continue
    
                block_open = base + lst_open + func_open_rel
                block_close = base + lst_open + func_close_rel
    
                syms.append(RSymbol(
                    name=f"{parent_class}${m.group('mname')}",
                    parent=parent_class,
                    start_line=self._pos_to_line(block_open),
                    end_line=self._pos_to_line(block_close),
                    docstring=self._roxygen_before(block_open),
                    params=args
                ))
        return syms


    def _parse_s4(self) -> List[RSymbol]:
        syms: List[RSymbol] = []
        for m in self.S4_CLASS_RE.finditer(self.text):
            syms.append(RSymbol(
                name=m.group('classname'), parent=None,
                start_line=self._pos_to_line(m.start()),
                end_line=self._pos_to_line(m.start()),
                docstring=self._roxygen_before(m.start()),
                params=[]
            ))
        for m in self.S4_METHOD_HEAD_RE.finditer(self.text):
            generic = m.group('generic')
        
            open_paren = m.end() - 1
            close_paren = self._matching_paren_pos_global(open_paren)
            if close_paren is None:
                continue
            args_text = self.text[open_paren + 1: close_paren]
            args = self._parse_params(args_text)
        
            block_open = self._find_next_code_brace_after(close_paren + 1)
            block_close = self._matching_brace_pos(block_open) if block_open is not None else m.end()
        
            sig_slice = self.text[m.start(): block_open or m.end()]
            cm = self.S4_SIG_CLASS_RE.search(sig_slice)
            clazz = cm.group('classname') if cm and cm.group('classname') else (cm.group('classname2') if cm else None)
            name = f"{generic}{'<' + clazz + '>' if clazz else ''}"
        
            syms.append(RSymbol(
                name=name, parent=generic,
                start_line=self._pos_to_line(block_open if block_open is not None else m.start()),
                end_line=self._pos_to_line(block_close),
                docstring=self._roxygen_before(m.start()),
                params=args
            ))

        return syms

    # ---------------- Utilities ----------------

    def _parse_params(self, arg_str: str) -> List[str]:
        params = []
        depth = 0
        token = []
        in_s: Optional[str] = None
        escape = False
        for ch in arg_str:
            if in_s:
                token.append(ch)
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == in_s:
                    in_s = None
                continue
            if ch in ('"', "'"):
                in_s = ch
                token.append(ch)
                continue
            if ch in '([{':
                depth += 1
                token.append(ch)
            elif ch in ')]}':
                depth -= 1
                token.append(ch)
            elif ch == ',' and depth == 0:
                params.append(''.join(token).strip())
                token = []
            else:
                token.append(ch)
        if token:
            params.append(''.join(token).strip())

        cleaned = []
        for p in params:
            p = p.strip()
            if not p:
                continue
            if p == '...':
                cleaned.append('...')
                continue
            name = p.split('=')[0].strip()
            if name:
                cleaned.append(name)
        return cleaned

    def _roxygen_before(self, pos: int) -> Optional[str]:
        line_idx = self._pos_to_line(pos) - 2
        if line_idx < 0:
            return None
        buf = []
        while line_idx >= 0:
            line = self.lines[line_idx]
            s = line.lstrip()
            if s.startswith("#'"):
                buf.append(s[2:].lstrip())
            elif s.strip() == "":
                pass
            else:
                # stop at first non-roxygen line (don’t cross blank + NULL padding blocks)
                break
            line_idx -= 1
        if not buf:
            return None
        buf.reverse()
        return '\n'.join(buf).strip() or None

    # -------- Position / brace helpers (comment/string aware) --------

    def _build_brace_map_safely(self):
        """
        Build a map of '{' -> matching '}' while ignoring braces inside:
          - comments starting with '#'
          - single- and double-quoted strings with escapes
        """
        stack = []
        pairs = {}
        in_string: Optional[str] = None
        escape = False
        in_comment = False

        for i, ch in enumerate(self.text):
            if in_comment:
                if ch == '\n':
                    in_comment = False
                continue

            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == in_string:
                    in_string = None
                continue

            # not in string/comment
            if ch == '#':
                in_comment = True
                continue
            if ch == '"' or ch == "'":
                in_string = ch
                continue

            if ch == '{':
                stack.append(i)
            elif ch == '}':
                if stack:
                    open_i = stack.pop()
                    pairs[open_i] = i
        return pairs

    def _matching_brace_pos(self, open_brace_pos: int) -> int:
        return self._brace_map.get(open_brace_pos, len(self.text) - 1)

    def _find_next_code_brace_after(self, start: int) -> Optional[int]:
        """Find next '{' after start, skipping ones in comments/strings by scanning forward again."""
        in_string: Optional[str] = None
        escape = False
        in_comment = False
        for i in range(start, len(self.text)):
            ch = self.text[i]
            if in_comment:
                if ch == '\n':
                    in_comment = False
                continue
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == in_string:
                    in_string = None
                continue
            if ch == '#':
                in_comment = True
                continue
            if ch == '"' or ch == "'":
                in_string = ch
                continue
            if ch == '{':
                return i
        return None

    def _pos_to_line(self, pos: int) -> int:
        return self.text.count('\n', 0, max(0, pos)) + 1

    def _find_next_char_in_text(self, text: str, ch: str, start: int) -> Optional[int]:
        idx = text.find(ch, start)
        return idx if idx != -1 else None

    # For nested parsing on a slice (already delimited correctly)
    def _matching_brace_pos_in_text(self, text: str, open_idx: int) -> Optional[int]:
        in_string: Optional[str] = None
        escape = False
        in_comment = False
        depth = 0
        for i in range(open_idx, len(text)):
            ch = text[i]
            if in_comment:
                if ch == '\n':
                    in_comment = False
                continue
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch == '#':
                in_comment = True
                continue
            if ch == '"' or ch == "'":
                in_string = ch
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _matching_paren_pos_in_text(self, text: str, open_idx: int) -> Optional[int]:
        in_string: Optional[str] = None
        escape = False
        in_comment = False
        depth = 0
        for i in range(open_idx, len(text)):
            ch = text[i]
            if in_comment:
                if ch == '\n':
                    in_comment = False
                continue
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch == '#':
                in_comment = True
                continue
            if ch == '"' or ch == "'":
                in_string = ch
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i
        return None
    
    def _matching_paren_pos_global(self, open_idx: int) -> Optional[int]:
        """Given an index of '(' in self.text, return the matching ')' index,
        ignoring parentheses inside strings/comments."""
        in_string: Optional[str] = None
        escape = False
        in_comment = False
        depth = 0
        for i in range(open_idx, len(self.text)):
            ch = self.text[i]
            if in_comment:
                if ch == '\n':
                    in_comment = False
                continue
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch == '#':
                in_comment = True
                continue
            if ch == '"' or ch == "'":
                in_string = ch
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i
        return None

