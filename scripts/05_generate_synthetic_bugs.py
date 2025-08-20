#!/usr/bin/env python3
"""
Generate synthetic buggy code variants from clean code.
Uses common bug patterns to create realistic buggy versions.
"""

import argparse
import ast
import astor
import random
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugGenerator:
    """Generate buggy variants of clean code."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.bug_patterns = [
            self._off_by_one_error,
            self._missing_null_check,
            self._wrong_variable_name,
            self._wrong_operator,
            self._missing_return,
            self._wrong_condition,
            self._array_bounds_error,
            self._type_error,
            self._logic_error,
            self._missing_initialization
        ]
    
    def generate_bugs(self, code: str, num_variants: int = 3) -> List[str]:
        """Generate multiple buggy variants of the given code."""
        variants = []
        
        for i in range(num_variants):
            try:
                # Parse the code
                tree = ast.parse(code)
                
                # Apply random bug patterns
                num_bugs = random.randint(1, 3)  # 1-3 bugs per variant
                for _ in range(num_bugs):
                    pattern = random.choice(self.bug_patterns)
                    tree = pattern(tree)
                
                # Convert back to code
                buggy_code = astor.to_source(tree)
                variants.append(buggy_code)
                
            except Exception as e:
                logger.warning(f"Failed to generate bug variant {i}: {e}")
                # Fallback: simple text-based mutation
                variants.append(self._simple_text_mutation(code))
        
        return variants
    
    def _off_by_one_error(self, tree: ast.AST) -> ast.AST:
        """Introduce off-by-one errors in loops and indexing."""
        class OffByOneVisitor(ast.NodeTransformer):
            def visit_For(self, node):
                if hasattr(node, 'iter') and isinstance(node.iter, ast.Call):
                    if hasattr(node.iter.func, 'id') and node.iter.func.id == 'range':
                        if node.iter.args and isinstance(node.iter.args[0], ast.Num):
                            # Change range(n) to range(n+1) or range(n-1)
                            if random.choice([True, False]):
                                node.iter.args[0].n += 1
                            else:
                                node.iter.args[0].n = max(0, node.iter.args[0].n - 1)
                return node
            
            def visit_Subscript(self, node):
                if isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Num):
                    # Change index by ±1
                    if random.choice([True, False]):
                        node.slice.value.n += 1
                    else:
                        node.slice.value.n = max(0, node.slice.value.n - 1)
                return node
        
        return OffByOneVisitor().visit(tree)
    
    def _missing_null_check(self, tree: ast.AST) -> ast.AST:
        """Remove or modify null checks."""
        class NullCheckVisitor(ast.NodeTransformer):
            def visit_If(self, node):
                # Sometimes remove null checks
                if (isinstance(node.test, ast.Compare) and 
                    any(isinstance(comp, ast.Is) for comp in node.test.ops)):
                    if random.random() < 0.3:  # 30% chance to remove
                        return node.body[0] if node.body else None
                return node
        
        return NullCheckVisitor().visit(tree)
    
    def _wrong_variable_name(self, tree: ast.AST) -> ast.AST:
        """Introduce wrong variable names."""
        class VariableVisitor(ast.NodeTransformer):
            def __init__(self):
                self.variables = []
            
            def visit_Name(self, node):
                self.variables.append(node.id)
                return node
            
            def visit_Assign(self, node):
                if node.targets and isinstance(node.targets[0], ast.Name):
                    self.variables.append(node.targets[0].id)
                return node
        
        visitor = VariableVisitor()
        visitor.visit(tree)
        
        if len(visitor.variables) > 1:
            # Replace a random variable with another
            old_var = random.choice(visitor.variables)
            new_var = random.choice(visitor.variables)
            
            class ReplaceVisitor(ast.NodeTransformer):
                def __init__(self, old, new):
                    self.old = old
                    self.new = new
                
                def visit_Name(self, node):
                    if node.id == self.old:
                        node.id = self.new
                    return node
            
            return ReplaceVisitor(old_var, new_var).visit(tree)
        
        return tree
    
    def _wrong_operator(self, tree: ast.AST) -> ast.AST:
        """Replace operators with wrong ones."""
        operator_map = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.Div,
            ast.Div: ast.Mult,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.Gt,
            ast.Gt: ast.Lt,
            ast.LtE: ast.GtE,
            ast.GtE: ast.LtE
        }
        
        class OperatorVisitor(ast.NodeTransformer):
            def visit_BinOp(self, node):
                if type(node.op) in operator_map and random.random() < 0.3:
                    node.op = operator_map[type(node.op)]()
                return node
            
            def visit_Compare(self, node):
                for i, op in enumerate(node.ops):
                    if type(op) in operator_map and random.random() < 0.3:
                        node.ops[i] = operator_map[type(op)]()
                return node
        
        return OperatorVisitor().visit(tree)
    
    def _missing_return(self, tree: ast.AST) -> ast.AST:
        """Remove or modify return statements."""
        class ReturnVisitor(ast.NodeTransformer):
            def visit_Return(self, node):
                if random.random() < 0.2:  # 20% chance to remove
                    return None
                elif random.random() < 0.3:  # 30% chance to modify
                    if node.value and isinstance(node.value, ast.Num):
                        node.value.n = random.randint(-100, 100)
                return node
        
        return ReturnVisitor().visit(tree)
    
    def _wrong_condition(self, tree: ast.AST) -> ast.AST:
        """Modify conditions in if/while statements."""
        class ConditionVisitor(ast.NodeTransformer):
            def visit_If(self, node):
                if random.random() < 0.3:
                    # Invert condition
                    if isinstance(node.test, ast.Compare):
                        for i, op in enumerate(node.test.ops):
                            if hasattr(op, '__class__'):
                                if op.__class__ == ast.Eq:
                                    node.test.ops[i] = ast.NotEq()
                                elif op.__class__ == ast.NotEq:
                                    node.test.ops[i] = ast.Eq()
                                elif op.__class__ == ast.Lt:
                                    node.test.ops[i] = ast.Gt()
                                elif op.__class__ == ast.Gt:
                                    node.test.ops[i] = ast.Lt()
                return node
        
        return ConditionVisitor().visit(tree)
    
    def _array_bounds_error(self, tree: ast.AST) -> ast.AST:
        """Introduce array bounds errors."""
        class ArrayVisitor(ast.NodeTransformer):
            def visit_Subscript(self, node):
                if isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Num):
                    # Make index negative or very large
                    if random.choice([True, False]):
                        node.slice.value.n = -random.randint(1, 10)
                    else:
                        node.slice.value.n = random.randint(100, 1000)
                return node
        
        return ArrayVisitor().visit(tree)
    
    def _type_error(self, tree: ast.AST) -> ast.AST:
        """Introduce type errors."""
        class TypeVisitor(ast.NodeTransformer):
            def visit_BinOp(self, node):
                if random.random() < 0.2:
                    # Add string to number
                    if isinstance(node.left, ast.Num):
                        node.left = ast.Str(s=f"error_{node.left.n}")
                    elif isinstance(node.right, ast.Num):
                        node.right = ast.Str(s=f"error_{node.right.n}")
                return node
        
        return TypeVisitor().visit(tree)
    
    def _logic_error(self, tree: ast.AST) -> ast.AST:
        """Introduce logic errors."""
        class LogicVisitor(ast.NodeTransformer):
            def visit_Assign(self, node):
                if random.random() < 0.2:
                    # Assign wrong value
                    if isinstance(node.value, ast.Num):
                        node.value.n = random.randint(-100, 100)
                return node
        
        return LogicVisitor().visit(tree)
    
    def _missing_initialization(self, tree: ast.AST) -> ast.AST:
        """Remove variable initializations."""
        class InitVisitor(ast.NodeTransformer):
            def visit_Assign(self, node):
                if random.random() < 0.1:  # 10% chance to remove
                    return None
                return node
        
        return InitVisitor().visit(tree)
    
    def _simple_text_mutation(self, code: str) -> str:
        """Fallback: simple text-based mutations."""
        lines = code.split('\n')
        mutations = [
            lambda x: x.replace('+', '-'),
            lambda x: x.replace('-', '+'),
            lambda x: x.replace('==', '!='),
            lambda x: x.replace('!=', '=='),
            lambda x: x.replace('<', '>'),
            lambda x: x.replace('>', '<'),
            lambda x: x.replace('True', 'False'),
            lambda x: x.replace('False', 'True'),
            lambda x: x.replace('return', '# return'),
            lambda x: x.replace('if ', '# if '),
        ]
        
        # Apply 1-2 random mutations
        for _ in range(random.randint(1, 2)):
            mutation = random.choice(mutations)
            lines = [mutation(line) for line in lines]
        
        return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic buggy code variants")
    parser.add_argument("--input", required=True, help="Input parquet file with clean code")
    parser.add_argument("--output", required=True, help="Output parquet file for buggy variants")
    parser.add_argument("--num_variants", type=int, default=3, help="Number of buggy variants per clean code")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load clean code
    logger.info(f"Loading clean code from {args.input}")
    df = pd.read_parquet(args.input)
    
    # Filter to only clean code
    clean_df = df[df['is_buggy'] == 0].copy()
    logger.info(f"Found {len(clean_df)} clean code samples")
    
    # Initialize bug generator
    generator = BugGenerator(seed=args.seed)
    
    # Generate buggy variants
    buggy_samples = []
    
    for idx, row in clean_df.iterrows():
        logger.info(f"Generating bugs for sample {idx + 1}/{len(clean_df)}")
        
        try:
            buggy_variants = generator.generate_bugs(row['code'], args.num_variants)
            
            for i, buggy_code in enumerate(buggy_variants):
                buggy_sample = row.copy()
                buggy_sample['code'] = buggy_code
                buggy_sample['is_buggy'] = 1
                buggy_sample['id'] = f"{row['id']}_bug_{i}"
                buggy_sample['pair_id'] = f"{row['pair_id']}_synthetic_{i}"
                buggy_sample['source'] = f"{row['source']}_synthetic"
                buggy_samples.append(buggy_sample)
                
        except Exception as e:
            logger.warning(f"Failed to generate bugs for sample {idx}: {e}")
    
    # Create buggy dataframe
    buggy_df = pd.DataFrame(buggy_samples)
    
    # Save results
    logger.info(f"Generated {len(buggy_df)} buggy variants")
    buggy_df.to_parquet(args.output, index=False)
    logger.info(f"Saved buggy variants to {args.output}")
    
    # Print statistics
    print(f"\nSynthetic Bug Generation Results:")
    print(f"  Clean samples processed: {len(clean_df)}")
    print(f"  Buggy variants generated: {len(buggy_df)}")
    print(f"  Average variants per sample: {len(buggy_df) / len(clean_df):.1f}")

if __name__ == "__main__":
    main()
