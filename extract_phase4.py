import ast
import os
import shutil

# Step 1: Copy pattern_scanner.py to core/orchestrator.py
os.makedirs('core', exist_ok=True)
with open('core/__init__.py', 'w', encoding='utf-8') as f:
    f.write('')

shutil.copy('pattern_scanner.py', 'core/orchestrator.py')

# Step 2: Remove CLI functions from core/orchestrator.py
cli_funcs = [
    '_emit_dt_rejected', '_emit_db_rejected', '_vol_str',
    'print_cup_and_handle', 'print_bull_flag', 'print_bear_flag', 'print_pennant',
    'print_head_and_shoulders', 'print_double_top', 'print_double_bottom', 'print_any_pattern',
    'test_cup_and_handle', 'test_bull_flag', 'test_bear_flag', 'test_pennant',
    'test_head_and_shoulders', 'test_double_top', 'test_double_bottom', 'run_self_tests',
    'print_summary_table', 'backtest_historical', 'run_scheduler',
    'prompt_for_config', 'show_menu', 'main',
    'compute_ema', 'compute_rsi', 'compute_atr', 'interval_to_candles_per_day', 'compute_adx'
]
cli_classes = ['DualWriter']
cli_vars = ['script_dir', 'log_file', 'PRINT_DISPATCH', 'ALGO_VERSION']

def remove_nodes(filepath, funcs, classes, vars, remove_main_block=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        src = f.read()
    
    tree = ast.parse(src)
    lines_to_remove = set()
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            lines_to_remove.update(range(node.lineno, node.end_lineno + 1))
            # Also capture decorators if any
            if node.decorator_list:
                lines_to_remove.update(range(node.decorator_list[0].lineno, node.lineno))
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            lines_to_remove.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in vars:
                    lines_to_remove.update(range(node.lineno, node.end_lineno + 1))
        elif remove_main_block and isinstance(node, ast.If):
            # Check if it is `if __name__ == "__main__":`
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == '__name__':
                    lines_to_remove.update(range(node.lineno, node.end_lineno + 1))
                    
    lines = src.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if (i + 1) not in lines_to_remove:
            new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

remove_nodes('core/orchestrator.py', cli_funcs, cli_classes, cli_vars, remove_main_block=True)

# Step 3: Remove Orchestrator functions from pattern_scanner.py
orch_funcs = [
    'get_nifty_list', 'fetch_batch_data', 'get_param_hash', '_restore_dates',
    'scan_ticker', '_get_ist_now', 'is_market_open', '_reset_dedup_cache_if_new_day',
    '_is_pattern_from_today', '_process_alert_history', 'scan_watchlist',
    'validate_yfinance_params', 'compute_ema', 'compute_rsi', 'compute_atr', 'interval_to_candles_per_day', 'compute_adx'
]
orch_vars = ['ALL_PATTERNS', 'PATTERN_NAMES', 'PATTERN_SIGNALS', '_live_alerted_today', '_live_alert_date']

remove_nodes('pattern_scanner.py', orch_funcs, [], orch_vars)

# Step 4: Inject imports in pattern_scanner.py
with open('pattern_scanner.py', 'r', encoding='utf-8') as f:
    src = f.read()

import_statement = "\nfrom core.orchestrator import *\n"
# Find a good place to inject, e.g., after `from scipy.stats import linregress` or `import sys`
lines = src.split('\n')
for i, line in enumerate(lines):
    if line.startswith('import ') or line.startswith('from '):
        last_import = i

lines.insert(last_import + 1, import_statement)

with open('pattern_scanner.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Extraction successful.")
